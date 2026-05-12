"""
DeepSeek-v4 style model adapted to nanochat training/eval/inference contracts.

Design goals for this implementation:
- Keep the same public interface as nanochat.gpt.GPT so all existing scripts work.
- Reuse nanochat's flash attention wrapper and optimizer stack.
- Bring in DeepSeek-v4 style ingredients (MLA + MoE) in a trainable form.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
from nanochat.flash_attention import flash_attn
from nanochat.gpt import GPTConfig, Linear
from nanochat.optim import DistMuonAdamW, MuonAdamW


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, D), cos/sin: (1, T, 1, D/2)
    if x.size(-1) == 0:
        return x
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


@dataclass
class DSV4Config(GPTConfig):
    # DeepSeek-v4 style attention / MoE knobs.
    n_dense_layers: int = 1
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: str = "softmax"  # "softmax" or "sigmoid"
    route_scale: float = 1.0

    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 64

    # YaRN-style rope extension controls (lightweight approximation).
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 1.0

    # Logit soft cap for stability (same spirit as GPT impl in this repo).
    softcap: float = 20.0


class DSV4MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, bias=False)
        self.w2 = Linear(inter_dim, dim, bias=False)
        self.w3 = Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DSV4Gate(nn.Module):
    def __init__(self, config: DSV4Config):
        super().__init__()
        self.topk = config.n_activated_experts
        self.n_experts = config.n_routed_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.proj = Linear(config.n_embd, config.n_routed_experts, bias=False)

        assert self.n_experts > 0, "n_routed_experts must be positive"
        assert 1 <= self.topk <= self.n_experts, "n_activated_experts must be in [1, n_routed_experts]"
        assert self.score_func in {"softmax", "sigmoid"}
        assert self.n_groups >= 1 and self.n_experts % self.n_groups == 0
        assert 1 <= self.topk_groups <= self.n_groups

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, D), returns weights/indices of shape (N, topk)
        scores = self.proj(x).float()
        if self.score_func == "softmax":
            probs = scores.softmax(dim=-1)
        else:
            probs = scores.sigmoid()

        routed_scores = probs
        if self.n_groups > 1:
            grouped = routed_scores.view(x.size(0), self.n_groups, -1)
            group_scores = grouped.amax(dim=-1)
            top_group_idx = group_scores.topk(self.topk_groups, dim=-1).indices
            group_mask = torch.ones(x.size(0), self.n_groups, dtype=torch.bool, device=x.device)
            group_mask.scatter_(1, top_group_idx, False)
            grouped = grouped.masked_fill(group_mask.unsqueeze(-1), float("-inf"))
            routed_scores = grouped.flatten(1)

        indices = torch.topk(routed_scores, self.topk, dim=-1).indices
        weights = probs.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights * self.route_scale
        return weights.type_as(x), indices


class DSV4MoE(nn.Module):
    def __init__(self, config: DSV4Config, inter_dim: int):
        super().__init__()
        self.dim = config.n_embd
        self.n_experts = config.n_routed_experts
        self.gate = DSV4Gate(config)
        self.experts = nn.ModuleList([DSV4MLP(config.n_embd, inter_dim) for _ in range(self.n_experts)])
        shared_inter_dim = max(1, config.n_shared_experts) * inter_dim
        self.shared_experts = DSV4MLP(config.n_embd, shared_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        weights, indices = self.gate(x_flat)

        y = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            selected = indices == expert_id
            if not torch.any(selected):
                continue
            token_idx, topk_slot = torch.where(selected)
            expert_out = expert(x_flat[token_idx])
            y[token_idx] += expert_out * weights[token_idx, topk_slot].unsqueeze(-1)

        y = y + self.shared_experts(x_flat)
        return y.view(shape)


class DSV4MLAAttention(nn.Module):
    def __init__(self, config: DSV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        rope_dim = min(config.qk_rope_head_dim, self.head_dim)
        rope_dim = max(2, rope_dim - (rope_dim % 2))
        self.rope_dim = rope_dim
        self.nope_dim = self.head_dim - self.rope_dim

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank if config.kv_lora_rank > 0 else max(64, self.n_embd // 4)

        if self.q_lora_rank > 0:
            self.wq_a = Linear(self.n_embd, self.q_lora_rank, bias=False)
            self.wq_b = Linear(self.q_lora_rank, self.n_head * self.head_dim, bias=False)
        else:
            self.wq = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)

        self.wkv_a = Linear(self.n_embd, self.kv_lora_rank + self.rope_dim, bias=False)
        self.wkv_b = Linear(self.kv_lora_rank, self.n_kv_head * (self.nope_dim + self.head_dim), bias=False)
        self.wo = Linear(self.n_head * self.head_dim, self.n_embd, bias=False)

        self.softmax_scale = self.head_dim ** -0.5
        if config.sequence_len > config.original_seq_len and config.rope_factor > 1.0:
            # Matches DeepSeek-v4's high-level idea: larger context benefits from a scale correction.
            mscale = 0.1 * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank > 0:
            q = self.wq_b(norm(self.wq_a(x)))
        else:
            q = self.wq(x)
        return q

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        kv_cache,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        q = self._project_q(x).view(bsz, seqlen, self.n_head, self.head_dim)
        q_nope = q[..., :self.nope_dim] if self.nope_dim > 0 else q[..., :0]
        q_pe = q[..., self.nope_dim:]

        kv = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.rope_dim], dim=-1)
        kv_latent = norm(kv_latent)
        kv = self.wkv_b(kv_latent).view(bsz, seqlen, self.n_kv_head, self.nope_dim + self.head_dim)
        k_nope, v = torch.split(kv, [self.nope_dim, self.head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.n_kv_head, -1)

        cos, sin = cos_sin
        q_pe = apply_rotary_emb(q_pe, cos, sin)
        k_pe = apply_rotary_emb(k_pe, cos, sin)

        q = torch.cat([q_nope, q_pe], dim=-1) if self.nope_dim > 0 else q_pe
        k = torch.cat([k_nope, k_pe], dim=-1) if self.nope_dim > 0 else k_pe

        # QK norm (same spirit as nanochat's GPT implementation).
        q = norm(q) * 1.1
        k = norm(k) * 1.1

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(seqlen)

        y = y.contiguous().view(bsz, seqlen, self.n_head * self.head_dim)
        return self.wo(y)


class DSV4Block(nn.Module):
    def __init__(self, config: DSV4Config, layer_idx: int):
        super().__init__()
        self.attn = DSV4MLAAttention(config, layer_idx)
        inter_dim = 4 * config.n_embd
        if layer_idx < config.n_dense_layers:
            self.ffn = DSV4MLP(config.n_embd, inter_dim)
        else:
            self.ffn = DSV4MoE(config, inter_dim=max(config.n_embd, inter_dim // 3))

    def forward(self, x: torch.Tensor, cos_sin, window_size, kv_cache) -> torch.Tensor:
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        x = x + self.ffn(norm(x))
        return x


class DSV4(nn.Module):
    def __init__(self, config: DSV4Config, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([DSV4Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # Rope is applied only on q/k rope channels inside MLA.
        self.rope_dim = self.transformer.h[0].attn.rope_dim
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.rope_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = math.sqrt(3.0) * (n_embd ** -0.5)

        def init_swiglu(m: DSV4MLP, scale: float = 1.0):
            torch.nn.init.uniform_(m.w1.weight, -s * scale, s * scale)
            torch.nn.init.uniform_(m.w3.weight, -s * scale, s * scale)
            torch.nn.init.zeros_(m.w2.weight)

        for block in self.transformer.h:
            attn = block.attn
            if hasattr(attn, "wq"):
                torch.nn.init.uniform_(attn.wq.weight, -s, s)
            else:
                torch.nn.init.uniform_(attn.wq_a.weight, -s, s)
                torch.nn.init.uniform_(attn.wq_b.weight, -s, s)
            torch.nn.init.uniform_(attn.wkv_a.weight, -s, s)
            torch.nn.init.uniform_(attn.wkv_b.weight, -s, s)
            torch.nn.init.zeros_(attn.wo.weight)

            if isinstance(block.ffn, DSV4MLP):
                init_swiglu(block.ffn, scale=0.5)
            else:
                init_swiglu(block.ffn.shared_experts, scale=0.5)
                torch.nn.init.uniform_(block.ffn.gate.proj.weight, 0.0, 0.02)
                for expert in block.ffn.experts:
                    init_swiglu(expert, scale=0.5)

        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.rope_dim)
        self.cos, self.sin = cos, sin

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len: int, rope_dim: int, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
        )

        # Lightweight YaRN-style extension: downscale frequencies when extending context.
        if seq_len > self.config.original_seq_len and self.config.rope_factor > 1.0:
            inv_freq = inv_freq / self.config.rope_factor

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().to(COMPUTE_DTYPE), freqs.sin().to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config: DSV4Config) -> list[tuple[int, int]]:
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."

        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }

        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_scaling_params(self):
        wte = 0
        lm_head = 0
        dense_matrices = 0
        moe_experts = 0
        moe_gates = 0
        scalars = 0

        for name, p in self.named_parameters():
            n = p.numel()
            if name.startswith("transformer.wte"):
                wte += n
            elif name.startswith("lm_head"):
                lm_head += n
            elif ".gate.proj." in name:
                moe_gates += n
            elif ".experts." in name and ".shared_experts." not in name:
                moe_experts += n
            elif p.ndim >= 2:
                dense_matrices += n
            else:
                scalars += n

        transformer_matrices = dense_matrices + moe_experts + moe_gates
        total = wte + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            "wte": wte,
            "lm_head": lm_head,
            "dense_matrices": dense_matrices,
            "moe_experts": moe_experts,
            "moe_gates": moe_gates,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def estimate_flops(self):
        counts = self.num_scaling_params()
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len

        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        # Routed MoE experts are sparsely activated at runtime.
        routed = counts["moe_experts"]
        active_frac = self.config.n_activated_experts / max(1, self.config.n_routed_experts)
        effective_matrices = counts["dense_matrices"] + counts["moe_gates"] + routed * active_frac

        return 6 * (effective_matrices + counts["lm_head"]) + attn_flops

    def setup_optimizer(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        scalar_lr: float = 0.5,
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        embedding_params = []
        lm_head_params = []
        matrix_params = []
        scalar_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("transformer.wte"):
                embedding_params.append(p)
            elif name.startswith("lm_head"):
                lm_head_params.append(p)
            elif p.ndim >= 2:
                matrix_params.append(p)
            else:
                scalar_params.append(p)

        assert len(list(self.parameters())) == (
            len(embedding_params) + len(lm_head_params) + len(matrix_params) + len(scalar_params)
        )

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.96),
                eps=1e-10,
                weight_decay=0.01,
            ),
            dict(
                kind="adamw",
                params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.001,
            ),
        ]

        if scalar_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=scalar_params,
                    lr=scalar_lr * 0.05,
                    betas=(0.9, 0.95),
                    eps=1e-10,
                    weight_decay=0.0,
                )
            )

        for shape in sorted({tuple(p.shape) for p in matrix_params}):
            group_params = [p for p in matrix_params if tuple(p.shape) == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.9,
                    weight_decay=weight_decay,
                )
            )

        factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        bsz, seqlen = idx.size()
        assert seqlen <= self.cos.size(1), f"Sequence length exceeded rope cache: {seqlen} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary cache and idx must share device: {idx.device} vs {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary cache must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"

        t0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, t0 : t0 + seqlen], self.sin[:, t0 : t0 + seqlen]

        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)

        for i, block in enumerate(self.transformer.h):
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)

        x = norm(x)
        logits = self.lm_head(x)[..., : self.config.vocab_size].float()

        softcap = self.config.softcap
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)[:, -1, :]
            if top_k is not None and top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_ids], dim=1)
            yield next_ids.item()
