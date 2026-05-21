from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _is_hf_hub_repo_id(raw: str) -> bool:
    """True for Hub ids like Qwen/Qwen2.5-0.5B (not filesystem paths)."""
    if "\\" in raw:
        return False
    if os.name == "nt" and len(raw) >= 2 and raw[1] == ":":
        return False
    if raw.startswith(("/", ".", "~")):
        return False
    parts = raw.split("/")
    return len(parts) == 2 and all(parts) and all(p.strip() for p in parts)


def resolve_hf_model_path(model_id: str) -> str:
    """
    Hub repo id (e.g. Qwen/Qwen2.5-0.5B) or a local directory for from_pretrained.

    On Windows, existing directories are resolved to absolute paths so
    huggingface_hub does not treat D:\\... as an invalid repo id (HFValidationError).
    """
    raw = model_id.strip()
    if _is_hf_hub_repo_id(raw):
        return raw

    candidates = [Path(raw).expanduser()]
    if os.name == "nt" and "\\" in raw:
        candidates.append(Path(raw.replace("\\", "/")).expanduser())

    for p in candidates:
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if resolved.is_dir() or resolved.is_file():
            return str(resolved)

    raise FileNotFoundError(
        f"Local model path not found: {model_id!r}\n"
        "Ensure the directory exists and contains config.json "
        "(e.g. huggingface-cli download ... --local-dir D:/hf_models/Qwen2.5-0.5B).\n"
        "For Hub models use the repo id, e.g. Qwen/Qwen2.5-0.5B (cached under "
        f"{Path.home() / '.cache' / 'huggingface' / 'hub'})."
    )


def resolve_hub_cached_snapshot(repo_id: str) -> Optional[str]:
    """Return snapshot directory if repo_id is already in the local HF hub cache."""
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
    except ImportError:
        return None
    config_path = try_to_load_from_cache(repo_id, "config.json")
    if config_path is None:
        return None
    return str(Path(config_path).parent.resolve())


def _prefer_offline_hub_load(model_id: str, load_path: str) -> tuple[str, bool]:
    """
    If weights are cached locally, load from snapshot with local_files_only=True
    so huggingface.co is not contacted (avoids timeouts without HF_ENDPOINT).
    """
    raw = model_id.strip()
    if not _is_hf_hub_repo_id(raw):
        return load_path, False

    if os.environ.get("HF_ENDPOINT", "").strip():
        return load_path, False

    cached = resolve_hub_cached_snapshot(raw)
    if cached:
        logger.info("Using local Hugging Face cache (offline): %s", cached)
        return cached, True

    logger.warning(
        "HF_ENDPOINT is not set; huggingface_hub may contact huggingface.co and time out. "
        "In PowerShell run: $env:HF_ENDPOINT = 'https://hf-mirror.com' "
        "Or pass a snapshot path as --hf-model-id (see tutorial/windows_qwen_setup.md)."
    )
    force = os.environ.get("NANOCHAT_HF_LOCAL_FILES_ONLY", "").strip().lower()
    if force in ("1", "true", "yes"):
        return load_path, True
    return load_path, False


@dataclass
class TransformersGenerateParams:
    temperature: float = 0.8
    top_k: int = 50
    max_new_tokens: int = 256


class TransformersChatBackend:
    """
    Minimal chat backend for HF Transformers causal LMs (e.g. Qwen Instruct).

    This backend is intentionally non-streaming: `generate_text()` returns the
    full assistant response as a single string.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        torch_dtype: Optional[torch.dtype] = None,
        max_layers: Optional[int] = None,
        max_context_len: Optional[int] = None,
        attn_implementation: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:  # pragma: no cover
            # Preserve the original ImportError message because it often contains
            # the exact dependency version constraint to fix.
            raise RuntimeError(
                "Failed to import transformers dependencies.\n\n"
                "Typical fixes:\n"
                "- pip install -U transformers\n"
                "- or pin a compatible huggingface-hub version (transformers will tell you).\n\n"
                f"Original error:\n{e}"
            ) from e

        load_path = resolve_hf_model_path(model_id)
        load_path, local_files_only = _prefer_offline_hub_load(model_id, load_path)
        self.model_id = load_path
        self.device = device

        pretrained_kw = dict(trust_remote_code=False, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            use_fast=True,
            **pretrained_kw,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **pretrained_kw,
        )
        self.use_cache = bool(use_cache)
        if attn_implementation is not None:
            # Best-effort: transformers respects config.attn_implementation for many decoder-only LMs.
            try:
                setattr(self.model.config, "attn_implementation", str(attn_implementation))
            except Exception:
                pass
        if max_context_len is not None:
            self._limit_context_inplace(self.model, tokenizer=self.tokenizer, max_context_len=max_context_len)
        if max_layers is not None:
            self._truncate_layers_inplace(self.model, max_layers=max_layers)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _limit_context_inplace(model: Any, tokenizer: Any, max_context_len: int) -> None:
        """
        Clamp model/tokenizer maximum context length to avoid huge cache allocations on MPS.

        Some long-context models may pre-allocate or cache RoPE tables up to the configured
        max position, which can exceed MPS temporary tensor limits (>4GB). For Mac/MPS,
        start with 2048 or 4096.
        """
        if max_context_len <= 0:
            raise ValueError(f"max_context_len must be > 0, got {max_context_len}")

        cfg = getattr(model, "config", None)
        if cfg is not None:
            for k in ("max_position_embeddings", "max_seq_len", "seq_length"):
                if hasattr(cfg, k):
                    try:
                        current = int(getattr(cfg, k))
                        setattr(cfg, k, min(current, int(max_context_len)))
                    except Exception:
                        # Some configs store None or strings; ignore.
                        pass

        # Tokenizer-side clamp (prevents warnings and accidental huge prompts)
        try:
            tokenizer.model_max_length = int(max_context_len)
        except Exception:
            pass

        # Best-effort reset of rotary caches if present.
        # Many implementations create cos/sin caches lazily and store max_seq_len_cached.
        possible_rope_paths = [
            ("model", "rotary_emb"),
            ("model", "model", "rotary_emb"),
            ("transformer", "rotary_emb"),
            ("transformer", "model", "rotary_emb"),
        ]
        for path in possible_rope_paths:
            obj = model
            ok = True
            for p in path:
                if not hasattr(obj, p):
                    ok = False
                    break
                obj = getattr(obj, p)
            if not ok:
                continue
            rope = obj
            # Reset cached length if present; forward will rebuild smaller when needed.
            for attr in ("max_seq_len_cached", "max_position_embeddings"):
                if hasattr(rope, attr):
                    try:
                        setattr(rope, attr, int(max_context_len))
                    except Exception:
                        pass
            if hasattr(rope, "cos_cached"):
                try:
                    rope.cos_cached = None
                except Exception:
                    pass
            if hasattr(rope, "sin_cached"):
                try:
                    rope.sin_cached = None
                except Exception:
                    pass

    @staticmethod
    def _truncate_layers_inplace(model: Any, max_layers: int) -> None:
        """
        Keep only the first `max_layers` transformer blocks in-place.

        This is a pragmatic way to experiment with smaller compute on-device while
        still using a HF model family. It loads full weights first, then drops
        the tail modules (GC will reclaim them).
        """
        if max_layers <= 0:
            raise ValueError(f"max_layers must be > 0, got {max_layers}")

        # Try common module layouts (LLaMA/Qwen-style, GPT2-style, etc.)
        candidates = [
            ("model", "layers"),      # LLaMA/Qwen: model.model.layers
            ("model", "h"),           # GPT-like: model.model.h
            ("transformer", "h"),     # GPT2: model.transformer.h
            (None, "layers"),         # some models: model.layers
        ]

        layer_list = None
        owner = None
        attr = None
        for owner_name, attr_name in candidates:
            try:
                obj = getattr(model, owner_name) if owner_name is not None else model
                ll = getattr(obj, attr_name)
                # nn.ModuleList behaves like list for slicing.
                if hasattr(ll, "__len__") and hasattr(ll, "__getitem__"):
                    layer_list = ll
                    owner = obj
                    attr = attr_name
                    break
            except Exception:
                continue

        if layer_list is None or owner is None or attr is None:
            raise RuntimeError(
                "Could not locate transformer layer stack to truncate. "
                "Tried: model.model.layers, model.model.h, model.transformer.h, model.layers."
            )

        num_layers = len(layer_list)
        if max_layers >= num_layers:
            return  # nothing to do

        # Slice and reassign.
        truncated = layer_list[:max_layers]
        try:
            import torch.nn as nn
            if isinstance(layer_list, nn.ModuleList):
                truncated = nn.ModuleList(list(truncated))
        except Exception:
            pass

        setattr(owner, attr, truncated)

        # Best-effort config update.
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for k in ("num_hidden_layers", "n_layer", "n_layers"):
                if hasattr(cfg, k):
                    try:
                        setattr(cfg, k, int(max_layers))
                    except Exception:
                        pass

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        # Qwen instruct models typically provide a chat template in tokenizer config.
        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            assert isinstance(ids, torch.Tensor)
            return ids

        # Fallback: naive concatenation (not ideal, but avoids hard failure).
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text += f"{role}: {content}\n"
        text += "assistant:\n"
        enc = self.tokenizer(text, return_tensors="pt")
        return enc["input_ids"]

    @torch.no_grad()
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        params: TransformersGenerateParams,
    ) -> str:
        input_ids = self._apply_chat_template(messages).to(self.device)
        # Hard safety clamp for on-device memory: never exceed tokenizer/model_max_length.
        try:
            max_len = int(getattr(self.tokenizer, "model_max_length", 0))
        except Exception:
            max_len = 0
        if max_len and input_ids.shape[-1] > max_len:
            input_ids = input_ids[:, -max_len:]
        prompt_len = int(input_ids.shape[-1])

        temperature = float(params.temperature)
        do_sample = temperature > 0.0

        gen = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=int(params.max_new_tokens),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=int(params.top_k) if do_sample and int(params.top_k) > 0 else None,
            use_cache=self.use_cache,
            pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        # gen shape: (1, prompt + new)
        new_tokens = gen[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def generate_text_iter(
        self,
        messages: List[Dict[str, str]],
        params: TransformersGenerateParams,
    ):
        """
        Blocking iterator that yields generated text chunks.

        Designed to be driven from a background thread (so it won't block an async server loop).
        """
        try:
            from transformers import TextIteratorStreamer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"TextIteratorStreamer unavailable: {e}") from e

        import threading

        input_ids = self._apply_chat_template(messages).to(self.device)
        try:
            max_len = int(getattr(self.tokenizer, "model_max_length", 0))
        except Exception:
            max_len = 0
        if max_len and input_ids.shape[-1] > max_len:
            input_ids = input_ids[:, -max_len:]

        temperature = float(params.temperature)
        do_sample = temperature > 0.0

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def _run_generate():
            self.model.generate(
                input_ids=input_ids,
                max_new_tokens=int(params.max_new_tokens),
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_k=int(params.top_k) if do_sample and int(params.top_k) > 0 else None,
                use_cache=self.use_cache,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                streamer=streamer,
            )

        t = threading.Thread(target=_run_generate, daemon=True)
        t.start()
        for text in streamer:
            # streamer yields strings (may be multiple characters at once)
            if text:
                yield text
        t.join()

