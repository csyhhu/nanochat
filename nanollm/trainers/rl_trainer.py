"""
RLTrainer — Reinforcement Learning Trainer (GRPO/REINFORCE)

This module implements reinforcement learning training based on GRPO (Group Relative Policy Optimization).

Design goals:
1. Allow users to clearly understand each step of RL training
2. Provide clear interfaces for users to implement GRPO and other algorithms
3. Modular design: data preparation, rollout generation, advantage computation, 
   loss calculation, and evaluation are separated

Core concepts:
- Rollout: Model generates multiple complete responses for a given prompt
- Reward: Evaluates generation quality based on task-defined reward function
- Advantage: Signal used for policy gradient (GRPO uses group relative advantage)
- Policy Gradient: Updates model by maximizing expected reward

GRPO core ideas:
1. For each prompt, generate G rollouts (num_samples)
2. Calculate reward for each rollout
3. Compute group relative advantage: A_i = (r_i - mean(R)) / std(R) [optional normalization]
4. Update model using policy gradient: loss = -mean(log_prob * advantage)

Differences from PPO:
- No value network (critic)
- No KL divergence constraint
- Uses within-group relative advantage instead of GAE

Usage example:
    from nanollm.trainers.rl_trainer import RLTrainer
    from nanollm.main import TrainConfig
    
    cfg = TrainConfig(stage="rl", model_id="Qwen/Qwen2.5-0.5B", ...)
    trainer = RLTrainer(cfg)
    trainer.run()
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from nanollm.trainers.base import BaseTrainer, TrainConfig


# =============================================================================
# CPU Optimization Utilities
# =============================================================================

def optimize_cpu_performance():
    """
    Optimize CPU performance for PyTorch.
    
    This function should be called at the start of training to maximize
    CPU inference and training speed.
    
    Optimizations:
    1. Set thread count to use all available CPU cores
    2. Enable MKL/DNNL optimizations (if available)
    3. Set memory allocation strategy
    """
    num_threads = os.cpu_count() or 4
    
    # Set PyTorch threads
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    
    # Set environment variables for libraries
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    
    # Enable PyTorch optimizations
    torch.backends.cudnn.benchmark = False  # Not relevant for CPU
    if hasattr(torch.backends, "mkl"):
        torch.backends.mkl.is_available()
    
    print(f"  [CPU Optimize] Using {num_threads} threads for CPU inference")
    print(f"  [CPU Optimize] OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")


def apply_dynamic_quantization(model):
    """
    Apply dynamic quantization to model for faster CPU inference.
    
    Dynamic quantization quantizes weights to int8, while activations
    are quantized on-the-fly. This can speed up inference by 2-4x on CPU.
    
    Args:
        model: PyTorch model
    
    Returns:
        Quantized model
    """
    print(f"  [Quantization] Applying dynamic quantization...")
    
    # Quantize the model
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize all Linear layers
        dtype=torch.qint8,
    )
    
    # Calculate size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    
    print(f"  [Quantization] [OK] Dynamic quantization applied")
    print(f"  [Quantization] Original size: {original_size / 1024**2:.1f} MB")
    print(f"  [Quantization] Quantized size: {quantized_size / 1024**2:.1f} MB")
    
    return quantized_model


def compile_model_for_fast_inference(model, mode: str = "default"):
    """
    Compile model with torch.compile() for faster inference.
    
    Args:
        model: PyTorch model
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
    
    Returns:
        Compiled model (or original if compilation fails)
    """
    if not hasattr(torch, "compile"):
        print(f"  [Compile] torch.compile not available (requires PyTorch 2.0+)")
        return model
    
    try:
        print(f"  [Compile] Compiling model with mode='{mode}'...")
        print(f"  [Compile] (First run will be slow, subsequent runs will be fast)")
        
        compiled_model = torch.compile(model, mode=mode, fullgraph=False, dynamic=True)
        
        print(f"  [Compile] [OK] Model compiled successfully")
        return compiled_model
    except Exception as e:
        print(f"  [Compile] [FAIL] Compilation failed: {e}")
        print(f"  [Compile] Using uncompiled model")
        return model


# =============================================================================
# vLLM Support — optional fast rollout backend
# =============================================================================
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM, SamplingParams = None, None


@dataclass
class StepTimer:
    """Step-level timer: records time for each training stage."""
    rollout_time: float = 0.0
    backward_time: float = 0.0
    update_time: float = 0.0
    eval_time: float = 0.0
    total_time: float = 0.0
    
    def reset(self):
        self.rollout_time = 0.0
        self.backward_time = 0.0
        self.update_time = 0.0
        self.eval_time = 0.0
        self.total_time = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "rollout_s": round(self.rollout_time, 3),
            "backward_s": round(self.backward_time, 3),
            "update_s": round(self.update_time, 3),
            "eval_s": round(self.eval_time, 3),
            "total_s": round(self.total_time, 3),
        }
    
    def format(self) -> str:
        parts = []
        if self.rollout_time > 0:
            parts.append(f"rollout={self.rollout_time:.2f}s")
        if self.backward_time > 0:
            parts.append(f"backward={self.backward_time:.2f}s")
        if self.update_time > 0:
            parts.append(f"update={self.update_time:.2f}s")
        if self.eval_time > 0:
            parts.append(f"eval={self.eval_time:.2f}s")
        parts.append(f"total={self.total_time:.2f}s")
        return " | ".join(parts)


# =============================================================================
# 1. Data Preparation Module
# =============================================================================

class RLTaskDataset(Dataset):
    """
    RL Task Dataset
    
    Each sample is a conversation dict containing:
    - messages: Conversation history [{"role": "user", "content": ...}, ...]
    
    RL tasks must implement reward() method:
    def reward(self, conversation, assistant_response) -> float
    """
    
    def __init__(self, task_name: str, split: str = "train"):
        self.task_name = task_name
        self.split = split
        self.task = self._create_task(task_name, split)
        self.examples = list(range(len(self.task)))
    
    def _create_task(self, task_name: str, split: str):
        """Create task object that supports reward()"""
        if task_name == "GSM8K":
            from tasks.gsm8k import GSM8K
            return GSM8K(subset="main", split=split)
        elif task_name == "SpellingBee":
            from tasks.spellingbee import SpellingBee
            return SpellingBee(size=256, split=split)
        elif task_name == "MMLU":
            from tasks.mmlu import MMLU
            return MMLU(subset="all", split=split)
        else:
            raise ValueError(
                f"Task '{task_name}' does not support reward(). "
                f"Available RL tasks: GSM8K, SpellingBee, MMLU"
            )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Returns conversation dict"""
        return self.task[self.examples[idx]]


class RLDataPreparer:
    """
    RL Data Preparer
    
    Functions:
    1. Load RL tasks (GSM8K, SpellingBee, etc.)
    2. Create training data iterator
    3. Support multi-task mixed training
    """
    
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_datasets = []
        self.eval_datasets = []
    
    def prepare_train_data(self):
        """Prepare training data"""
        print(f"\n[RL Data] Loading training tasks: {self.cfg.train_task_list}")
        
        for task_name in self.cfg.train_task_list:
            dataset = RLTaskDataset(task_name, split="train")
            self.train_datasets.append(dataset)
            print(f"  {task_name}: {len(dataset)} examples (train split)")
        
        print(f"  Total training tasks: {len(self.train_datasets)}")
        return self.train_datasets
    
    def prepare_eval_data(self):
        """Prepare evaluation data"""
        print(f"\n[RL Data] Loading eval tasks: {self.cfg.eval_task_list}")
        
        for task_name in self.cfg.eval_task_list:
            dataset = RLTaskDataset(task_name, split="test")
            self.eval_datasets.append(dataset)
            print(f"  {task_name}: {len(dataset)} examples (test split)")
        
        return self.eval_datasets
    
    def create_train_iterator(self):
        """
        Create training data iterator
        
        Returns: Infinite iterator that yields one batch of rollout data at a time
        """
        # Create cyclic iterator for each task
        task_iters = []
        for dataset in self.train_datasets:
            task_iters.append(itertools.cycle(range(len(dataset))))
        
        # Round-robin scheduling: alternate between tasks
        task_idx = 0
        
        while True:
            current_dataset = self.train_datasets[task_idx % len(self.train_datasets)]
            current_iter = task_iters[task_idx % len(self.train_datasets)]
            task_idx += 1
            
            # Get one sample
            sample_idx = next(current_iter)
            yield current_dataset[sample_idx], current_dataset.task


# =============================================================================
# 2. Rollout Generation Module
# =============================================================================

class RolloutGenerator:
    """
    Rollout Generator
    
    Functions:
    1. Generate multiple complete responses (rollouts) for a given prompt
    2. Calculate reward for each rollout
    3. Return data for training: (input_ids, attention_mask, rewards, advantages)
    """
    
    def __init__(self, model, tokenizer, cfg: TrainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(model.parameters()).device
    
    def render_prompt(self, conversation: Dict) -> List[int]:
        """
        Render prompt for generation
        
        Steps:
        1. Deep copy conversation
        2. Remove last assistant message (if any)
        3. Add generation prompt
        
        Returns: token ids list
        """
        import copy
        conv = copy.deepcopy(conversation)
        messages = conv["messages"]
        
        # Remove last assistant message
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()
        
        # Use chat template for rendering
        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze(0).tolist()
            return ids
        
        # Fallback: manual rendering
        text_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        text_parts.append("<|im_start|>assistant\n")
        text = "".join(text_parts)
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    @torch.no_grad()
    def generate_rollouts(
        self,
        conversation: Dict,
        task,
        num_samples: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Generate multiple rollouts for a prompt (optimized with batch generation)
        
        Performance optimization:
        - Uses batch generation (num_return_sequences) instead of sequential generation
        - This is 3-10x faster on CPU compared to generating one by one
        
        Args:
            conversation: Conversation history
            task: Task object (used to calculate reward)
            num_samples: Number of rollouts to generate (G in GRPO)
            max_new_tokens: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            generated_sequences: List[List[int]], each element is a complete token sequence (prompt + generation)
            rewards: List[float], reward for each rollout
        """
        # Render prompt
        prompt_ids = self.render_prompt(conversation)
        prompt_len = len(prompt_ids)
        
        # Convert prompt to tensor (batch size = 1)
        device = self.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        # Generation parameters
        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample and top_k > 0 else None,
            num_return_sequences=num_samples,  # Generate all at once (batch mode)
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        self.model.eval()  # Use eval mode for rollout generation
        
        # Batch generation: generate all samples in one forward pass
        t0 = time.perf_counter()
        gen = self.model.generate(input_ids=input_ids, **gen_kwargs)
        gen_time = time.perf_counter() - t0
        
        # Parse outputs
        generated_sequences = []
        rewards = []
        
        for i in range(num_samples):
            seq = gen[i].tolist()
            generated_sequences.append(seq)
            
            # Decode generated part
            gen_tokens = seq[prompt_len:]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Calculate reward
            reward = task.reward(conversation, gen_text)
            rewards.append(reward)
        
        print(f"  [Rollout] Generated {num_samples} rollouts in {gen_time:.2f}s (batch mode, {gen_time/num_samples:.2f}s per sample)")
        
        return generated_sequences, rewards
    
    def compute_advantages(
        self,
        rewards: List[float],
        method: str = "grpo",
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Compute advantage
        
        Core step in GRPO: Convert rewards to advantage signals
        
        Args:
            rewards: List[float], reward for each rollout
            method: Advantage computation method
                - "naive": advantage = reward - mean(reward)  (original GRPO)
                - "grpo": advantage = (reward - mean(reward)) / std(reward)  (optional normalization)
            normalize: Whether to normalize advantages (z-score)
        
        Returns:
            advantages: torch.Tensor, shape (num_samples,)
        
        Note:
        - Original GRPO paper uses method="naive" (no normalization)
        - Normalization can stabilize training but is not required
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float, device=self.device)
        
        if method == "naive":
            # Original GRPO: advantage = reward - mean(reward)
            advantages = rewards_t - rewards_t.mean()
        
        elif method == "grpo":
            # GRPO with normalization: advantage = (reward - mean) / std
            mean_reward = rewards_t.mean()
            std_reward = rewards_t.std()
            if std_reward > 0:
                advantages = (rewards_t - mean_reward) / (std_reward + 1e-8)
            else:
                advantages = rewards_t - mean_reward
        
        else:
            raise ValueError(f"Unknown advantage method: {method}")
        
        # Optional z-score normalization
        if normalize:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 0:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        return advantages


# =============================================================================
# 2.1 vLLM Rollout Generator (optional fast backend)
# =============================================================================

class vLLMRolloutGenerator:
    """
    Fast rollout generator using vLLM.
    
    Replaces the slow Transformers generate() with vLLM's continuous batching.
    Speedup: 5-10x on GPU.
    
    Usage:
        gen = vLLMRolloutGenerator("Qwen/Qwen2.5-0.5B")
        sequences, rewards = gen.generate_rollouts(conversation, task, num_samples=8)
    """
    
    def __init__(self, model_path: str, tokenizer, tensor_parallel_size: int = 1):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Note: vLLM requires GPU and CUDA."
            )
        
        self.tokenizer = tokenizer
        self.model_path = model_path
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,  # Required for Qwen
            gpu_memory_utilization=0.85,
            dtype="float16",  # vLLM works best with FP16
        )
        
        # vLLM doesn't have a tokenizer, use the original one
        print(f"  [vLLM] Initialized with model: {model_path}")
        print(f"  [vLLM] Tensor parallel size: {tensor_parallel_size}")
    
    def render_prompt(self, conversation: Dict) -> str:
        """
        Render prompt as text (vLLM expects text, not token ids).
        
        Returns: prompt text string
        """
        import copy
        conv = copy.deepcopy(conversation)
        messages = conv["messages"]
        
        # Remove last assistant message
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()
        
        # Use chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return text
        
        # Fallback: manual rendering
        text_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        text_parts.append("<|im_start|>assistant\n")
        return "".join(text_parts)
    
    def generate_rollouts(
        self,
        conversation: Dict,
        task,
        num_samples: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Generate multiple rollouts using vLLM.
        
        Args:
            conversation: Conversation history
            task: Task object (used to calculate reward)
            num_samples: Number of rollouts to generate (G in GRPO)
            max_new_tokens: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            generated_sequences: List[List[int]], each element is a complete token sequence
            rewards: List[float], reward for each rollout
        """
        # Render prompt as text
        prompt_text = self.render_prompt(conversation)
        
        # Setup sampling parameters
        params = SamplingParams(
            n=num_samples,  # Generate num_samples per prompt
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate (vLLM automatically batches across prompts)
        t0 = time.perf_counter()
        outputs = self.llm.generate([prompt_text], params)
        gen_time = time.perf_counter() - t0
        
        # Parse outputs
        generated_sequences = []
        rewards = []
        
        for output in outputs:
            for i, sample in enumerate(output.outputs):
                # Decode tokens to text
                gen_text = sample.text
                
                # Encode full sequence (prompt + generation) as tokens
                full_text = prompt_text + gen_text
                full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
                
                generated_sequences.append(full_tokens)
                
                # Calculate reward
                reward = task.reward(conversation, gen_text)
                rewards.append(reward)
        
        print(f"  [vLLM] Generated {len(generated_sequences)} rollouts in {gen_time:.2f}s")
        
        return generated_sequences, rewards


# =============================================================================
# 3. Loss Calculation Module
# =============================================================================

class GRPOLossCalculator:
    """
    GRPO Loss Calculator
    
    Core of GRPO: Policy Gradient with Group Relative Advantage
    
    Loss formula:
        loss = -mean(log_prob * advantage)
    
    Where:
    - log_prob: Log probability of tokens generated by the model
    - advantage: Group relative advantage (reward - mean(reward))
    
    Implementation steps:
    1. Forward pass to get logits
    2. Calculate log probability for each token
    3. Only calculate loss for generated part (mask out prompt part)
    4. Weight by advantage
    5. Take negative mean as loss (because we want to maximize expected reward)
    """
    
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
    
    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate log probability for each token
        
        Args:
            model: Language model
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len), optional
        
        Returns:
            log_probs: (batch_size, seq_len), log probability at each position
            Note: log_prob for prompt part will be masked out (when used in training)
        """
        # Forward pass to get logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        # Calculate log softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def gather_token_log_probs(
        self,
        log_probs: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather log probability of target tokens
        
        Args:
            log_probs: (batch_size, seq_len, vocab_size)
            target_ids: (batch_size, seq_len), target token ids
                       -1 means ignore this position
        
        Returns:
            gathered_log_probs: (batch_size, seq_len), log prob of target token at each position
        """
        # Replace -1 in target_ids with 0 (to avoid index error)
        target_ids_clamped = target_ids.clamp(min=0).unsqueeze(-1)
        
        # gather: Extract log prob of target token from log_probs
        gathered = torch.gather(log_probs, dim=-1, index=target_ids_clamped)
        gathered = gathered.squeeze(-1)  # (batch_size, seq_len)
        
        # Set log_prob to 0 for positions where target_ids is -1 (will be ignored by mask later)
        mask = (target_ids >= 0).float()
        gathered = gathered * mask
        
        return gathered
    
    def compute_loss(
        self,
        model,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        advantages: torch.Tensor,
        generation_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate GRPO loss
        
        Args:
            model: Language model
            input_ids: (batch_size, seq_len), complete sequence (prompt + generation)
            target_ids: (batch_size, seq_len), target token ids
                       -1 means ignore this position (prompt part and padding)
            advantages: (batch_size,), advantage for each sequence
            generation_mask: (batch_size, seq_len), 1 for generated part, 0 for prompt/padding
        
        Returns:
            loss: scalar tensor
            metrics: dict, containing auxiliary information (mean_reward, mean_advantage, etc.)
        """
        # 1. Calculate log probs
        log_probs = self.compute_log_probs(model, input_ids)
        
        # 2. Gather log prob of target tokens
        token_log_probs = self.gather_token_log_probs(log_probs, target_ids)
        
        # 3. Only calculate log_prob for generated part (mask out prompt)
        generation_log_probs = token_log_probs * generation_mask
        
        # 4. Calculate total log_prob for each sequence (sum over sequence)
        seq_log_probs = generation_log_probs.sum(dim=-1)  # (batch_size,)
        
        # 5. Weight by advantage
        #    Note: Here we use the same advantage for the entire sequence's log_prob
        #    Alternatively, you can use token-level advantage (more complex)
        weighted_log_probs = seq_log_probs * advantages
        
        # 6. Policy gradient loss
        #    We want to maximize expected reward, so loss = -mean(weighted_log_probs)
        loss = -weighted_log_probs.mean()
        
        # 7. Calculate auxiliary metrics
        metrics = {
            "mean_reward": advantages.mean().item() + 1.0,  # Restore original reward mean
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
            "mean_log_prob": seq_log_probs.mean().item(),
        }
        
        return loss, metrics
    
    def compute_loss_token_level(
        self,
        model,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        advantages: torch.Tensor,
        generation_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Token-level GRPO loss (DAPO style)
        
        Difference from sequence-level:
        - Each token uses the same advantage (simple)
        - Alternatively: Use token-level advantage (more complex, requires baseline)
        
        Here we implement the simple version: each token uses sequence-level advantage
        """
        # 1. Calculate log probs
        log_probs = self.compute_log_probs(model, input_ids)
        
        # 2. Gather log prob of target tokens
        token_log_probs = self.gather_token_log_probs(log_probs, target_ids)
        
        # 3. Only calculate for generated part
        generation_token_log_probs = token_log_probs * generation_mask
        
        # 4. Expand advantages to token level
        #    advantages: (batch_size,) -> (batch_size, 1) -> broadcast to (batch_size, seq_len)
        advantages_expanded = advantages.unsqueeze(-1)  # (batch_size, 1)
        
        # 5. Token-level weighting
        weighted_token_log_probs = generation_token_log_probs * advantages_expanded
        
        # 6. Calculate number of valid tokens
        num_valid_tokens = generation_mask.sum().clamp(min=1)
        
        # 7. Token-level loss
        loss = -weighted_token_log_probs.sum() / num_valid_tokens
        
        # 8. Calculate auxiliary metrics
        metrics = {
            "mean_reward": advantages.mean().item() + 1.0,
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
            "num_valid_tokens": num_valid_tokens.item(),
        }
        
        return loss, metrics


# =============================================================================
# 4. Evaluation Module
# =============================================================================

class RLEvaluator:
    """
    RL Evaluator
    
    Two evaluation methods:
    1. Reward-based eval: Calculate pass@k (on training tasks)
    2. ChatCORE eval: Evaluate on standard benchmarks (MMLU, GSM8K, etc.)
    """
    
    def __init__(self, model, tokenizer, cfg: TrainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def evaluate_pass_at_k(
        self,
        task_name: str,
        split: str = "test",
        max_examples: int = 100,
        k: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 50,
    ) -> Dict[str, float]:
        """
        Calculate pass@k metric
        
        Definition of pass@k:
        - For each question, generate k responses
        - If at least 1 response is correct, count as success
        - pass@k = number of successes / total number of questions
        
        Args:
            task_name: Task name
            split: Dataset split
            max_examples: Maximum number of evaluation samples
            k: Generate k responses for each question
            max_new_tokens: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            metrics: dict, containing pass@{1..k}
        """
        print(f"\n  [Eval] pass@{k} on {task_name} ({split} split)")
        
        # Create task
        from nanollm.trainers.rl_trainer import RLTaskDataset
        dataset = RLTaskDataset(task_name, split=split)
        num_examples = min(max_examples, len(dataset))
        
        # Create rollout generator
        rollout_gen = RolloutGenerator(self.model, self.tokenizer, self.cfg)
        
        # Calculate pass@k
        passk_counts = [0] * k  # pass@1, pass@2, ..., pass@k
        
        for idx in range(num_examples):
            conversation = dataset[idx]
            prompt_ids = rollout_gen.render_prompt(conversation)
            
            # Generate k rollouts
            generated_sequences, _ = rollout_gen.generate_rollouts(
                conversation,
                dataset.task,
                num_samples=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            
            # Calculate reward for each rollout
            outcomes = []
            for seq in generated_sequences:
                gen_tokens = seq[len(prompt_ids):]
                gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                reward = dataset.task.reward(conversation, gen_text)
                outcomes.append(reward > 0.0)
            
            # pass@i = at least one correct in first i samples
            for i in range(1, k + 1):
                if any(outcomes[:i]):
                    passk_counts[i - 1] += 1
        
        # Calculate pass@k metrics
        results = {}
        for i in range(1, k + 1):
            results[f"pass@{i}"] = passk_counts[i - 1] / num_examples
        
        # Print results
        for metric_name, value in results.items():
            print(f"    {metric_name}: {value:.4f}")
        
        return results
    
    @torch.no_grad()
    def evaluate_chatcore(
        self,
        eval_tasks: List[str],
        max_problems: int = 200,
    ) -> Dict[str, float]:
        """
        Evaluate on ChatCORE benchmark
        
        Args:
            eval_tasks: List of task names (e.g., ["MMLU", "GSM8K", "ARC-Easy"])
            max_problems: Maximum number of problems to evaluate per task
        
        Returns:
            results: dict, {task_name: accuracy}
        """
        print(f"\n  [Eval] ChatCORE on {eval_tasks} (max {max_problems} problems each)")
        
        try:
            from scripts.qwen_eval import run_qwen_eval
        except ImportError:
            print("    WARNING: Cannot import qwen_eval. Skipping ChatCORE eval.", flush=True)
            return {}
        
        # Set model to eval mode
        self.model.eval()
        
        # Run evaluation
        results = run_qwen_eval(
            task_names=eval_tasks,
            model=self.model,
            tokenizer=self.tokenizer,
            mode="rl",
            batch_size=4,
            num_samples=1,
            max_new_tokens=256,
            temperature=0.0,  # greedy decoding for eval
            top_k=50,
            max_problems=max_problems,
        )
        
        # Print results
        for task_name, acc in results.items():
            print(f"    {task_name}: {100*acc:.2f}%")
        
        return results


# =============================================================================
# 5. RLTrainer Main Class
# =============================================================================

class RLTrainer(BaseTrainer):
    """
    RL Trainer
    
    Inherits from BaseTrainer, implements three abstract methods: 
    prepare_data(), _train(), _eval()
    
    Training flow:
    1. prepare_data(): Load RL tasks
    2. _train(): 
       a. For each step:
           - Sample prompt
           - Generate rollouts (num_samples)
           - Calculate reward
           - Calculate advantage
           - Calculate loss
           - Backpropagation
       b. Periodic eval
       c. Periodic checkpoint saving
    3. _eval(): Run reward-based eval and ChatCORE eval
    """
    
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        self.rollout_generator = None
        self.loss_calculator = None
        self.evaluator = None
        self.optimizer = None
        self.train_datasets = []
        self.eval_datasets = []
    
    def prepare_data(self) -> None:
        """
        Prepare RL training data
        
        Sets:
        - self.train_datasets: List of training tasks
        - self.eval_datasets: List of evaluation tasks
        - self.rollout_generator: Rollout generator (vLLM or Transformers)
        """
        print(f"\n[RL] Preparing data...")
        
        # CPU Optimization: Call at the start of training
        if self.cfg.device_type == "cpu":
            optimize_cpu_performance()
        
        # Create data preparer
        data_preparer = RLDataPreparer(self.cfg)
        
        # Prepare training data
        self.train_datasets = data_preparer.prepare_train_data()
        
        # Prepare evaluation data
        self.eval_datasets = data_preparer.prepare_eval_data()
        
        # Create rollout generator (vLLM if available and on GPU)
        self.use_vllm = False
        if VLLM_AVAILABLE and self.cfg.device_type != "cpu":
            try:
                print(f"\n  [vLLM] Initializing vLLM rollout generator...")
                model_path = self._resolve_model_path(self.cfg.model_id)
                self.rollout_generator = vLLMRolloutGenerator(
                    model_path=model_path,
                    tokenizer=self.tokenizer,
                    tensor_parallel_size=1,
                )
                self.use_vllm = True
                print(f"  [vLLM] [OK] Using vLLM for fast rollout generation")
            except Exception as e:
                print(f"  [vLLM] [FAIL] Failed to initialize vLLM: {e}")
                print(f"  [vLLM] Falling back to Transformers generate()")
                self.rollout_generator = RolloutGenerator(self.model, self.tokenizer, self.cfg)
        else:
            if not VLLM_AVAILABLE:
                print(f"\n  [Rollout] vLLM not available. Using Transformers generate().")
                print(f"  [Rollout] To enable vLLM (5-10x speedup), install: pip install vllm")
            else:
                print(f"\n  [Rollout] CPU mode detected. Using Transformers generate().")
                print(f"  [Rollout] CPU optimizations enabled: batch generation + thread optimization")
                if getattr(self.cfg, "rl_quantize_cpu", False):
                    print(f"  [Rollout] Dynamic quantization enabled (2-4x speedup)")
                if getattr(self.cfg, "rl_compile_model", False):
                    print(f"  [Rollout] torch.compile enabled (PyTorch 2.0+)")
            self.rollout_generator = RolloutGenerator(self.model, self.tokenizer, self.cfg)
        
        # Create loss calculator
        self.loss_calculator = GRPOLossCalculator(self.cfg)
        
        # Create evaluator
        self.evaluator = RLEvaluator(self.model, self.tokenizer, self.cfg)
        
        print(f"\n  Data preparation complete.")
        print(f"  Rollout backend: {'vLLM' if self.use_vllm else 'Transformers (CPU optimized)'}")
    
    def _eval(self, step: int, tag: str = "eval") -> Dict[str, float]:
        """
        Run evaluation
        
        Two types of evaluation:
        1. Reward-based eval (pass@k)
        2. ChatCORE eval
        
        Args:
            step: Current step
            tag: Tag for printing
        
        Returns:
            metrics: dict, containing all evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"  {tag} Eval (Step {step})")
        print(f"{'='*60}")
        
        metrics = {}
        
        # Fix: Initialize evaluator if not yet initialized
        # (This can happen when _eval() is called before prepare_data())
        if self.evaluator is None:
            print(f"  [Eval] Initializing evaluator...")
            self.evaluator = RLEvaluator(self.model, self.tokenizer, self.cfg)
        
        # 1. Reward-based eval (pass@k)
        print("\n  [1/2] Reward-based eval (pass@k)")
        # Fix: Initialize eval_datasets if not yet initialized
        if not hasattr(self, 'eval_datasets') or not self.eval_datasets:
            print(f"  [Eval] Initializing eval datasets...")
            data_preparer = RLDataPreparer(self.cfg)
            self.eval_datasets = data_preparer.prepare_eval_data()

        for dataset in self.eval_datasets:
            task_name = dataset.task_name
            results = self.evaluator.evaluate_pass_at_k(
                task_name=task_name,
                split="test",
                max_examples=self.cfg.eval_max_problems,
                k=min(4, self.cfg.rl_num_samples),
                max_new_tokens=self.cfg.max_seq_len,
                temperature=0.6,
            )
            # Add results to metrics
            for metric_name, value in results.items():
                metrics[f"{task_name}_{metric_name}"] = value

        # 2. ChatCORE eval
        print("\n  [2/2] ChatCORE eval")
        chatcore_results = self.evaluator.evaluate_chatcore(
            eval_tasks=self.cfg.eval_task_list,
            max_problems=self.cfg.eval_max_problems,
        )
        # Add results to metrics
        for task_name, acc in chatcore_results.items():
            metrics[task_name] = acc
        
        print(f"\n  {tag} Eval (Step {step}) complete.")
        return metrics
    
    def _train(self) -> Dict[str, Any]:
        """
        RL training loop
        
        Returns:
            summary: dict, containing training summary
        """
        print(f"\n[RL] Starting training...")
        print(f"  Steps: {self.cfg.max_steps}")
        print(f"  Num samples per prompt (G): {self.cfg.rl_num_samples}")
        print(f"  Examples per step: {self.cfg.rl_examples_per_step}")
        print(f"  Temperature: {self.cfg.rl_temperature}")
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate,
            weight_decay=0.0,
        )
        
        # Create data iterator
        data_preparer = RLDataPreparer(self.cfg)
        train_iterators = []
        for dataset in self.train_datasets:
            train_iterators.append(itertools.cycle(range(len(dataset))))
        
        # Training loop
        step = 0
        task_idx = 0
        total_rewards = []
        step_timer = StepTimer()  # Initialize step timer
        
        # Create history writer
        history_path = Path(self.cfg.output_dir) / "loss_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  Timing: enabled (rollout | backward | update | eval | total)")
        
        while step < self.cfg.max_steps:
            # Reset timer for this step
            step_timer.reset()
            t_step_start = time.perf_counter()
            
            # 1. Select task (round-robin)
            current_dataset = self.train_datasets[task_idx % len(self.train_datasets)]
            current_iter = train_iterators[task_idx % len(self.train_datasets)]
            task_idx += 1
            
            # 2. Collect one batch of data
            batch_rewards = []
            
            for ex_step in range(self.cfg.rl_examples_per_step):
                # 2.1 Sample prompt
                sample_idx = next(current_iter)
                conversation = current_dataset[sample_idx]
                
                # 2.2 Generate rollouts (timed)
                t_rollout_start = time.perf_counter()
                generated_sequences, rewards = self.rollout_generator.generate_rollouts(
                    conversation=conversation,
                    task=current_dataset.task,
                    num_samples=self.cfg.rl_num_samples,
                    max_new_tokens=self.cfg.max_seq_len,
                    temperature=self.cfg.rl_temperature,
                    top_k=50,
                )
                step_timer.rollout_time += time.perf_counter() - t_rollout_start
                
                batch_rewards.extend(rewards)
                
                # 2.3 Prepare training data
                #     Convert generated_sequences to input_ids and target_ids
                #     Calculate advantage
                #     Calculate loss
                #     Backpropagation
                
                # a. Pad sequences to same length
                assistant_end = self.tokenizer.eos_token_id
                max_len = max(len(seq) for seq in generated_sequences)
                padded_seqs = [seq + [assistant_end] * (max_len - len(seq)) for seq in generated_sequences]
                
                # b. Create generation mask (1 for generated, 0 for prompt)
                prompt_len = len(self.rollout_generator.render_prompt(conversation))
                masks = []
                for seq in generated_sequences:
                    mask = [0] * prompt_len + [1] * (len(seq) - prompt_len)
                    # Pad mask
                    mask = mask + [0] * (max_len - len(mask))
                    masks.append(mask)
                
                # c. Convert to tensor
                input_ids = torch.tensor(padded_seqs, dtype=torch.long, device=self.device)
                generation_mask = torch.tensor(masks, dtype=torch.float, device=self.device)
                
                # d. Create target_ids (input_ids shifted right, prompt part set to -1)
                target_ids = input_ids[:, 1:].clone()
                gen_mask_shifted = generation_mask[:, 1:].clone()
                target_ids[gen_mask_shifted == 0] = -1  # Mask prompt part
                
                # Adjust input_ids and generation_mask (remove last token)
                input_ids = input_ids[:, :-1]
                generation_mask = generation_mask[:, :-1]
                
                # e. Calculate advantage
                advantages = self.rollout_generator.compute_advantages(
                    rewards, method="naive", normalize=False
                )
                
                # f. Training: forward + backward (timed)
                t_backward_start = time.perf_counter()
                self.model.train()
                
                # Process in batches (avoid OOM)
                bs = input_ids.size(0)
                device_bs = min(self.cfg.rl_device_batch_size, bs)
                num_passes = max(1, bs // device_bs)
                
                for pass_idx in range(num_passes):
                    b0 = pass_idx * device_bs
                    b1 = min(b0 + device_bs, bs)
                    
                    inp = input_ids[b0:b1]
                    tgt = target_ids[b0:b1]
                    msk = generation_mask[b0:b1]
                    adv = advantages[b0:b1]
                    
                    # Calculate loss
                    loss, metrics = self.loss_calculator.compute_loss_token_level(
                        self.model, inp, tgt, adv, msk
                    )
                    
                    # Backpropagation
                    loss.backward()
                
                step_timer.backward_time += time.perf_counter() - t_backward_start
                batch_rewards.extend(rewards)
            
            # 3. Update model (timed)
            t_update_start = time.perf_counter()
            
            # Gradient clipping
            if self.cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=float(self.cfg.gradient_clip),
                )
            
            # Learning rate scheduling (linear warmup)
            if step < self.cfg.warmup_steps:
                lr_mult = (step + 1) / self.cfg.warmup_steps
            else:
                lr_mult = 1.0
            
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.cfg.learning_rate * lr_mult
            
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            step_timer.update_time = time.perf_counter() - t_update_start
            
            # 4. Logging and eval
            if step % self.cfg.logging_steps == 0:
                mean_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
                total_rewards.append(mean_reward)
                
                # Calculate total step time
                step_timer.total_time = time.perf_counter() - t_step_start
                
                # Write to history
                history_record = {
                    "step": step,
                    "mean_reward": round(mean_reward, 6),
                    "lr": self.cfg.learning_rate * lr_mult,
                    "timing": step_timer.to_dict(),
                }
                with open(history_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(history_record) + "\n")
                
                print(f"  Step {step}/{self.cfg.max_steps} | mean_reward={mean_reward:.4f} | lr={self.cfg.learning_rate * lr_mult:.2e}")
                print(f"       {step_timer.format()}")
            
            # 5. Periodic eval (timed)
            if (step > 0 and 
                self.cfg.eval_steps > 0 and 
                step % self.cfg.eval_steps == 0):
                t_eval_start = time.perf_counter()
                self._eval(step, tag=f"Step {step}")
                step_timer.eval_time += time.perf_counter() - t_eval_start
            
            # 6. Periodic saving
            if (step > 0 and 
                self.cfg.save_steps > 0 and 
                step % self.cfg.save_steps == 0):
                save_dir = Path(self.cfg.output_dir) / f"checkpoint-{step}"
                save_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(save_dir))
                self.tokenizer.save_pretrained(str(save_dir))
                print(f"  [OK] Saved checkpoint to {save_dir}")
            
            step += 1
        
        # Note: Final eval is handled by base.run() after _train() returns
        summary = {
            "total_steps": step,
            "mean_reward_history": total_rewards,
        }
        
        print(f"\n[RL] Training complete.")
        return summary
    
    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    
    def _resolve_model_path(self, model_id: str) -> str:
        """Resolve model path"""
        from nanochat.transformers_backend import resolve_hf_model_path
        return resolve_hf_model_path(model_id)
    
    def _load_tokenizer(self, model_path: str):
        """Load tokenizer"""
        from transformers import AutoTokenizer
        from nanochat.transformers_backend import _prefer_offline_hub_load
        
        # 使用已有的工具函数决定是否使用本地文件
        load_path, local_files_only = _prefer_offline_hub_load(self.cfg.model_id, model_path)
        
        if local_files_only:
            print(f"  [Tokenizer] Using local cache only (offline mode)")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                load_path, 
                use_fast=True, 
                local_files_only=local_files_only
            )
        except Exception as e:
            if local_files_only:
                print(f"  [Tokenizer] Failed to load from local cache: {e}")
                print(f"  [Tokenizer] Try downloading the model first using:")
                print(f"  [Tokenizer]   huggingface-cli download {self.cfg.model_id}")
            raise e
        
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _load_model(self, model_path: str):
        """Load model (with CPU optimization support)"""
        from transformers import AutoModelForCausalLM
        from nanochat.transformers_backend import _prefer_offline_hub_load
        
        # 使用已有的工具函数决定是否使用本地文件
        load_path, local_files_only = _prefer_offline_hub_load(self.cfg.model_id, model_path)
        
        if local_files_only:
            print(f"  [Model] Using local cache only (offline mode)")
        
        try:
            # Fix: Ignore torch_dtype deprecation warning
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
                model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=local_files_only,
                )
        except Exception as e:
            if local_files_only:
                print(f"  [Model] Failed to load from local cache: {e}")
                print(f"  [Model] Try downloading the model first using:")
                print(f"  [Model]   huggingface-cli download {self.cfg.model_id}")
            raise e
        
        # CPU Optimization: Dynamic Quantization (for CPU only)
        if self.cfg.device_type == "cpu" and getattr(self.cfg, "rl_quantize_cpu", False):
            print(f"\n  [Model] Applying CPU optimizations...")
            model = apply_dynamic_quantization(model)
        
        # torch.compile() (PyTorch 2.0+)
        if getattr(self.cfg, "rl_compile_model", False):
            print(f"\n  [Model] Compiling model...")
            model = compile_model_for_fast_inference(model, mode="reduce-overhead")
        
        return model
    
    def _apply_layer_truncation(self, max_layers: int) -> None:
        """Apply layer truncation"""
        from nanochat.transformers_backend import TransformersChatBackend
        TransformersChatBackend._truncate_layers_inplace(self.model, max_layers=max_layers)
        print(f"  Truncated to {max_layers} layers")
    
    def _apply_lora(self):
        """Apply LoRA"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            print("peft not installed. Install with: pip install peft", flush=True)
            raise
        
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  LoRA applied: {trainable:,} trainable / {total:,} total")
        
        return self.model
