"""
RL Data Preparation Module

This module contains data preparation functions for RL training:
1. Task loading (GSM8K, SpellingBee, etc.)
2. Dataset wrapping
3. Data iteration

Key concepts for understanding RL data preparation:
- RL data is not fixed (input, target) pairs
- Instead, it's (prompt, task) pairs, where the model generates multiple rollouts,
  and then learns from the rewards
"""

from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import itertools


# =============================================================================
# RL Task Interface
# =============================================================================

class RLTaskWrapper:
    """
    RL Task Wrapper
    
    All RL tasks must implement:
        reward(conversation, assistant_response) -> float
    
    This wrapper ensures tasks have a unified interface
    """
    
    def __init__(self, task, task_name: str):
        self.task = task
        self.task_name = task_name
    
    def __len__(self):
        return len(self.task)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Returns conversation dict
        
        Conversation format:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}  # optional
            ]
        }
        """
        return self.task[idx]
    
    def reward(self, conversation: Dict, assistant_response: str) -> float:
        """
        Calculate reward
        
        Args:
            conversation: Conversation history
            assistant_response: Model-generated response
        
        Returns:
            reward: float, reward value (usually 0.0 or 1.0)
        """
        return self.task.reward(conversation, assistant_response)


# =============================================================================
# Dataset Classes
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
        self.task_wrapper = self._create_task(task_name, split)
        self.examples = list(range(len(self.task_wrapper)))
    
    def _create_task(self, task_name: str, split: str):
        """Create task object that supports reward()"""
        if task_name == "GSM8K":
            from tasks.gsm8k import GSM8K
            task = GSM8K(subset="main", split=split)
        elif task_name == "SpellingBee":
            from tasks.spellingbee import SpellingBee
            task = SpellingBee(size=256, split=split)
        else:
            raise ValueError(
                f"Task '{task_name}' does not support reward(). "
                f"Available RL tasks: GSM8K, SpellingBee"
            )
        
        return RLTaskWrapper(task, task_name)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Returns conversation dict"""
        return self.task_wrapper[idx]


# =============================================================================
# Data Preparer
# =============================================================================

class RLDataPreparer:
    """
    RL Data Preparer
    
    Functions:
    1. Load RL tasks (GSM8K, SpellingBee, etc.)
    2. Create training data iterator
    3. Support multi-task mixed training
    """
    
    def __init__(self, train_tasks: List[str], eval_tasks: List[str]):
        """
        Args:
            train_tasks: List of training task names
            eval_tasks: List of evaluation task names
        """
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.train_datasets = []
        self.eval_datasets = []
    
    def prepare_train_data(self) -> List[RLTaskDataset]:
        """Prepare training data"""
        print(f"\n[RL Data] Loading training tasks: {self.train_tasks}")
        
        for task_name in self.train_tasks:
            dataset = RLTaskDataset(task_name, split="train")
            self.train_datasets.append(dataset)
            print(f"  {task_name}: {len(dataset)} examples (train split)")
        
        print(f"  Total training tasks: {len(self.train_datasets)}")
        return self.train_datasets
    
    def prepare_eval_data(self) -> List[RLTaskDataset]:
        """Prepare evaluation data"""
        print(f"\n[RL Data] Loading eval tasks: {self.eval_tasks}")
        
        for task_name in self.eval_tasks:
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
            yield current_dataset[sample_idx], current_dataset.task_wrapper


# =============================================================================
# Helper Functions
# =============================================================================

def create_rl_dataset(task_name: str, split: str = "train") -> RLTaskDataset:
    """Create RL dataset (factory function)"""
    return RLTaskDataset(task_name, split)


def get_available_rl_tasks() -> List[str]:
    """Get list of available RL tasks"""
    return ["GSM8K", "SpellingBee"]
