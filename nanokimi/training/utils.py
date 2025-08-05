"""
Training utilities for nanoKimi.

This module contains helper functions for training including:
- Learning rate schedulers
- Loss computation utilities
- Training metrics
- Optimization helpers
"""

import math
from typing import Dict, Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with linear warmup.
    
    This scheduler first linearly increases the learning rate during warmup,
    then follows a cosine decay schedule.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Linear learning rate scheduler with warmup.
    
    Linearly increases during warmup, then linearly decreases.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 - progress)
                for base_lr in self.base_lrs
            ]


class WarmupConstantScheduler(_LRScheduler):
    """
    Constant learning rate after warmup.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Constant learning rate
            return self.base_lrs


def get_lr_scheduler(
    optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 1000,
    max_steps: int = 100000,
    min_lr: float = 0.0,
) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "linear":
        return WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "constant":
        return WarmupConstantScheduler(
            optimizer,
            warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute cross-entropy loss with optional label smoothing.
    
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        labels: Ground truth labels of shape (batch_size, seq_len)
        ignore_index: Index to ignore in loss computation
        label_smoothing: Label smoothing factor
        reduction: Loss reduction method
        
    Returns:
        Computed loss
    """
    # Reshape tensors
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    
    if label_smoothing > 0.0:
        # Label smoothing
        confidence = 1.0 - label_smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        
        # True class probability
        nll_loss = F.nll_loss(
            log_probs,
            labels,
            ignore_index=ignore_index,
            reduction='none'
        )
        
        # Smooth probability (uniform over vocab)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combine losses
        loss = confidence * nll_loss + label_smoothing * smooth_loss
        
        if reduction == "mean":
            # Only average over non-ignored tokens
            mask = (labels != ignore_index).float()
            loss = (loss * mask).sum() / mask.sum()
        elif reduction == "sum":
            loss = loss.sum()
        
        return loss
    else:
        # Standard cross-entropy
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction=reduction,
        )


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return torch.exp(loss)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def clip_grad_norm(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """
    Clip gradient norm of parameters.
    
    This is a wrapper around torch.nn.utils.clip_grad_norm_ with
    additional logging capabilities.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use
        error_if_nonfinite: Whether to error on non-finite gradients
        
    Returns:
        Total norm of gradients
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
    
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`'
        )
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    
    return total_norm


def get_memory_stats() -> Dict[str, float]:
    """
    Get GPU memory statistics.
    
    Returns:
        Dictionary of memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {}
    
    stats = {}
    for i in range(torch.cuda.device_count()):
        device_name = f"cuda:{i}"
        stats[f"{device_name}_allocated"] = torch.cuda.memory_allocated(i) / 1024**2
        stats[f"{device_name}_cached"] = torch.cuda.memory_reserved(i) / 1024**2
        stats[f"{device_name}_max_allocated"] = torch.cuda.max_memory_allocated(i) / 1024**2
    
    return stats


def log_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Log gradient statistics for debugging.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of gradient statistics
    """
    stats = {}
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            stats[f"grad_norm/{name}"] = grad_norm.item()
            total_norm += grad_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        stats["grad_norm/total"] = math.sqrt(total_norm)
        stats["grad_norm/average"] = math.sqrt(total_norm / param_count)
    
    return stats


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Create attention mask from input ids.
    
    Args:
        input_ids: Input token ids
        pad_token_id: Padding token id
        
    Returns:
        Attention mask (1 for real tokens, 0 for padding)
    """
    return (input_ids != pad_token_id).long()


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def estimate_mfu(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    dt: float,
) -> float:
    """
    Estimate model flops utilization (MFU).
    
    Args:
        model: PyTorch model
        batch_size: Batch size
        seq_len: Sequence length
        dt: Time per step in seconds
        
    Returns:
        MFU as a percentage
    """
    # Estimate FLOPs per forward pass
    # This is a rough estimate for transformer models
    N = count_parameters(model)  # Number of parameters
    L = getattr(model.config, 'n_layer', 12)  # Number of layers
    H = getattr(model.config, 'n_head', 12)  # Number of heads
    
    # Rough FLOP count (forward pass only)
    flops_per_token = 6 * N + 12 * L * H * seq_len**2
    flops_per_fwdbwd = 3 * flops_per_token  # Forward + backward + optimizer
    
    total_flops = flops_per_fwdbwd * batch_size * seq_len
    flops_per_sec = total_flops / dt
    
    # Estimate hardware peak FLOPS (rough estimate for modern GPUs)
    if torch.cuda.is_available():
        # Rough estimate: ~100 TFLOPS for modern GPUs
        hardware_flops = 100e12
    else:
        # Rough estimate for CPU
        hardware_flops = 1e12
    
    mfu = flops_per_sec / hardware_flops * 100
    return min(mfu, 100.0)  # Cap at 100%


def format_time(seconds: float) -> str:
    """
    Format time duration.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def save_training_config(config: Dict[str, Any], path: str):
    """
    Save training configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Path to save configuration
    """
    import json
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def get_batch(dataset, batch_size: int, block_size: int, device: torch.device):
    """
    Get a batch of data from the dataset.
    
    Args:
        dataset: Dataset to sample from
        batch_size: Batch size
        block_size: Sequence length
        device: Device to move data to
        
    Returns:
        Tuple of (input_ids, labels)
    """
    # Sample random indices
    indices = torch.randint(0, len(dataset), (batch_size,))
    
    # Get samples
    batch_data = [dataset[i] for i in indices]
    
    # Stack into tensors
    input_ids = torch.stack([item['input_ids'] for item in batch_data])
    labels = torch.stack([item['labels'] for item in batch_data])
    
    # Move to device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    return input_ids, labels