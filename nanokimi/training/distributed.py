"""
Distributed training utilities for nanoKimi.

This module provides utilities for multi-GPU distributed training using
PyTorch's DistributedDataParallel (DDP) and other distributed training strategies.
"""

import os
import socket
from typing import Optional, Dict, Any, Tuple
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    warnings.warn("Accelerate not available. Multi-GPU training will use basic DDP.")


class DistributedConfig:
    """Configuration for distributed training."""
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        master_addr: str = "localhost",
        master_port: str = "12355",
        use_accelerate: bool = True,
        mixed_precision: str = "fp16",  # "no", "fp16", "bf16"
        gradient_accumulation_steps: int = 1,
        device_placement: bool = True,
        split_batches: bool = False,
    ):
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.use_accelerate = use_accelerate and ACCELERATE_AVAILABLE
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device_placement = device_placement
        self.split_batches = split_batches
    
    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """Create config from environment variables."""
        return cls(
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            master_addr=os.environ.get("MASTER_ADDR", "localhost"),
            master_port=os.environ.get("MASTER_PORT", "12355"),
        )


class DistributedTrainingManager:
    """Manager for distributed training setup and coordination."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.accelerator = None
        self.device = None
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
    
    def setup(self):
        """Setup distributed training environment."""
        if self.config.use_accelerate:
            self._setup_accelerate()
        else:
            self._setup_ddp()
        
        self.is_initialized = True
    
    def _setup_accelerate(self):
        """Setup using HuggingFace Accelerate."""
        if not ACCELERATE_AVAILABLE:
            raise RuntimeError("Accelerate not available. Install with: pip install accelerate")
        
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            device_placement=self.config.device_placement,
            split_batches=self.config.split_batches,
        )
        
        self.device = self.accelerator.device
        self.world_size = self.accelerator.num_processes
        self.rank = self.accelerator.process_index
        self.local_rank = self.accelerator.local_process_index
        
        if self.is_main_process():
            print(f"Accelerate setup: {self.world_size} processes, rank {self.rank}")
    
    def _setup_ddp(self):
        """Setup using PyTorch DistributedDataParallel."""
        # Get distributed parameters
        self.world_size = self.config.world_size or int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = self.config.rank or int(os.environ.get("RANK", "0"))
        self.local_rank = self.config.local_rank or int(os.environ.get("LOCAL_RANK", "0"))
        
        if self.world_size > 1:
            # Setup environment variables
            os.environ["MASTER_ADDR"] = self.config.master_addr
            os.environ["MASTER_PORT"] = self.config.master_port
            os.environ["RANK"] = str(self.rank)
            os.environ["WORLD_SIZE"] = str(self.world_size)
            os.environ["LOCAL_RANK"] = str(self.local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.world_size,
                rank=self.rank
            )
            
            if self.is_main_process():
                print(f"DDP setup: {self.world_size} processes, rank {self.rank}")
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare model for distributed training."""
        if not self.is_initialized:
            raise RuntimeError("Must call setup() before preparing model")
        
        if self.config.use_accelerate and self.accelerator is not None:
            # Accelerate handles model preparation
            return model  # Will be prepared later with other components
        else:
            # Manual DDP setup
            model = model.to(self.device)
            
            if self.world_size > 1:
                model = DDP(
                    model,
                    device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                    output_device=self.local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=True,  # Useful for MoE models
                )
            
            return model
    
    def prepare_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Prepare dataloader for distributed training."""
        if not self.is_initialized:
            raise RuntimeError("Must call setup() before preparing dataloader")
        
        if self.config.use_accelerate and self.accelerator is not None:
            # Accelerate will handle this
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                **kwargs
            )
        else:
            # Manual distributed sampler
            sampler = None
            if self.world_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=shuffle
                )
                shuffle = False  # Sampler handles shuffling
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                **kwargs
            )
    
    def prepare_optimizer(self, optimizer) -> object:
        """Prepare optimizer for distributed training."""
        if self.config.use_accelerate and self.accelerator is not None:
            # Accelerate handles optimizer preparation
            return optimizer
        else:
            # No special preparation needed for DDP optimizer
            return optimizer
    
    def prepare_scheduler(self, scheduler) -> object:
        """Prepare scheduler for distributed training."""
        if self.config.use_accelerate and self.accelerator is not None:
            return scheduler
        else:
            return scheduler
    
    def prepare_all(
        self,
        model,
        optimizer,
        train_dataloader,
        eval_dataloader=None,
        scheduler=None
    ) -> Tuple:
        """Prepare all components at once (recommended for Accelerate)."""
        if self.config.use_accelerate and self.accelerator is not None:
            components = [model, optimizer, train_dataloader]
            if eval_dataloader is not None:
                components.append(eval_dataloader)
            if scheduler is not None:
                components.append(scheduler)
            
            prepared = self.accelerator.prepare(*components)
            
            # Unpack components
            model = prepared[0]
            optimizer = prepared[1]
            train_dataloader = prepared[2]
            
            idx = 3
            if eval_dataloader is not None:
                eval_dataloader = prepared[idx]
                idx += 1
            if scheduler is not None:
                scheduler = prepared[idx]
            
            return model, optimizer, train_dataloader, eval_dataloader, scheduler
        else:
            # Manual preparation
            model = self.prepare_model(model)
            optimizer = self.prepare_optimizer(optimizer)
            train_dataloader = self.prepare_dataloader(
                train_dataloader.dataset,
                train_dataloader.batch_size,
                shuffle=True
            )
            if eval_dataloader is not None:
                eval_dataloader = self.prepare_dataloader(
                    eval_dataloader.dataset,
                    eval_dataloader.batch_size,
                    shuffle=False
                )
            scheduler = self.prepare_scheduler(scheduler)
            
            return model, optimizer, train_dataloader, eval_dataloader, scheduler
    
    def backward(self, loss):
        """Backward pass with proper scaling for distributed training."""
        if self.config.use_accelerate and self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
    
    def step_optimizer(self, optimizer):
        """Optimizer step with gradient synchronization."""
        if self.config.use_accelerate and self.accelerator is not None:
            # Accelerate handles this automatically
            optimizer.step()
        else:
            optimizer.step()
    
    def gather(self, tensor):
        """Gather tensor from all processes."""
        if self.config.use_accelerate and self.accelerator is not None:
            return self.accelerator.gather(tensor)
        else:
            if self.world_size <= 1:
                return tensor
            
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_tensors, tensor)
            return torch.cat(gathered_tensors, dim=0)
    
    def reduce_mean(self, tensor):
        """Reduce tensor across all processes (mean)."""
        if self.config.use_accelerate and self.accelerator is not None:
            return self.accelerator.reduce(tensor, reduction="mean")
        else:
            if self.world_size <= 1:
                return tensor
            
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor / self.world_size
    
    def is_main_process(self) -> bool:
        """Check if current process is the main process."""
        if self.config.use_accelerate and self.accelerator is not None:
            return self.accelerator.is_main_process
        else:
            return self.rank == 0
    
    def wait_for_everyone(self):
        """Wait for all processes to reach this point."""
        if self.config.use_accelerate and self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        else:
            if self.world_size > 1:
                dist.barrier()
    
    def save_state(self, output_dir: str, **kwargs):
        """Save state across all processes."""
        if self.config.use_accelerate and self.accelerator is not None:
            self.accelerator.save_state(output_dir, **kwargs)
        else:
            # Only save from main process
            if self.is_main_process():
                return True
            return False
    
    def load_state(self, input_dir: str, **kwargs):
        """Load state across all processes."""
        if self.config.use_accelerate and self.accelerator is not None:
            self.accelerator.load_state(input_dir, **kwargs)
        else:
            # Implementation would depend on specific saving format
            pass
    
    def print(self, *args, **kwargs):
        """Print only from main process."""
        if self.is_main_process():
            print(*args, **kwargs)
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.world_size > 1 and not self.config.use_accelerate:
            dist.destroy_process_group()


def setup_distributed_training(
    config: Optional[DistributedConfig] = None
) -> DistributedTrainingManager:
    """
    Setup distributed training environment.
    
    Args:
        config: Distributed training configuration
        
    Returns:
        Configured distributed training manager
    """
    if config is None:
        config = DistributedConfig.from_env()
    
    manager = DistributedTrainingManager(config)
    manager.setup()
    
    return manager


def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_world_size() -> int:
    """Get world size for distributed training."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_rank() -> int:
    """Get rank for distributed training."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return get_world_size() > 1


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def launch_distributed_training(
    train_func,
    config: DistributedConfig,
    nprocs: Optional[int] = None,
):
    """
    Launch distributed training using multiprocessing.
    
    Args:
        train_func: Training function to run
        config: Distributed configuration
        nprocs: Number of processes (defaults to number of GPUs)
    """
    if nprocs is None:
        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if nprocs <= 1:
        # Single process training
        train_func(config)
    else:
        # Multi-process training
        mp.spawn(
            _distributed_worker,
            args=(train_func, config, nprocs),
            nprocs=nprocs,
            join=True
        )


def _distributed_worker(rank: int, train_func, config: DistributedConfig, world_size: int):
    """Worker function for distributed training."""
    # Update config with rank and world size
    config.rank = rank
    config.local_rank = rank
    config.world_size = world_size
    
    # Setup environment
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # Run training function
    train_func(config)


class DistributedMetrics:
    """Utilities for handling metrics in distributed training."""
    
    def __init__(self, manager: DistributedTrainingManager):
        self.manager = manager
    
    def reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Reduce metrics across all processes."""
        reduced_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor_value = torch.tensor(value, device=self.manager.device)
                reduced_value = self.manager.reduce_mean(tensor_value)
                reduced_metrics[key] = reduced_value.item()
            else:
                reduced_metrics[key] = value
        
        return reduced_metrics
    
    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, list]:
        """Gather metrics from all processes."""
        gathered_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor_value = torch.tensor(value, device=self.manager.device)
                gathered_values = self.manager.gather(tensor_value)
                gathered_metrics[key] = gathered_values.cpu().tolist()
            else:
                gathered_metrics[key] = [value]
        
        return gathered_metrics