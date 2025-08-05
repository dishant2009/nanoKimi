"""
Training loop and trainer class for nanoKimi.

This module implements the main training logic including:
- Training loop with proper logging
- Checkpointing and resumption
- Loss tracking and metrics
- Integration with wandb for experiment tracking
"""

import os
import time
import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from ..model import KimiModel, KimiConfig
from .optimizer import MuonOptimizer
from .data import TokenDataset
from .utils import get_lr_scheduler, compute_loss, get_batch
from .distributed import DistributedTrainingManager, DistributedConfig


@dataclass 
class TrainingConfig:
    """Configuration for training."""
    
    # Basic training settings
    batch_size: int = 32
    micro_batch_size: int = 8  # For gradient accumulation
    max_steps: int = 100000
    eval_interval: int = 1000
    log_interval: int = 100
    
    # Optimization settings
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Expert-specific settings
    expert_lr_multiplier: float = 1.0
    load_balance_loss_coeff: float = 0.01
    
    # Data settings
    dataset_name: str = "openwebtext"
    data_dir: str = "./data"
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 5000
    keep_last_n_checkpoints: int = 3
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "nanokimi"
    wandb_run_name: Optional[str] = None
    
    # Hardware settings
    device: str = "auto"  # auto, cpu, cuda, mps
    compile_model: bool = False  # Use torch.compile
    mixed_precision: bool = True  # Use automatic mixed precision
    
    # Distributed training
    distributed: bool = False  # Enable distributed training
    use_accelerate: bool = True  # Use HuggingFace Accelerate
    ddp_backend: str = "nccl"  # DDP backend
    find_unused_parameters: bool = True  # For MoE models
    
    # Evaluation
    eval_max_batches: int = 100
    

class Trainer:
    """
    Main trainer class for nanoKimi models.
    
    Handles training loop, optimization, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: KimiModel,
        config: TrainingConfig,
        train_dataset: Optional[TokenDataset] = None,
        val_dataset: Optional[TokenDataset] = None,
        distributed_manager: Optional[DistributedTrainingManager] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Setup distributed training
        self.distributed_manager = distributed_manager
        if config.distributed and distributed_manager is None:
            # Auto-setup distributed training
            dist_config = DistributedConfig(
                use_accelerate=config.use_accelerate,
                backend=config.ddp_backend,
                mixed_precision="fp16" if config.mixed_precision else "no",
            )
            self.distributed_manager = DistributedTrainingManager(dist_config)
            self.distributed_manager.setup()
        
        # Setup device
        if self.distributed_manager:
            self.device = self.distributed_manager.device
        else:
            self.device = self._setup_device()
            self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision (if not using Accelerate)
        use_manual_amp = (config.mixed_precision and 
                         torch.cuda.is_available() and 
                         not (self.distributed_manager and self.distributed_manager.config.use_accelerate))
        self.scaler = torch.cuda.amp.GradScaler() if use_manual_amp else None
        
        # Prepare for distributed training
        if self.distributed_manager:
            self._prepare_distributed()
        
        # Compile model if requested (after distributed setup)
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        
        # Setup logging (only on main process)
        if not self.distributed_manager or self.distributed_manager.is_main_process():
            self._setup_logging()
        
        # Create checkpoint directory (only on main process)
        if not self.distributed_manager or self.distributed_manager.is_main_process():
            os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        
        print(f"Using device: {device}")
        return device
    
    def _prepare_distributed(self):
        """Prepare components for distributed training."""
        if not self.distributed_manager:
            return
        
        # Prepare model, optimizer, and data loaders
        train_loader = self._create_dataloader(self.train_dataset, shuffle=True) if self.train_dataset else None
        val_loader = self._create_dataloader(self.val_dataset, shuffle=False) if self.val_dataset else None
        
        # Use Accelerate's prepare method or manual DDP setup
        prepared = self.distributed_manager.prepare_all(
            self.model,
            self.optimizer,
            train_loader,
            val_loader,
            self.scheduler
        )
        
        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.train_loader = prepared[2]
        self.val_loader = prepared[3]
        self.scheduler = prepared[4]
        
        if self.distributed_manager.is_main_process():
            print(f"Distributed training setup complete")
            print(f"  World size: {self.distributed_manager.world_size}")
            print(f"  Backend: {self.distributed_manager.config.backend}")
            if self.distributed_manager.config.use_accelerate:
                print(f"  Using Accelerate with {self.distributed_manager.config.mixed_precision} precision")
    
    def _create_optimizer(self) -> MuonOptimizer:
        """Create Muon optimizer."""
        # Separate parameters for different learning rates
        param_groups = []
        
        # Regular parameters
        regular_params = []
        expert_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(keyword in name.lower() for keyword in ['expert', 'moe', 'router', 'gate']):
                    expert_params.append(param)
                else:
                    regular_params.append(param)
        
        # Add parameter groups
        if regular_params:
            param_groups.append({
                'params': regular_params,
                'lr': self.config.learning_rate,
                'expert_lr_multiplier': 1.0,
            })
        
        if expert_params:
            param_groups.append({
                'params': expert_params,
                'lr': self.config.learning_rate,
                'expert_lr_multiplier': self.config.expert_lr_multiplier,
            })
        
        optimizer = MuonOptimizer(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return get_lr_scheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.lr_decay_steps,
            min_lr=self.config.min_learning_rate,
        )
    
    def _setup_logging(self):
        """Setup logging with wandb."""
        if self.config.use_wandb:
            run_name = self.config.wandb_run_name or f"nanokimi-{int(time.time())}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    **self.config.__dict__,
                    **self.model.config.to_dict(),
                    "model_params": self.model.get_num_params(),
                }
            )
    
    def train(self):
        """Main training loop."""
        if not self.distributed_manager or self.distributed_manager.is_main_process():
            print(f"Starting training for {self.config.max_steps} steps")
            print(f"Model has {self.model.get_num_params():,} parameters")
        
        # Create data loaders (or use prepared ones from distributed setup)
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            train_loader = self.train_loader
            val_loader = self.val_loader
        else:
            train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
            val_loader = self._create_dataloader(self.val_dataset, shuffle=False) if self.val_dataset else None
        
        self.model.train()
        
        # Training metrics
        losses = []
        auxiliary_losses = []
        
        while self.step < self.config.max_steps:
            for batch in train_loader:
                if self.step >= self.config.max_steps:
                    break
                
                # Training step
                loss, aux_loss = self._training_step(batch)
                losses.append(loss)
                if aux_loss is not None:
                    auxiliary_losses.append(aux_loss)
                
                # Logging
                if self.step % self.config.log_interval == 0:
                    self._log_training_metrics(losses, auxiliary_losses)
                    losses = []
                    auxiliary_losses = []
                
                # Evaluation
                if val_loader and self.step % self.config.eval_interval == 0:
                    val_loss = self._evaluate(val_loader)
                    self._log_evaluation_metrics(val_loss)
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)
                
                # Checkpointing
                if self.step % self.config.save_interval == 0:
                    self._save_checkpoint()
                
                self.step += 1
            
            self.epoch += 1
        
        print("Training completed!")
        if self.config.use_wandb:
            wandb.finish()
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Optional[float]]:
        """Perform single training step."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids)
        
        # Handle gradient accumulation
        effective_batch_size = self.config.batch_size
        micro_batch_size = self.config.micro_batch_size
        accumulation_steps = effective_batch_size // micro_batch_size
        
        total_loss = 0.0
        total_aux_loss = 0.0
        
        self.optimizer.zero_grad()
        
        for i in range(accumulation_steps):
            start_idx = i * micro_batch_size
            end_idx = min((i + 1) * micro_batch_size, input_ids.size(0))
            
            micro_input_ids = input_ids[start_idx:end_idx]
            micro_labels = labels[start_idx:end_idx]
            
            # Forward pass with mixed precision
            use_amp = (self.scaler is not None or 
                      (self.distributed_manager and self.distributed_manager.config.use_accelerate))
            
            if self.scaler is not None:
                # Manual mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(micro_input_ids, labels=micro_labels)
                    loss = outputs['loss'] / accumulation_steps
                    aux_loss = outputs.get('auxiliary_loss', 0.0) / accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss + aux_loss).backward()
            elif self.distributed_manager and self.distributed_manager.config.use_accelerate:
                # Accelerate handles mixed precision
                outputs = self.model(micro_input_ids, labels=micro_labels)
                loss = outputs['loss'] / accumulation_steps
                aux_loss = outputs.get('auxiliary_loss', 0.0) / accumulation_steps
                
                # Use distributed manager's backward
                self.distributed_manager.backward(loss + aux_loss)
            else:
                # Standard forward/backward
                outputs = self.model(micro_input_ids, labels=micro_labels)
                loss = outputs['loss'] / accumulation_steps
                aux_loss = outputs.get('auxiliary_loss', 0.0) / accumulation_steps
                
                (loss + aux_loss).backward()
            
            total_loss += loss.item()
            total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        
        # Gradient clipping and optimization step
        if self.scaler is not None:
            # Manual mixed precision with gradient scaling
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.distributed_manager and self.distributed_manager.config.use_accelerate:
            # Accelerate handles gradient clipping and optimizer step
            self.distributed_manager.accelerator.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
            self.distributed_manager.step_optimizer(self.optimizer)
        else:
            # Standard gradient clipping and step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        return total_loss, total_aux_loss if total_aux_loss > 0 else None
    
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= self.config.eval_max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, labels=labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _create_dataloader(self, dataset: Optional[TokenDataset], shuffle: bool = True) -> Optional[DataLoader]:
        """Create data loader from dataset."""
        if dataset is None:
            return None
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    
    def _log_training_metrics(self, losses: List[float], auxiliary_losses: List[float]):
        """Log training metrics."""
        if not losses:
            return
        
        # Only log from main process
        if self.distributed_manager and not self.distributed_manager.is_main_process():
            return
        
        avg_loss = sum(losses) / len(losses)
        avg_aux_loss = sum(auxiliary_losses) / len(auxiliary_losses) if auxiliary_losses else 0.0
        
        # Reduce losses across processes if distributed
        if self.distributed_manager:
            from .distributed import DistributedMetrics
            dist_metrics = DistributedMetrics(self.distributed_manager)
            reduced = dist_metrics.reduce_metrics({
                "loss": avg_loss,
                "aux_loss": avg_aux_loss
            })
            avg_loss = reduced["loss"]
            avg_aux_loss = reduced["aux_loss"]
        
        lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate
        
        elapsed_time = time.time() - self.start_time
        tokens_per_sec = (self.step * self.config.batch_size * self.model.config.block_size) / elapsed_time
        
        # Adjust for distributed training
        if self.distributed_manager:
            tokens_per_sec *= self.distributed_manager.world_size
        
        metrics = {
            "train/loss": avg_loss,
            "train/auxiliary_loss": avg_aux_loss,
            "train/learning_rate": lr,
            "train/tokens_per_second": tokens_per_sec,
            "train/step": self.step,
            "train/epoch": self.epoch,
        }
        
        # Log to console
        print(f"Step {self.step:6d} | Loss: {avg_loss:.4f} | Aux Loss: {avg_aux_loss:.4f} | LR: {lr:.2e} | Tokens/s: {tokens_per_sec:.0f}")
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log(metrics, step=self.step)
    
    def _log_evaluation_metrics(self, val_loss: float):
        """Log evaluation metrics."""
        # Only log from main process
        if self.distributed_manager and not self.distributed_manager.is_main_process():
            return
            
        metrics = {
            "eval/loss": val_loss,
            "eval/perplexity": math.exp(val_loss),
            "eval/step": self.step,
        }
        
        print(f"Validation | Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
        
        if self.config.use_wandb:
            wandb.log(metrics, step=self.step)
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        # Only save from main process
        if self.distributed_manager and not self.distributed_manager.is_main_process():
            return
        
        # Wait for all processes to reach this point
        if self.distributed_manager:
            self.distributed_manager.wait_for_everyone()
        
        # Get model state dict (unwrap DDP if needed)
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
            model_config = self.model.module.config.to_dict()
        else:
            model_state_dict = self.model.state_dict()
            model_config = self.model.config.to_dict()
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': model_config,
            'training_config': self.config.__dict__,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{self.step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoint_files = [f for f in os.listdir(self.config.checkpoint_dir) 
                           if f.startswith("checkpoint_step_") and f.endswith(".pt")]
        
        if len(checkpoint_files) <= self.config.keep_last_n_checkpoints:
            return
        
        # Sort by step number and remove oldest
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        files_to_remove = checkpoint_files[:-self.config.keep_last_n_checkpoints]
        
        for filename in files_to_remove:
            file_path = os.path.join(self.config.checkpoint_dir, filename)
            os.remove(file_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from step {self.step}")
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        self.load_checkpoint(checkpoint_path)
        self.train()


def create_trainer(
    model_config: KimiConfig,
    training_config: TrainingConfig,
    train_dataset: Optional[TokenDataset] = None,
    val_dataset: Optional[TokenDataset] = None,
    distributed_manager: Optional[DistributedTrainingManager] = None,
) -> Trainer:
    """
    Factory function to create trainer with model.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        distributed_manager: Optional distributed training manager
        
    Returns:
        Configured trainer instance
    """
    model = KimiModel(model_config)
    
    return Trainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        distributed_manager=distributed_manager,
    )