#!/usr/bin/env python3
"""
Distributed training script for nanoKimi models.

This script provides distributed training capabilities using either
PyTorch's native DistributedDataParallel or HuggingFace Accelerate.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add the parent directory to the path so we can import nanokimi
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.multiprocessing as mp

from nanokimi.model import KimiConfig
from nanokimi.training import (
    Trainer, TrainingConfig, create_datasets, DataConfig,
    DistributedConfig, DistributedTrainingManager, setup_distributed_training
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_configs_from_yaml(yaml_config: dict) -> tuple:
    """Create model and training configs from YAML configuration."""
    # Extract model config
    model_config_dict = yaml_config['model']
    model_config = KimiConfig(
        n_layer=model_config_dict['n_layer'],
        n_head=model_config_dict['n_head'],
        n_embd=model_config_dict['n_embd'],
        vocab_size=model_config_dict['vocab_size'],
        block_size=model_config_dict['block_size'],
        dropout=model_config_dict.get('dropout', 0.1),
        moe=model_config_dict.get('moe', {}),
        attention=model_config_dict.get('attention', {}),
    )
    
    # Extract training config
    training_config_dict = yaml_config['training']
    training_config = TrainingConfig(
        batch_size=training_config_dict.get('batch_size', 32),
        micro_batch_size=training_config_dict.get('micro_batch_size', 8),
        max_steps=training_config_dict.get('max_steps', 100000),
        eval_interval=training_config_dict.get('eval_interval', 1000),
        log_interval=training_config_dict.get('log_interval', 100),
        learning_rate=training_config_dict.get('learning_rate', 3e-4),
        min_learning_rate=training_config_dict.get('min_learning_rate', 3e-5),
        warmup_steps=training_config_dict.get('warmup_steps', 1000),
        lr_decay_steps=training_config_dict.get('lr_decay_steps', 100000),
        weight_decay=training_config_dict.get('weight_decay', 0.1),
        grad_clip=training_config_dict.get('grad_clip', 1.0),
        expert_lr_multiplier=training_config_dict.get('expert_lr_multiplier', 1.0),
        load_balance_loss_coeff=training_config_dict.get('load_balance_loss_coeff', 0.01),
        dataset_name=training_config_dict.get('dataset_name', 'toy'),
        data_dir=training_config_dict.get('data_dir', './data'),
        num_workers=training_config_dict.get('num_workers', 4),
        checkpoint_dir=training_config_dict.get('checkpoint_dir', './checkpoints'),
        save_interval=training_config_dict.get('save_interval', 5000),
        keep_last_n_checkpoints=training_config_dict.get('keep_last_n_checkpoints', 3),
        use_wandb=training_config_dict.get('use_wandb', True),
        wandb_project=training_config_dict.get('wandb_project', 'nanokimi'),
        wandb_run_name=training_config_dict.get('wandb_run_name', None),
        device=training_config_dict.get('device', 'auto'),
        compile_model=training_config_dict.get('compile_model', False),
        mixed_precision=training_config_dict.get('mixed_precision', True),
        eval_max_batches=training_config_dict.get('eval_max_batches', 100),
        
        # Distributed training settings
        distributed=True,  # Enable distributed training
        use_accelerate=training_config_dict.get('use_accelerate', True),
        ddp_backend=training_config_dict.get('ddp_backend', 'nccl'),
        find_unused_parameters=training_config_dict.get('find_unused_parameters', True),
    )
    
    return model_config, training_config


def train_worker(rank: int, world_size: int, args):
    """Training worker function for distributed training."""
    # Setup environment variables
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    # Load configuration
    config = load_config(args.config)
    model_config, training_config = create_configs_from_yaml(config)
    
    # Apply command line overrides
    if args.data_dir:
        training_config.data_dir = args.data_dir
    if args.checkpoint_dir:
        training_config.checkpoint_dir = args.checkpoint_dir
    if args.max_steps:
        training_config.max_steps = args.max_steps
    if args.wandb_project:
        training_config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        training_config.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        training_config.use_wandb = False
    
    # Setup distributed training
    dist_config = DistributedConfig(
        use_accelerate=training_config.use_accelerate,
        backend=training_config.ddp_backend,
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        master_addr=args.master_addr,
        master_port=str(args.master_port),
        mixed_precision="fp16" if training_config.mixed_precision else "no",
    )
    
    distributed_manager = DistributedTrainingManager(dist_config)
    distributed_manager.setup()
    
    # Print configuration (only from main process)
    if distributed_manager.is_main_process():
        print(f"Distributed Training Configuration:")
        print(f"  World size: {world_size}")
        print(f"  Backend: {training_config.ddp_backend}")
        print(f"  Use Accelerate: {training_config.use_accelerate}")
        print(f"  Mixed precision: {training_config.mixed_precision}")
        
        print(f"\nModel Configuration:")
        print(f"  Layers: {model_config.n_layer}")
        print(f"  Heads: {model_config.n_head}")
        print(f"  Embedding dim: {model_config.n_embd}")
        print(f"  Vocab size: {model_config.vocab_size}")
        print(f"  Block size: {model_config.block_size}")
        print(f"  MoE enabled: {model_config.moe.get('use_moe', False)}")
        if model_config.moe.get('use_moe', False):
            print(f"  MoE experts: {model_config.moe.get('num_experts', 8)}")
            print(f"  MoE top-k: {model_config.moe.get('top_k', 2)}")
    
    # Create data configuration
    data_config = DataConfig(
        tokenizer_name="gpt2",
        block_size=model_config.block_size,
        data_dir=training_config.data_dir,
    )
    
    # Create datasets
    if distributed_manager.is_main_process():
        print(f"\nLoading dataset: {training_config.dataset_name}")
    
    try:
        train_dataset, val_dataset = create_datasets(
            data_config, 
            training_config.dataset_name
        )
        if distributed_manager.is_main_process():
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Validation dataset size: {len(val_dataset)}")
    except Exception as e:
        if distributed_manager.is_main_process():
            print(f"Error loading dataset: {e}")
            if training_config.dataset_name == "openwebtext":
                print("\nTo use OpenWebText dataset:")
                print("1. Download the dataset")
                print("2. Preprocess it using the data processing utilities")
                print("3. Or use 'toy' dataset for testing")
        return
    
    # Create trainer with distributed manager
    from nanokimi.training.trainer import create_trainer
    
    if distributed_manager.is_main_process():
        print(f"\nCreating distributed trainer...")
    
    trainer = create_trainer(
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Override the trainer's distributed manager
    trainer.distributed_manager = distributed_manager
    trainer._prepare_distributed()
    
    # Print model info (only from main process)
    if distributed_manager.is_main_process():
        model_params = trainer.model.get_num_params()
        print(f"Model parameters: {model_params:,}")
        print(f"Effective batch size: {training_config.batch_size * world_size}")
    
    # Start training
    if distributed_manager.is_main_process():
        print(f"\nStarting distributed training...")
    
    trainer.train()
    
    # Cleanup
    distributed_manager.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Distributed training for nanoKimi')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--nprocs',
        type=int,
        default=None,
        help='Number of processes (defaults to number of GPUs)'
    )
    parser.add_argument(
        '--master-addr',
        type=str,
        default='localhost',
        help='Master address for distributed training'
    )
    parser.add_argument(
        '--master-port',
        type=int,
        default=12355,
        help='Master port for distributed training'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        choices=['nccl', 'gloo', 'mpi'],
        help='Distributed backend'
    )
    parser.add_argument(
        '--use-accelerate',
        action='store_true',
        default=True,
        help='Use HuggingFace Accelerate (recommended)'
    )
    parser.add_argument(
        '--no-accelerate',
        action='store_true',
        help='Disable Accelerate and use native PyTorch DDP'
    )
    
    # Training overrides
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Override data directory from config'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Override checkpoint directory from config'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Override max training steps'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='Override wandb project name'
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='Override wandb run name'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    
    args = parser.parse_args()
    
    # Handle accelerate flag
    if args.no_accelerate:
        args.use_accelerate = False
    
    # Determine number of processes
    if args.nprocs is None:
        if torch.cuda.is_available():
            args.nprocs = torch.cuda.device_count()
        else:
            args.nprocs = 1
    
    print(f"Launching distributed training with {args.nprocs} processes")
    print(f"Master: {args.master_addr}:{args.master_port}")
    print(f"Backend: {args.backend}")
    print(f"Use Accelerate: {args.use_accelerate}")
    
    if args.nprocs <= 1:
        # Single process training
        train_worker(0, 1, args)
    else:
        # Multi-process training
        mp.spawn(
            train_worker,
            args=(args.nprocs, args),
            nprocs=args.nprocs,
            join=True
        )
    
    print("Distributed training completed!")


if __name__ == "__main__":
    main()