#!/usr/bin/env python3
"""
Training script for nanoKimi models.

This script provides a command-line interface for training nanoKimi models
with various configurations and datasets.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add the parent directory to the path so we can import nanokimi
sys.path.append(str(Path(__file__).parent.parent))

import torch
from nanokimi.model import KimiConfig
from nanokimi.training import Trainer, TrainingConfig, create_datasets, DataConfig


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
    )
    
    return model_config, training_config


def main():
    parser = argparse.ArgumentParser(description='Train nanoKimi model')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
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
        '--device',
        type=str,
        default=None,
        help='Override device (cpu, cuda, mps, auto)'
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
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create model and training configs
    model_config, training_config = create_configs_from_yaml(config)
    
    # Apply command line overrides
    if args.data_dir:
        training_config.data_dir = args.data_dir
    if args.checkpoint_dir:
        training_config.checkpoint_dir = args.checkpoint_dir
    if args.max_steps:
        training_config.max_steps = args.max_steps
    if args.device:
        training_config.device = args.device
    if args.wandb_project:
        training_config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        training_config.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        training_config.use_wandb = False
    
    # Print configuration summary
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
    
    print(f"  Attention type: {model_config.attention.get('type', 'standard')}")
    if model_config.attention.get('use_flash_attention', False):
        from nanokimi.model.flash_attention import is_flash_attention_available, is_xformers_available
        print(f"  Flash Attention available: {is_flash_attention_available()}")
        print(f"  xFormers available: {is_xformers_available()}")
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Max steps: {training_config.max_steps}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Dataset: {training_config.dataset_name}")
    print(f"  Device: {training_config.device}")
    print(f"  Mixed precision: {training_config.mixed_precision}")
    print(f"  Compile model: {training_config.compile_model}")
    
    # Create data configuration
    data_config = DataConfig(
        tokenizer_name="gpt2",
        block_size=model_config.block_size,
        data_dir=training_config.data_dir,
    )
    
    # Create datasets
    print(f"\nLoading dataset: {training_config.dataset_name}")
    try:
        train_dataset, val_dataset = create_datasets(
            data_config, 
            training_config.dataset_name
        )
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if training_config.dataset_name == "openwebtext":
            print("\nTo use OpenWebText dataset:")
            print("1. Download the dataset")
            print("2. Preprocess it using the data processing utilities")
            print("3. Or use 'toy' dataset for testing")
        sys.exit(1)
    
    # Create trainer
    from nanokimi.training.trainer import create_trainer
    
    print(f"\nCreating trainer...")
    trainer = create_trainer(
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Print model info
    model_params = trainer.model.get_num_params()
    print(f"Model parameters: {model_params:,}")
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.resume_training(args.resume)
    else:
        # Start training
        print(f"\nStarting training...")
        trainer.train()


if __name__ == "__main__":
    main()