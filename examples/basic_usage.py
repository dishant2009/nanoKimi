#!/usr/bin/env python3
"""
Basic usage example for nanoKimi.

This script demonstrates how to create, train, and use a nanoKimi model
for text generation.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import nanokimi
sys.path.append(str(Path(__file__).parent.parent))

import torch
from nanokimi.model import KimiModel, KimiConfig
from nanokimi.training import create_trainer, TrainingConfig, create_datasets, DataConfig
from nanokimi.inference import Generator, GenerationConfig


def main():
    print("nanoKimi Basic Usage Example")
    print("=" * 40)
    
    # 1. Create model configuration
    print("\n1. Creating model configuration...")
    model_config = KimiConfig(
        n_layer=6,          # Small model for demo
        n_head=6,
        n_embd=384,
        vocab_size=50257,
        block_size=512,
        dropout=0.1,
        moe={
            "use_moe": True,
            "num_experts": 4,
            "top_k": 2,
            "expert_capacity_factor": 1.25,
            "load_balance_loss_coeff": 0.01,
            "moe_layers": [2, 4],
        },
        attention={
            "type": "latent",
            "latent_dim": 32,
            "num_latents": 16,
        }
    )
    
    print(f"Model configuration:")
    print(f"  - Layers: {model_config.n_layer}")
    print(f"  - Heads: {model_config.n_head}")
    print(f"  - Embedding dim: {model_config.n_embd}")
    print(f"  - MoE experts: {model_config.moe['num_experts']}")
    
    # 2. Create model
    print("\n2. Creating model...")
    model = KimiModel(model_config)
    num_params = model.get_num_params()
    print(f"Model created with {num_params:,} parameters")
    
    # 3. Create datasets
    print("\n3. Creating toy dataset...")
    data_config = DataConfig(
        tokenizer_name="gpt2",
        block_size=model_config.block_size,
        data_dir="./data",
    )
    
    train_dataset, val_dataset = create_datasets(data_config, "toy")
    print(f"Dataset created:")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    
    # 4. Setup training configuration
    print("\n4. Setting up training...")
    training_config = TrainingConfig(
        batch_size=2,
        micro_batch_size=1,
        max_steps=10,           # Very short training for demo
        eval_interval=5,
        log_interval=5,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
        warmup_steps=2,
        lr_decay_steps=10,
        weight_decay=0.1,
        grad_clip=1.0,
        dataset_name="toy",
        data_dir="./data",
        checkpoint_dir="./checkpoints/demo",
        save_interval=10,
        use_wandb=False,        # Disable wandb for demo
        device="cpu",           # Use CPU to avoid MPS issues
        compile_model=False,    # Disable compilation for demo
        mixed_precision=False,  # Disable for demo
        eval_max_batches=5,
    )
    
    # 5. Train model (very brief training for demo)
    print("\n5. Training model (brief demo training)...")
    trainer = create_trainer(
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    # 6. Test text generation
    print("\n6. Testing text generation...")
    
    # Create generator
    generator = Generator(trainer.model)
    
    # Generation configuration
    gen_config = GenerationConfig(
        max_new_tokens=50,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time",
        "In a world where",
    ]
    
    print("Generated text samples:")
    print("-" * 30)
    
    for i, prompt in enumerate(prompts, 1):
        try:
            generated_text = generator.generate_text(prompt, gen_config)
            print(f"\nSample {i}:")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
        except Exception as e:
            print(f"Error generating text for prompt {i}: {e}")
    
    # 7. Demonstrate streaming generation
    print("\n7. Demonstrating streaming generation...")
    stream_prompt = "The key to machine learning is"
    print(f"Streaming: {stream_prompt}", end='', flush=True)
    
    try:
        for token in generator.stream_generate(stream_prompt, gen_config):
            print(token, end='', flush=True)
        print()  # New line
    except Exception as e:
        print(f"Error in streaming generation: {e}")
    
    # 8. Model analysis
    print("\n8. Model analysis...")
    
    # Count parameters by component
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: {total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # Expert utilization (if available)
    moe_layers = [layer for layer in model.h if hasattr(layer.mlp, 'router')]
    if moe_layers:
        print(f"  - MoE layers: {len(moe_layers)}")
        print(f"  - Experts per MoE layer: {model_config.moe['num_experts']}")
    
    print("\n✅ Basic usage example completed successfully!")
    print("\nNext steps:")
    print("  - Try training with a real dataset (OpenWebText)")
    print("  - Experiment with different model configurations")
    print("  - Explore the benchmarking utilities")
    print("  - Check out the advanced examples")


if __name__ == "__main__":
    main()