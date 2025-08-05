#!/usr/bin/env python3
"""
Flash Attention demonstration for nanoKimi.

This script demonstrates the performance benefits of Flash Attention
and shows how to use different attention implementations.
"""

import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import nanokimi
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nanokimi.model import KimiModel, KimiConfig
from nanokimi.model.flash_attention import (
    FlashAttention, FlashMHA, get_attention_implementation,
    is_flash_attention_available, is_xformers_available, get_attention_info
)
from nanokimi.model.attention import StandardAttention


def benchmark_attention_implementations():
    """Benchmark different attention implementations."""
    print("Flash Attention Performance Demonstration")
    print("=" * 50)
    
    # Check available implementations
    info = get_attention_info()
    print(f"Available implementations:")
    print(f"  Flash Attention: {info['flash_attention']}")
    print(f"  xFormers: {info['xformers']}")
    print(f"  Recommended: {info['recommended']}")
    print()
    
    # Test configurations
    configs = [
        {"name": "Small", "n_embd": 256, "n_head": 8, "seq_len": 512, "batch_size": 4},
        {"name": "Medium", "n_embd": 512, "n_head": 8, "seq_len": 1024, "batch_size": 2},
        {"name": "Large", "n_embd": 768, "n_head": 12, "seq_len": 2048, "batch_size": 1},
    ]
    
    # Only test if we have GPU and appropriate packages
    if not torch.cuda.is_available():
        print("CUDA not available. Running CPU benchmark (limited)...")
        device = torch.device('cpu')
        configs = configs[:1]  # Only test small config on CPU
    else:
        device = torch.device('cuda')
        print(f"Using device: {device}")
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration:")
        print(f"  Embedding: {config['n_embd']}, Heads: {config['n_head']}")
        print(f"  Sequence length: {config['seq_len']}, Batch size: {config['batch_size']}")
        
        # Create test data
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        n_embd = config['n_embd']
        n_head = config['n_head']
        
        x = torch.randn(batch_size, seq_len, n_embd, device=device, dtype=torch.float16)
        
        # Test different implementations
        implementations = []
        
        # Standard attention
        try:
            model_config = KimiConfig(
                n_embd=n_embd, 
                n_head=n_head, 
                block_size=seq_len,
                attention={"type": "standard"}
            )
            standard_attn = StandardAttention(model_config).to(device)
            
            time_taken = benchmark_implementation(standard_attn, x, "Standard")
            implementations.append(("Standard", time_taken))
            
        except Exception as e:
            print(f"  Standard attention failed: {e}")
        
        # Flash attention
        if is_flash_attention_available() and device.type == 'cuda':
            try:
                flash_attn = FlashMHA(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=0.0,
                    causal=True,
                    block_size=seq_len,
                    use_flash=True
                ).to(device)
                
                time_taken = benchmark_implementation(flash_attn, x, "Flash")
                implementations.append(("Flash", time_taken))
                
            except Exception as e:
                print(f"  Flash attention failed: {e}")
        
        # xFormers attention
        if is_xformers_available():
            try:
                xformers_attn = FlashMHA(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=0.0,
                    causal=True,
                    block_size=seq_len,
                    use_flash=False  # Will use xFormers
                ).to(device)
                
                time_taken = benchmark_implementation(xformers_attn, x, "xFormers")
                implementations.append(("xFormers", time_taken))
                
            except Exception as e:
                print(f"  xFormers attention failed: {e}")
        
        # Auto selection
        try:
            auto_attn = get_attention_implementation(
                n_embd=n_embd,
                n_head=n_head,
                dropout=0.0,
                causal=True,
                block_size=seq_len,
                attention_type="auto"
            ).to(device)
            
            time_taken = benchmark_implementation(auto_attn, x, "Auto")
            implementations.append(("Auto", time_taken))
            
        except Exception as e:
            print(f"  Auto attention failed: {e}")
        
        results.append({
            "config": config,
            "implementations": implementations
        })
    
    return results


def benchmark_implementation(attention_module, x, name):
    """Benchmark a single attention implementation."""
    attention_module.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            try:
                output, _ = attention_module(x)
            except Exception:
                # Some implementations might not support the exact interface
                output = attention_module(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    num_runs = 10
    with torch.no_grad():
        for _ in range(num_runs):
            try:
                output, _ = attention_module(x)
            except Exception:
                output = attention_module(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"  {name}: {avg_time*1000:.2f} ms/forward")
    
    return avg_time


def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency of Flash Attention."""
    print("\nMemory Efficiency Demonstration")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping memory test.")
        return
    
    device = torch.device('cuda')
    
    # Test configurations with increasing sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]
    n_embd = 512
    n_head = 8
    batch_size = 1
    
    memory_usage = {"Standard": [], "Flash": [], "Sequence Length": seq_lengths}
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # Test standard attention memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            model_config = KimiConfig(
                n_embd=n_embd,
                n_head=n_head,
                block_size=seq_len,
                attention={"type": "standard"}
            )
            standard_attn = StandardAttention(model_config).to(device)
            
            x = torch.randn(batch_size, seq_len, n_embd, device=device, dtype=torch.float16)
            
            with torch.no_grad():
                output, _ = standard_attn(x)
            
            standard_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"  Standard attention: {standard_memory:.1f} MB")
            memory_usage["Standard"].append(standard_memory)
            
        except Exception as e:
            print(f"  Standard attention failed: {e}")
            memory_usage["Standard"].append(None)
        
        # Test Flash attention memory usage
        if is_flash_attention_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                flash_attn = FlashMHA(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=0.0,
                    causal=True,
                    block_size=seq_len,
                    use_flash=True
                ).to(device)
                
                x = torch.randn(batch_size, seq_len, n_embd, device=device, dtype=torch.float16)
                
                with torch.no_grad():
                    output, _ = flash_attn(x)
                
                flash_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                print(f"  Flash attention: {flash_memory:.1f} MB")
                memory_usage["Flash"].append(flash_memory)
                
                if memory_usage["Standard"][-1] is not None:
                    reduction = (memory_usage["Standard"][-1] - flash_memory) / memory_usage["Standard"][-1] * 100
                    print(f"  Memory reduction: {reduction:.1f}%")
                
            except Exception as e:
                print(f"  Flash attention failed: {e}")
                memory_usage["Flash"].append(None)
        else:
            memory_usage["Flash"].append(None)
    
    return memory_usage


def demonstrate_model_training():
    """Demonstrate training with Flash Attention."""
    print("\nTraining with Flash Attention")
    print("=" * 35)
    
    # Create model with Flash Attention
    config = KimiConfig(
        n_layer=2,
        n_embd=128,
        n_head=8,
        vocab_size=1000,
        block_size=256,
        attention={
            "type": "flash",
            "use_flash_attention": True,
            "use_rope": False,
        }
    )
    
    model = KimiModel(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model created with Flash Attention")
    print(f"Device: {device}")
    print(f"Parameters: {model.get_num_params():,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Simulate training
    model.train()
    
    for step in range(5):
        # Create random batch
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Forward pass
        start_time = time.time()
        outputs = model(input_ids, labels=labels)
        forward_time = time.time() - start_time
        
        loss = outputs['loss']
        
        # Backward pass
        start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - start_time
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}, "
              f"Forward = {forward_time*1000:.1f}ms, "
              f"Backward = {backward_time*1000:.1f}ms")
    
    print("Training simulation completed!")


def plot_results(results, memory_usage=None):
    """Plot benchmark results."""
    print("\nGenerating performance plots...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Performance comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Speed comparison
        config_names = [r['config']['name'] for r in results]
        implementations = list(set([impl[0] for r in results for impl in r['implementations']]))
        
        for impl in implementations:
            times = []
            for result in results:
                impl_data = [impl_time for impl_name, impl_time in result['implementations'] if impl_name == impl]
                times.append(impl_data[0] if impl_data else None)
            
            # Filter out None values
            valid_configs = [name for name, time in zip(config_names, times) if time is not None]
            valid_times = [time for time in times if time is not None]
            
            if valid_times:
                axes[0].bar([f"{name}\n{impl}" for name in valid_configs], 
                           [t * 1000 for t in valid_times], 
                           label=impl, alpha=0.7)
        
        axes[0].set_title('Attention Implementation Speed Comparison')
        axes[0].set_ylabel('Time (ms)')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Memory usage plot
        if memory_usage and any(memory_usage.get("Standard", [])) and any(memory_usage.get("Flash", [])):
            seq_lengths = memory_usage["Sequence Length"]
            standard_mem = memory_usage["Standard"]
            flash_mem = memory_usage["Flash"]
            
            # Filter out None values
            valid_indices = [i for i in range(len(seq_lengths)) 
                           if standard_mem[i] is not None and flash_mem[i] is not None]
            
            if valid_indices:
                valid_seq_lengths = [seq_lengths[i] for i in valid_indices]
                valid_standard = [standard_mem[i] for i in valid_indices]
                valid_flash = [flash_mem[i] for i in valid_indices]
                
                axes[1].plot(valid_seq_lengths, valid_standard, 'o-', label='Standard', linewidth=2)
                axes[1].plot(valid_seq_lengths, valid_flash, 's-', label='Flash', linewidth=2)
                axes[1].set_title('Memory Usage vs Sequence Length')
                axes[1].set_xlabel('Sequence Length')
                axes[1].set_ylabel('Memory Usage (MB)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Memory data not available\n(requires CUDA)', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Memory Usage Comparison')
        
        plt.tight_layout()
        plt.savefig('flash_attention_benchmark.png', dpi=300, bbox_inches='tight')
        print("Results saved to flash_attention_benchmark.png")
        
    except ImportError:
        print("Matplotlib not available. Skipping plots.")


def main():
    """Main demonstration function."""
    print(" nanoKimi Flash Attention Demonstration\n")
    
    # Check system capabilities
    print("System Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name()}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    attention_info = get_attention_info()
    print(f"  Flash Attention: {attention_info['flash_attention']}")
    print(f"  xFormers: {attention_info['xformers']}")
    print()
    
    # Run benchmarks
    results = benchmark_attention_implementations()
    
    # Memory efficiency test
    memory_usage = demonstrate_memory_efficiency()
    
    # Training demonstration
    demonstrate_model_training()
    
    # Generate plots
    plot_results(results, memory_usage)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if attention_info['flash_attention']:
        print(" Flash Attention is available and provides:")
        print("   - Reduced memory usage (especially for long sequences)")
        print("   - Faster training and inference")
        print("   - Better numerical stability")
    else:
        print(" Flash Attention not available.")
        print("   Install with: pip install flash-attn --no-build-isolation")
    
    if attention_info['xformers']:
        print(" xFormers is available as a fallback")
    else:
        print(" Consider installing xFormers: pip install xformers")
    
    print(f"\nRecommended implementation: {attention_info['recommended']}")
    
    print("\nTo use Flash Attention in your models:")
    print('  config.attention["use_flash_attention"] = True')
    print('  config.attention["type"] = "flash"')


if __name__ == "__main__":
    main()
