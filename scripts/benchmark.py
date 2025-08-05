#!/usr/bin/env python3
"""
Benchmarking script for nanoKimi models.

This script provides comprehensive evaluation and comparison of nanoKimi models
against standard baselines on various benchmark datasets.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add the parent directory to the path so we can import nanokimi
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from nanokimi.model import KimiModel, KimiConfig
from nanokimi.benchmark import (
    ModelEvaluator, 
    NanoGPTComparison, 
    ModelComparisonSuite,
    BenchmarkDatasetManager,
    load_benchmark_dataset,
    run_comparison_benchmark,
)


def main():
    parser = argparse.ArgumentParser(description='Benchmark nanoKimi models')
    
    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to nanoKimi checkpoint to evaluate'
    )
    parser.add_argument(
        '--baseline-checkpoint',
        type=str,
        default=None,
        help='Path to baseline model checkpoint (e.g., nanoGPT)'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='wikitext103',
        choices=['wikitext103', 'penntreebank', 'lambada', 'hellaswag', 'toy'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'validation', 'test'],
        help='Dataset split to use'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to store/load datasets'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset if not found'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum number of batches to evaluate'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=None,
        help='Override block size for evaluation'
    )
    
    # Generation evaluation
    parser.add_argument(
        '--no-generation',
        action='store_true',
        help='Skip generation quality evaluation'
    )
    parser.add_argument(
        '--generation-prompts',
        type=str,
        nargs='+',
        default=None,
        help='Custom prompts for generation evaluation'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=50,
        help='Maximum tokens to generate for quality evaluation'
    )
    
    # Comparison arguments
    parser.add_argument(
        '--compare-baseline',
        action='store_true',
        help='Compare against baseline models'
    )
    parser.add_argument(
        '--create-equivalent',
        action='store_true',
        help='Create and compare against equivalent untrained model'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmarks',
        help='Directory to save benchmark results'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Generate and save comparison plots'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda, mps)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load nanoKimi model
    print(f"Loading nanoKimi model from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = KimiConfig.from_dict(checkpoint['config'])
        model = KimiModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"Model loaded successfully!")
        print(f"Model parameters: {model.get_num_params():,}")
        print(f"Model config: {config.n_layer} layers, {config.n_embd} embedding dim")
        
        if config.moe.get('use_moe', False):
            print(f"MoE enabled: {config.moe.get('num_experts')} experts, top-{config.moe.get('top_k')}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset: {args.dataset} ({args.split} split)")
    
    try:
        if args.dataset == 'toy':
            from nanokimi.training.data import create_datasets, DataConfig
            data_config = DataConfig(
                tokenizer_name="gpt2",
                block_size=args.block_size or config.block_size,
                data_dir=args.data_dir,
            )
            train_dataset, val_dataset = create_datasets(data_config, "toy")
            dataset = val_dataset  # Use validation set for evaluation
        else:
            dataset = load_benchmark_dataset(
                args.dataset,
                split=args.split,
                data_dir=args.data_dir,
                tokenizer_name="gpt2",
                block_size=args.block_size or config.block_size,
                download=args.download,
            )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if not args.download:
            print("Try using --download to download the dataset")
        sys.exit(1)
    
    # Run evaluation
    print(f"\nStarting evaluation...")
    evaluator = ModelEvaluator(device)
    
    # Prepare generation prompts
    generation_prompts = args.generation_prompts
    if generation_prompts is None and not args.no_generation:
        generation_prompts = [
            "The future of artificial intelligence",
            "In a world where technology",
            "Scientists have recently discovered",
            "The most important aspect of machine learning",
            "Climate change represents",
        ]
    
    # Evaluate nanoKimi model
    start_time = time.time()
    
    results = evaluator.evaluate_model(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer if not args.no_generation else None,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        include_generation=not args.no_generation,
        generation_prompts=generation_prompts,
    )
    
    eval_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print(f"NANOKIMI EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Model: {model.get_num_params():,} parameters")
    print(f"\nLanguage Modeling Metrics:")
    print(f"  Perplexity: {results.perplexity:.2f}")
    print(f"  Loss: {results.loss:.4f}")
    print(f"  Bits per byte: {results.bits_per_byte:.4f}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Tokens/second: {results.tokens_per_second:.1f}")
    print(f"  Memory usage: {results.memory_usage_mb:.1f} MB")
    
    if not args.no_generation and results.generation_length is not None:
        print(f"\nGeneration Quality:")
        print(f"  Average length: {results.generation_length:.1f} tokens")
        print(f"  Repetition rate: {results.repetition_rate:.3f}")
    
    if results.expert_utilization:
        print(f"\nExpert Utilization (MoE):")
        util = results.expert_utilization
        print(f"  Number of experts: {util.get('num_experts', 'N/A')}")
        print(f"  Average utilization: {util.get('average_utilization', 0):.3f}")
        print(f"  Active experts: {util.get('active_experts', 'N/A')}")
        print(f"  Utilization std: {util.get('utilization_std', 0):.3f}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"nanokimi_benchmark_{args.dataset}_{timestamp}.json"
    results_path = os.path.join(args.output_dir, results_filename)
    evaluator.save_results(results, results_path)
    print(f"\nResults saved to: {results_path}")
    
    # Run comparison if requested
    if args.compare_baseline or args.create_equivalent:
        print(f"\n{'='*60}")
        print(f"RUNNING BASELINE COMPARISON")
        print(f"{'='*60}")
        
        comparison_suite = ModelComparisonSuite(args.output_dir)
        
        baseline_checkpoints = {}
        if args.baseline_checkpoint:
            baseline_checkpoints['nanogpt'] = args.baseline_checkpoint
        
        try:
            comparison_results = comparison_suite.run_comprehensive_comparison(
                nanokimi_model=model,
                dataset=dataset,
                baseline_checkpoints=baseline_checkpoints if args.compare_baseline else None,
                tokenizer=tokenizer if not args.no_generation else None,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                include_generation=not args.no_generation,
                generation_prompts=generation_prompts,
            )
            
            # Generate and display report
            report = comparison_suite.generate_comparison_report(comparison_results)
            print(f"\n{report}")
            
            # Save report
            report_filename = f"comparison_report_{args.dataset}_{timestamp}.md"
            report_path = os.path.join(args.output_dir, report_filename)
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\nComparison report saved to: {report_path}")
            
            # Generate plots if requested
            if args.save_plots:
                comparison_suite.plot_comparison_charts(comparison_results, args.output_dir)
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"All results saved in: {args.output_dir}")
    
    # Summary recommendations
    print(f"\nNext steps:")
    print(f"  - Review detailed results in {results_path}")
    if args.compare_baseline or args.create_equivalent:
        print(f"  - Check comparison report for performance analysis")
    print(f"  - Try benchmarking on different datasets")
    print(f"  - Experiment with different model configurations")


if __name__ == "__main__":
    main()