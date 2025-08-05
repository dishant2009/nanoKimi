#!/usr/bin/env python3
"""
Text generation script for nanoKimi models.

This script provides a command-line interface for generating text
with trained nanoKimi models.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import nanokimi
sys.path.append(str(Path(__file__).parent.parent))

import torch
from nanokimi.inference import Generator, GenerationConfig, load_generator_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Generate text with nanoKimi model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="The future of artificial intelligence",
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Top-p (nucleus) sampling parameter'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda, mps)'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream generation token by token'
    )
    parser.add_argument(
        '--no-sample',
        action='store_true',
        help='Use greedy decoding instead of sampling'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='gpt2',
        help='Tokenizer to use'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='Repetition penalty'
    )
    parser.add_argument(
        '--length-penalty',
        type=float,
        default=1.0,
        help='Length penalty'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive generation mode'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
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
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    try:
        generator = load_generator_from_checkpoint(
            args.checkpoint,
            device=device,
            tokenizer_name=args.tokenizer
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
    )
    
    # Print generation settings
    print(f"\nGeneration Settings:")
    print(f"  Max tokens: {gen_config.max_new_tokens}")
    print(f"  Temperature: {gen_config.temperature}")
    print(f"  Top-k: {gen_config.top_k}")
    print(f"  Top-p: {gen_config.top_p}")
    print(f"  Sampling: {gen_config.do_sample}")
    print(f"  Repetition penalty: {gen_config.repetition_penalty}")
    
    def generate_single(prompt: str):
        """Generate text for a single prompt."""
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        if args.stream:
            # Streaming generation
            print(prompt, end='', flush=True)
            for token in generator.stream_generate(prompt, gen_config):
                print(token, end='', flush=True)
            print()  # New line at the end
        else:
            # Regular generation
            if args.num_samples == 1:
                generated_text = generator.generate_text(prompt, gen_config)
                print(f"{prompt}{generated_text}")
            else:
                # Multiple samples
                prompts = [prompt] * args.num_samples
                generated_texts = generator.batch_generate_text(prompts, gen_config)
                
                for i, text in enumerate(generated_texts, 1):
                    print(f"\nSample {i}:")
                    print(f"{prompt}{text}")
        
        print("\n" + "="*50)
    
    if args.interactive:
        # Interactive mode
        print("\nInteractive mode - type 'quit' to exit")
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if prompt:
                    generate_single(prompt)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        # Single generation
        generate_single(args.prompt)


if __name__ == "__main__":
    main()