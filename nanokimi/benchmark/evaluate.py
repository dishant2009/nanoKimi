"""
Model evaluation framework for nanoKimi.

This module provides comprehensive evaluation metrics and benchmarking
utilities for comparing model performance.
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer

from ..model import KimiModel
from ..training.data import TokenDataset


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    perplexity: float
    bits_per_byte: float
    tokens_per_second: float
    memory_usage_mb: float
    loss: float
    
    # Generation quality metrics
    generation_length: Optional[float] = None
    repetition_rate: Optional[float] = None
    coherence_score: Optional[float] = None
    
    # Expert utilization (for MoE models)
    expert_utilization: Optional[Dict[str, float]] = None
    
    # Additional metrics
    additional_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        result = {
            'perplexity': self.perplexity,
            'bits_per_byte': self.bits_per_byte,
            'tokens_per_second': self.tokens_per_second,
            'memory_usage_mb': self.memory_usage_mb,
            'loss': self.loss,
        }
        
        if self.generation_length is not None:
            result['generation_length'] = self.generation_length
        if self.repetition_rate is not None:
            result['repetition_rate'] = self.repetition_rate
        if self.coherence_score is not None:
            result['coherence_score'] = self.coherence_score
        if self.expert_utilization is not None:
            result['expert_utilization'] = self.expert_utilization
        if self.additional_metrics is not None:
            result.update(self.additional_metrics)
            
        return result


class PerplexityMetric:
    """Calculate perplexity on a dataset."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def evaluate(
        self, 
        model: nn.Module, 
        dataset: TokenDataset,
        batch_size: int = 8,
        max_batches: Optional[int] = None,
        stride: int = 512,
    ) -> Tuple[float, float]:
        """
        Calculate perplexity on dataset.
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            max_batches: Maximum number of batches to evaluate
            stride: Stride for sliding window evaluation
            
        Returns:
            Tuple of (perplexity, loss)
        """
        model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Use sliding window for long sequences
                seq_len = input_ids.size(1)
                
                if seq_len <= model.config.block_size:
                    # Short sequence - evaluate directly
                    outputs = model(input_ids, labels=labels)
                    loss = outputs['loss']
                    
                    batch_tokens = (labels != -100).sum().item()
                    total_loss += loss.item() * batch_tokens
                    total_tokens += batch_tokens
                else:
                    # Long sequence - use sliding window
                    for start_pos in range(0, seq_len - model.config.block_size + 1, stride):
                        end_pos = start_pos + model.config.block_size
                        
                        window_input = input_ids[:, start_pos:end_pos]
                        window_labels = labels[:, start_pos:end_pos]
                        
                        # Only count loss on the last part of the window
                        if start_pos > 0:
                            window_labels[:, :-stride] = -100
                        
                        outputs = model(window_input, labels=window_labels)
                        loss = outputs['loss']
                        
                        window_tokens = (window_labels != -100).sum().item()
                        total_loss += loss.item() * window_tokens
                        total_tokens += window_tokens
                
                num_batches += 1
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        
        return perplexity, avg_loss


class GenerationQualityMetric:
    """Evaluate text generation quality."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def evaluate(
        self,
        model: nn.Module,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> Dict[str, float]:
        """
        Evaluate generation quality on prompts.
        
        Args:
            model: Model to evaluate
            prompts: List of prompts to generate from
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Dictionary of quality metrics
        """
        from ..inference import Generator, GenerationConfig
        
        generator = Generator(model, self.tokenizer)
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )
        
        generated_texts = []
        generation_lengths = []
        
        # Generate text for each prompt
        for prompt in prompts:
            generated = generator.generate_text(prompt, gen_config)
            generated_texts.append(generated)
            generation_lengths.append(len(self.tokenizer.encode(generated)))
        
        # Calculate metrics
        avg_length = np.mean(generation_lengths)
        repetition_rate = self._calculate_repetition_rate(generated_texts)
        
        return {
            'average_generation_length': avg_length,
            'repetition_rate': repetition_rate,
            'num_samples': len(prompts),
        }
    
    def _calculate_repetition_rate(self, texts: List[str]) -> float:
        """Calculate repetition rate in generated texts."""
        total_repetitions = 0
        total_ngrams = 0
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            
            # Check for 4-gram repetitions
            ngram_size = 4
            if len(tokens) < ngram_size * 2:
                continue
            
            ngrams = {}
            for i in range(len(tokens) - ngram_size + 1):
                ngram = tuple(tokens[i:i + ngram_size])
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
                total_ngrams += 1
            
            # Count repetitions
            for count in ngrams.values():
                if count > 1:
                    total_repetitions += count - 1
        
        return total_repetitions / total_ngrams if total_ngrams > 0 else 0.0


class PerformanceMetric:
    """Measure model performance (speed, memory)."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def evaluate_inference_speed(
        self,
        model: nn.Module,
        dataset: TokenDataset,
        batch_size: int = 1,
        num_batches: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate inference speed.
        
        Args:
            model: Model to evaluate
            dataset: Dataset for evaluation
            batch_size: Batch size
            num_batches: Number of batches to test
            
        Returns:
            Performance metrics
        """
        model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                batch = next(iter(dataloader))
                input_ids = batch['input_ids'].to(self.device)
                _ = model(input_ids)
        
        # Measure performance
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                _ = model(input_ids)
                
                total_tokens += input_ids.numel()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time
        
        return {
            'tokens_per_second': tokens_per_second,
            'elapsed_time': elapsed_time,
            'total_tokens': total_tokens,
        }
    
    def measure_memory_usage(self, model: nn.Module) -> Dict[str, float]:
        """Measure model memory usage."""
        # Model parameters memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Buffer memory
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_memory = param_memory + buffer_memory
        
        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            }
        
        return {
            'model_memory_mb': total_memory / 1024**2,
            'param_memory_mb': param_memory / 1024**2,
            'buffer_memory_mb': buffer_memory / 1024**2,
            **gpu_memory,
        }


class ExpertUtilizationMetric:
    """Measure expert utilization in MoE models."""
    
    def evaluate(
        self,
        model: nn.Module,
        dataset: TokenDataset,
        batch_size: int = 8,
        num_batches: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate expert utilization.
        
        Args:
            model: MoE model to evaluate
            dataset: Dataset for evaluation
            batch_size: Batch size
            num_batches: Number of batches
            
        Returns:
            Expert utilization statistics
        """
        model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        expert_counts = {}
        total_tokens = 0
        
        def hook_fn(module, input, output):
            """Hook to capture expert routing decisions."""
            if hasattr(module, 'router') and hasattr(module.router, 'gate'):
                # Get router logits from the last forward pass
                if hasattr(module, '_last_router_logits'):
                    router_logits = module._last_router_logits
                    router_probs = F.softmax(router_logits, dim=-1)
                    
                    # Count expert assignments
                    expert_assignments = torch.argmax(router_probs, dim=-1)
                    for expert_id in expert_assignments.flatten():
                        expert_id = expert_id.item()
                        expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1
        
        # Register hooks for MoE layers
        hooks = []
        for name, module in model.named_modules():
            if 'moe' in name.lower() or 'expert' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Evaluate
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(model.device)
                _ = model(input_ids)
                
                total_tokens += input_ids.numel()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate utilization statistics
        if expert_counts:
            total_assignments = sum(expert_counts.values())
            num_experts = len(expert_counts)
            
            utilization_rates = {
                f'expert_{i}': expert_counts.get(i, 0) / total_assignments
                for i in range(num_experts)
            }
            
            # Overall statistics
            utilization_values = list(utilization_rates.values())
            
            return {
                'num_experts': num_experts,
                'average_utilization': np.mean(utilization_values),
                'utilization_std': np.std(utilization_values),
                'min_utilization': np.min(utilization_values),
                'max_utilization': np.max(utilization_values),
                'active_experts': sum(1 for v in utilization_values if v > 0.01),
                **utilization_rates,
            }
        
        return {'error': 'No expert routing detected'}


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize metrics
        self.perplexity_metric = PerplexityMetric(self.device)
        self.performance_metric = PerformanceMetric(self.device)
        self.expert_metric = ExpertUtilizationMetric()
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataset: TokenDataset,
        tokenizer: Optional[AutoTokenizer] = None,
        batch_size: int = 8,
        max_batches: Optional[int] = None,
        include_generation: bool = True,
        generation_prompts: Optional[List[str]] = None,
    ) -> EvaluationResults:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            tokenizer: Tokenizer for generation tasks
            batch_size: Batch size for evaluation
            max_batches: Maximum batches to evaluate
            include_generation: Whether to include generation metrics
            generation_prompts: Prompts for generation evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        model.to(self.device)
        
        # Language modeling evaluation
        print("Evaluating perplexity...")
        perplexity, loss = self.perplexity_metric.evaluate(
            model, dataset, batch_size, max_batches
        )
        
        # Performance evaluation
        print("Evaluating performance...")
        perf_metrics = self.performance_metric.evaluate_inference_speed(
            model, dataset, batch_size, min(10, max_batches or 10)
        )
        
        memory_metrics = self.performance_metric.measure_memory_usage(model)
        
        # Expert utilization (for MoE models)
        expert_utilization = None
        try:
            expert_utilization = self.expert_metric.evaluate(
                model, dataset, batch_size, min(5, max_batches or 5)
            )
        except Exception:
            pass  # Not a MoE model or evaluation failed
        
        # Generation quality evaluation
        generation_metrics = {}
        if include_generation and tokenizer:
            print("Evaluating generation quality...")
            if generation_prompts is None:
                generation_prompts = [
                    "The future of artificial intelligence",
                    "In a world where technology",
                    "Scientists have discovered",
                    "The most important thing in life",
                    "Climate change is",
                ]
            
            gen_metric = GenerationQualityMetric(tokenizer)
            generation_metrics = gen_metric.evaluate(model, generation_prompts)
        
        # Calculate bits per byte (approximate)
        bits_per_byte = loss / math.log(2)
        
        return EvaluationResults(
            perplexity=perplexity,
            bits_per_byte=bits_per_byte,
            tokens_per_second=perf_metrics['tokens_per_second'],
            memory_usage_mb=memory_metrics['model_memory_mb'],
            loss=loss,
            generation_length=generation_metrics.get('average_generation_length'),
            repetition_rate=generation_metrics.get('repetition_rate'),
            expert_utilization=expert_utilization,
            additional_metrics={
                **perf_metrics,
                **memory_metrics,
                **generation_metrics,
            }
        )
    
    def save_results(self, results: EvaluationResults, filepath: str):
        """Save evaluation results to file."""
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def benchmark_model(
    model: nn.Module,
    dataset: TokenDataset,
    config_name: str = "unknown",
    output_dir: str = "./benchmarks",
    **kwargs
) -> EvaluationResults:
    """
    Convenience function for benchmarking a model.
    
    Args:
        model: Model to benchmark
        dataset: Evaluation dataset
        config_name: Name of model configuration
        output_dir: Directory to save results
        **kwargs: Additional arguments for evaluation
        
    Returns:
        Evaluation results
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, dataset, **kwargs)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{config_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    evaluator.save_results(results, filepath)
    
    return results