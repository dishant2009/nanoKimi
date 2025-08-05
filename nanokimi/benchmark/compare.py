"""
Model comparison framework for nanoKimi vs baselines.

This module provides utilities for comparing nanoKimi against other models
like nanoGPT, standard transformers, and other baselines.
"""

import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

from .evaluate import ModelEvaluator, EvaluationResults
from ..training.data import TokenDataset


@dataclass
class ComparisonResults:
    """Results from model comparison."""
    
    nanokimi_results: EvaluationResults
    baseline_results: Dict[str, EvaluationResults]
    
    # Comparative metrics
    improvements: Dict[str, float]
    efficiency_ratios: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'nanokimi': self.nanokimi_results.to_dict(),
            'baselines': {name: results.to_dict() for name, results in self.baseline_results.items()},
            'improvements': self.improvements,
            'efficiency_ratios': self.efficiency_ratios,
        }


class NanoGPTComparison:
    """
    Comparison framework specifically for nanoGPT.
    
    This class handles loading nanoGPT models and comparing them
    with nanoKimi on various metrics.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator(self.device)
    
    def load_nanogpt_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load nanoGPT model from checkpoint.
        
        Args:
            checkpoint_path: Path to nanoGPT checkpoint
            
        Returns:
            Loaded nanoGPT model
        """
        try:
            # Try to import nanoGPT (assumes it's installed or available)
            import sys
            nanogpt_path = os.environ.get('NANOGPT_PATH', './nanoGPT')
            if os.path.exists(nanogpt_path):
                sys.path.append(nanogpt_path)
            
            from model import GPTConfig, GPT
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract config
            config_keys = [k for k in checkpoint.keys() if k.startswith('model_args')]
            if config_keys:
                model_args = checkpoint[config_keys[0]]
                gptconf = GPTConfig(**model_args)
            else:
                # Default config if not found
                gptconf = GPTConfig()
            
            model = GPT(gptconf)
            model.load_state_dict(checkpoint['model'])
            model.to(self.device)
            
            return model
            
        except ImportError:
            raise ImportError(
                "nanoGPT not found. Please install or set NANOGPT_PATH environment variable."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load nanoGPT model: {e}")
    
    def create_equivalent_nanogpt(self, nanokimi_config) -> nn.Module:
        """
        Create nanoGPT model with equivalent parameters to nanoKimi.
        
        Args:
            nanokimi_config: nanoKimi model configuration
            
        Returns:
            Equivalent nanoGPT model (untrained)
        """
        try:
            from model import GPTConfig, GPT
            
            # Map nanoKimi config to nanoGPT config
            gpt_config = GPTConfig(
                block_size=nanokimi_config.block_size,
                vocab_size=nanokimi_config.vocab_size,
                n_layer=nanokimi_config.n_layer,
                n_head=nanokimi_config.n_head,
                n_embd=nanokimi_config.n_embd,
                dropout=nanokimi_config.dropout,
                bias=True,  # nanoGPT default
            )
            
            model = GPT(gpt_config)
            model.to(self.device)
            
            return model
            
        except ImportError:
            # Create a simplified GPT-like model if nanoGPT is not available
            return self._create_simple_gpt(nanokimi_config)
    
    def _create_simple_gpt(self, config) -> nn.Module:
        """Create a simple GPT-like model for comparison."""
        class SimpleGPT(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Token and position embeddings
                self.wte = nn.Embedding(config.vocab_size, config.n_embd)
                self.wpe = nn.Embedding(config.block_size, config.n_embd)
                self.drop = nn.Dropout(config.dropout)
                
                # Transformer blocks
                self.h = nn.ModuleList([self._make_block(config) for _ in range(config.n_layer)])
                
                # Final layer norm and output
                self.ln_f = nn.LayerNorm(config.n_embd)
                self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
                
                # Tie weights
                self.lm_head.weight = self.wte.weight
                
                # Initialize
                self.apply(self._init_weights)
            
            def _make_block(self, config):
                class Block(nn.Module):
                    def __init__(self, config):
                        super().__init__()
                        self.ln_1 = nn.LayerNorm(config.n_embd)
                        self.attn = nn.MultiheadAttention(
                            config.n_embd, config.n_head, dropout=config.dropout, batch_first=True
                        )
                        self.ln_2 = nn.LayerNorm(config.n_embd)
                        self.mlp = nn.Sequential(
                            nn.Linear(config.n_embd, 4 * config.n_embd),
                            nn.GELU(),
                            nn.Linear(4 * config.n_embd, config.n_embd),
                            nn.Dropout(config.dropout),
                        )
                    
                    def forward(self, x):
                        # Self-attention with causal mask
                        seq_len = x.size(1)
                        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
                        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))
                        
                        attn_out, _ = self.attn(
                            self.ln_1(x), self.ln_1(x), self.ln_1(x),
                            attn_mask=causal_mask
                        )
                        x = x + attn_out
                        
                        # MLP
                        x = x + self.mlp(self.ln_2(x))
                        return x
                
                return Block(config)
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            def forward(self, input_ids, labels=None):
                b, t = input_ids.size()
                pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
                
                # Embeddings
                tok_emb = self.wte(input_ids)
                pos_emb = self.wpe(pos)
                x = self.drop(tok_emb + pos_emb)
                
                # Transformer blocks
                for block in self.h:
                    x = block(x)
                
                # Output
                x = self.ln_f(x)
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                
                return {'logits': logits, 'loss': loss}
            
            def get_num_params(self):
                return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return SimpleGPT(config)
    
    def compare_models(
        self,
        nanokimi_model: nn.Module,
        baseline_model: nn.Module,
        dataset: TokenDataset,
        tokenizer: Optional[AutoTokenizer] = None,
        baseline_name: str = "nanoGPT",
        **eval_kwargs
    ) -> ComparisonResults:
        """
        Compare nanoKimi with baseline model.
        
        Args:
            nanokimi_model: nanoKimi model
            baseline_model: Baseline model (e.g., nanoGPT)
            dataset: Evaluation dataset
            tokenizer: Tokenizer for generation tasks
            baseline_name: Name of baseline model
            **eval_kwargs: Additional evaluation arguments
            
        Returns:
            Comprehensive comparison results
        """
        print(f"Comparing nanoKimi vs {baseline_name}...")
        
        # Evaluate nanoKimi
        print("Evaluating nanoKimi...")
        nanokimi_results = self.evaluator.evaluate_model(
            nanokimi_model, dataset, tokenizer, **eval_kwargs
        )
        
        # Evaluate baseline
        print(f"Evaluating {baseline_name}...")
        baseline_results = self.evaluator.evaluate_model(
            baseline_model, dataset, tokenizer, **eval_kwargs
        )
        
        # Calculate improvements and efficiency ratios
        improvements = self._calculate_improvements(nanokimi_results, baseline_results)
        efficiency_ratios = self._calculate_efficiency_ratios(
            nanokimi_model, baseline_model, nanokimi_results, baseline_results
        )
        
        return ComparisonResults(
            nanokimi_results=nanokimi_results,
            baseline_results={baseline_name: baseline_results},
            improvements=improvements,
            efficiency_ratios=efficiency_ratios,
        )
    
    def _calculate_improvements(
        self, 
        nanokimi_results: EvaluationResults, 
        baseline_results: EvaluationResults
    ) -> Dict[str, float]:
        """Calculate performance improvements."""
        improvements = {}
        
        # Lower is better metrics
        lower_better = ['perplexity', 'loss', 'bits_per_byte', 'repetition_rate']
        for metric in lower_better:
            nanokimi_val = getattr(nanokimi_results, metric, None)
            baseline_val = getattr(baseline_results, metric, None)
            
            if nanokimi_val is not None and baseline_val is not None and baseline_val > 0:
                improvement = (baseline_val - nanokimi_val) / baseline_val * 100
                improvements[f'{metric}_improvement_percent'] = improvement
        
        # Higher is better metrics
        higher_better = ['tokens_per_second']
        for metric in higher_better:
            nanokimi_val = getattr(nanokimi_results, metric, None)
            baseline_val = getattr(baseline_results, metric, None)
            
            if nanokimi_val is not None and baseline_val is not None and baseline_val > 0:
                improvement = (nanokimi_val - baseline_val) / baseline_val * 100
                improvements[f'{metric}_improvement_percent'] = improvement
        
        return improvements
    
    def _calculate_efficiency_ratios(
        self,
        nanokimi_model: nn.Module,
        baseline_model: nn.Module,
        nanokimi_results: EvaluationResults,
        baseline_results: EvaluationResults,
    ) -> Dict[str, float]:
        """Calculate efficiency ratios (performance per parameter, etc.)."""
        ratios = {}
        
        # Get parameter counts
        nanokimi_params = nanokimi_model.get_num_params() if hasattr(nanokimi_model, 'get_num_params') else sum(p.numel() for p in nanokimi_model.parameters())
        baseline_params = baseline_model.get_num_params() if hasattr(baseline_model, 'get_num_params') else sum(p.numel() for p in baseline_model.parameters())
        
        # Parameter efficiency (lower perplexity per parameter is better)
        if baseline_results.perplexity > 0 and nanokimi_results.perplexity > 0:
            nanokimi_ppl_per_param = nanokimi_results.perplexity / nanokimi_params * 1e6
            baseline_ppl_per_param = baseline_results.perplexity / baseline_params * 1e6
            ratios['perplexity_per_million_params_ratio'] = baseline_ppl_per_param / nanokimi_ppl_per_param
        
        # Speed efficiency (tokens per second per parameter)
        nanokimi_speed_per_param = nanokimi_results.tokens_per_second / nanokimi_params * 1e6
        baseline_speed_per_param = baseline_results.tokens_per_second / baseline_params * 1e6
        if baseline_speed_per_param > 0:
            ratios['speed_per_million_params_ratio'] = nanokimi_speed_per_param / baseline_speed_per_param
        
        # Memory efficiency
        nanokimi_memory_per_param = nanokimi_results.memory_usage_mb / nanokimi_params * 1e6
        baseline_memory_per_param = baseline_results.memory_usage_mb / baseline_params * 1e6
        if baseline_memory_per_param > 0:
            ratios['memory_per_million_params_ratio'] = baseline_memory_per_param / nanokimi_memory_per_param
        
        # Overall parameter ratio
        ratios['parameter_count_ratio'] = nanokimi_params / baseline_params
        
        return ratios


class ModelComparisonSuite:
    """Comprehensive model comparison suite."""
    
    def __init__(self, output_dir: str = "./comparisons"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.nanogpt_comparison = NanoGPTComparison()
    
    def run_comprehensive_comparison(
        self,
        nanokimi_model: nn.Module,
        dataset: TokenDataset,
        baseline_checkpoints: Optional[Dict[str, str]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        **eval_kwargs
    ) -> Dict[str, ComparisonResults]:
        """
        Run comprehensive comparison against multiple baselines.
        
        Args:
            nanokimi_model: nanoKimi model to evaluate
            dataset: Evaluation dataset
            baseline_checkpoints: Dict of {name: checkpoint_path}
            tokenizer: Tokenizer for generation
            **eval_kwargs: Additional evaluation arguments
            
        Returns:
            Comparison results for each baseline
        """
        all_results = {}
        
        # Compare against nanoGPT (if available)
        if baseline_checkpoints and 'nanogpt' in baseline_checkpoints:
            try:
                nanogpt_model = self.nanogpt_comparison.load_nanogpt_model(
                    baseline_checkpoints['nanogpt']
                )
                results = self.nanogpt_comparison.compare_models(
                    nanokimi_model, nanogpt_model, dataset, tokenizer, **eval_kwargs
                )
                all_results['nanoGPT'] = results
            except Exception as e:
                print(f"Failed to compare with nanoGPT: {e}")
        
        # Compare against equivalent untrained model
        try:
            equivalent_model = self.nanogpt_comparison.create_equivalent_nanogpt(
                nanokimi_model.config
            )
            results = self.nanogpt_comparison.compare_models(
                nanokimi_model, equivalent_model, dataset, tokenizer, 
                baseline_name="Untrained GPT", **eval_kwargs
            )
            all_results['Untrained GPT'] = results
        except Exception as e:
            print(f"Failed to create equivalent model: {e}")
        
        # Save all results
        self._save_comparison_results(all_results)
        
        return all_results
    
    def _save_comparison_results(self, results: Dict[str, ComparisonResults]):
        """Save comparison results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for baseline_name, comparison in results.items():
            filename = f"comparison_{baseline_name.lower().replace(' ', '_')}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(comparison.to_dict(), f, indent=2)
            
            print(f"Comparison with {baseline_name} saved to {filepath}")
    
    def generate_comparison_report(
        self, 
        results: Dict[str, ComparisonResults],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate human-readable comparison report.
        
        Args:
            results: Comparison results
            output_file: Optional file to save report
            
        Returns:
            Report text
        """
        report_lines = ["# nanoKimi Model Comparison Report\n"]
        report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for baseline_name, comparison in results.items():
            report_lines.append(f"\n## Comparison with {baseline_name}")
            report_lines.append("=" * 50)
            
            # Model parameters
            nanokimi_results = comparison.nanokimi_results
            baseline_results = list(comparison.baseline_results.values())[0]
            
            report_lines.append(f"\n### Performance Metrics")
            report_lines.append(f"| Metric | nanoKimi | {baseline_name} | Improvement |")
            report_lines.append(f"|--------|----------|-------------|-------------|")
            
            metrics = [
                ('Perplexity', 'perplexity', '{:.2f}'),
                ('Loss', 'loss', '{:.4f}'),
                ('Tokens/sec', 'tokens_per_second', '{:.1f}'),
                ('Memory (MB)', 'memory_usage_mb', '{:.1f}'),
            ]
            
            for metric_name, attr_name, fmt in metrics:
                nanokimi_val = getattr(nanokimi_results, attr_name)
                baseline_val = getattr(baseline_results, attr_name)
                
                if nanokimi_val is not None and baseline_val is not None:
                    improvement = comparison.improvements.get(f'{attr_name}_improvement_percent', 0)
                    improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "0%"
                    
                    report_lines.append(
                        f"| {metric_name} | {fmt.format(nanokimi_val)} | "
                        f"{fmt.format(baseline_val)} | {improvement_str} |"
                    )
            
            # Efficiency ratios
            if comparison.efficiency_ratios:
                report_lines.append(f"\n### Efficiency Ratios")
                for ratio_name, ratio_value in comparison.efficiency_ratios.items():
                    formatted_name = ratio_name.replace('_', ' ').title()
                    report_lines.append(f"- {formatted_name}: {ratio_value:.2f}x")
            
            # Expert utilization (if available)
            if (hasattr(nanokimi_results, 'expert_utilization') and 
                nanokimi_results.expert_utilization):
                report_lines.append(f"\n### Expert Utilization (nanoKimi)")
                util = nanokimi_results.expert_utilization
                report_lines.append(f"- Number of experts: {util.get('num_experts', 'N/A')}")
                report_lines.append(f"- Average utilization: {util.get('average_utilization', 0):.3f}")
                report_lines.append(f"- Active experts: {util.get('active_experts', 'N/A')}")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text
    
    def plot_comparison_charts(
        self, 
        results: Dict[str, ComparisonResults],
        save_dir: Optional[str] = None
    ):
        """Generate comparison charts."""
        if save_dir is None:
            save_dir = self.output_dir
        
        # Collect data for plotting
        models = ['nanoKimi']
        perplexities = []
        speeds = []
        memory_usage = []
        
        # Add nanoKimi data
        nanokimi_results = next(iter(results.values())).nanokimi_results
        perplexities.append(nanokimi_results.perplexity)
        speeds.append(nanokimi_results.tokens_per_second)
        memory_usage.append(nanokimi_results.memory_usage_mb)
        
        # Add baseline data
        for baseline_name, comparison in results.items():
            models.append(baseline_name)
            baseline_results = list(comparison.baseline_results.values())[0]
            perplexities.append(baseline_results.perplexity)
            speeds.append(baseline_results.tokens_per_second)
            memory_usage.append(baseline_results.memory_usage_mb)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Perplexity comparison
        axes[0, 0].bar(models, perplexities, color=['#1f77b4'] + ['#ff7f0e'] * (len(models) - 1))
        axes[0, 0].set_title('Perplexity Comparison (Lower is Better)')
        axes[0, 0].set_ylabel('Perplexity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Speed comparison
        axes[0, 1].bar(models, speeds, color=['#1f77b4'] + ['#ff7f0e'] * (len(models) - 1))
        axes[0, 1].set_title('Speed Comparison (Higher is Better)')
        axes[0, 1].set_ylabel('Tokens/Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        axes[1, 0].bar(models, memory_usage, color=['#1f77b4'] + ['#ff7f0e'] * (len(models) - 1))
        axes[1, 0].set_title('Memory Usage Comparison')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Efficiency scatter plot (perplexity vs speed)
        axes[1, 1].scatter(perplexities, speeds, s=100, 
                          c=['#1f77b4'] + ['#ff7f0e'] * (len(models) - 1))
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (perplexities[i], speeds[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Perplexity')
        axes[1, 1].set_ylabel('Tokens/Second')
        axes[1, 1].set_title('Efficiency: Perplexity vs Speed')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_filename = f"comparison_charts_{timestamp}.png"
        plot_filepath = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison charts saved to {plot_filepath}")


def run_comparison_benchmark(
    nanokimi_checkpoint: str,
    dataset_path: str,
    output_dir: str = "./comparisons",
    nanogpt_checkpoint: Optional[str] = None,
    **kwargs
) -> Dict[str, ComparisonResults]:
    """
    Convenience function to run full comparison benchmark.
    
    Args:
        nanokimi_checkpoint: Path to nanoKimi checkpoint
        dataset_path: Path to evaluation dataset
        output_dir: Output directory for results
        nanogpt_checkpoint: Optional path to nanoGPT checkpoint
        **kwargs: Additional arguments
        
    Returns:
        Comparison results
    """
    from ..model import KimiModel, KimiConfig
    from ..training.data import TokenDataset
    from transformers import AutoTokenizer
    
    # Load nanoKimi model
    checkpoint = torch.load(nanokimi_checkpoint, map_location='cpu')
    config = KimiConfig.from_dict(checkpoint['config'])
    model = KimiModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load dataset
    dataset = TokenDataset(dataset_path, block_size=config.block_size)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup comparison suite
    comparison_suite = ModelComparisonSuite(output_dir)
    
    # Prepare baseline checkpoints
    baseline_checkpoints = {}
    if nanogpt_checkpoint:
        baseline_checkpoints['nanogpt'] = nanogpt_checkpoint
    
    # Run comparison
    results = comparison_suite.run_comprehensive_comparison(
        model, dataset, baseline_checkpoints, tokenizer, **kwargs
    )
    
    # Generate report and charts
    report = comparison_suite.generate_comparison_report(results)
    print("\n" + report)
    
    comparison_suite.plot_comparison_charts(results)
    
    return results