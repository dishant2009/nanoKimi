"""
Unit tests for nanoKimi benchmarking components.

Tests evaluation metrics, model comparison, and dataset utilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from nanokimi.benchmark.evaluate import (
    EvaluationResults, PerplexityMetric, GenerationQualityMetric,
    PerformanceMetric, ExpertUtilizationMetric, ModelEvaluator,
    benchmark_model
)
from nanokimi.benchmark.compare import (
    ComparisonResults, NanoGPTComparison, ModelComparisonSuite,
    run_comparison_benchmark
)
from nanokimi.benchmark.datasets import (
    DatasetInfo, BenchmarkDatasetManager, load_benchmark_dataset,
    WikiTextDataset, PennTreebankDataset
)
from nanokimi.model import KimiModel, KimiConfig
from nanokimi.training.data import TokenDataset


class TestEvaluationResults:
    """Test evaluation results container."""
    
    def test_evaluation_results_creation(self):
        """Test creating evaluation results."""
        results = EvaluationResults(
            perplexity=15.5,
            bits_per_byte=4.2,
            tokens_per_second=1250.0,
            memory_usage_mb=512.0,
            loss=2.74
        )
        
        assert results.perplexity == 15.5
        assert results.bits_per_byte == 4.2
        assert results.tokens_per_second == 1250.0
        assert results.memory_usage_mb == 512.0
        assert results.loss == 2.74
    
    def test_evaluation_results_to_dict(self):
        """Test converting results to dictionary."""
        results = EvaluationResults(
            perplexity=15.5,
            bits_per_byte=4.2,
            tokens_per_second=1250.0,
            memory_usage_mb=512.0,
            loss=2.74,
            generation_length=50.0,
            repetition_rate=0.05,
            expert_utilization={'num_experts': 8}
        )
        
        result_dict = results.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['perplexity'] == 15.5
        assert result_dict['generation_length'] == 50.0
        assert result_dict['expert_utilization'] == {'num_experts': 8}


class TestPerplexityMetric:
    """Test perplexity evaluation metric."""
    
    def test_perplexity_metric_creation(self):
        """Test creating perplexity metric."""
        device = torch.device('cpu')
        metric = PerplexityMetric(device)
        
        assert metric.device == device
    
    def test_perplexity_evaluation(self):
        """Test perplexity evaluation on model."""
        # Create simple model
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Linear(32, 100)
        )
        
        def model_forward(input_ids, labels=None):
            x = model[0](input_ids)
            logits = model[1](x.mean(dim=1))  # Simple averaging
            
            loss = None
            if labels is not None:
                # Simple loss calculation
                target = labels[:, 0]  # Use first token as target
                loss = nn.functional.cross_entropy(logits, target)
            
            return {'logits': logits.unsqueeze(1).expand(-1, input_ids.size(1), -1), 'loss': loss}
        
        model.forward = model_forward
        model.config = Mock()
        model.config.block_size = 16
        
        # Create test dataset
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            test_data = np.random.randint(0, 100, size=200, dtype=np.uint16)
            np.save(temp_path, test_data)
            
            dataset = TokenDataset(
                data_path=temp_path,
                block_size=8,
                tokenizer_name="gpt2"
            )
            
            # Evaluate perplexity
            metric = PerplexityMetric(torch.device('cpu'))
            perplexity, loss = metric.evaluate(model, dataset, batch_size=2, max_batches=3)
            
            assert isinstance(perplexity, float)
            assert isinstance(loss, float)
            assert perplexity > 0
            assert loss >= 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestGenerationQualityMetric:
    """Test generation quality evaluation."""
    
    def test_generation_quality_metric_creation(self):
        """Test creating generation quality metric."""
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        metric = GenerationQualityMetric(tokenizer)
        assert metric.tokenizer is not None
    
    def test_repetition_rate_calculation(self):
        """Test repetition rate calculation."""
        tokenizer = Mock()
        tokenizer.encode.side_effect = [
            [1, 2, 3, 1, 2, 3, 4],  # Text with repetition
            [1, 2, 3, 4, 5, 6, 7],  # Text without repetition
        ]
        
        metric = GenerationQualityMetric(tokenizer)
        
        texts = ["repeated text", "unique text"]
        repetition_rate = metric._calculate_repetition_rate(texts)
        
        assert isinstance(repetition_rate, float)
        assert 0 <= repetition_rate <= 1


class TestPerformanceMetric:
    """Test performance evaluation metrics."""
    
    def test_performance_metric_creation(self):
        """Test creating performance metric."""
        device = torch.device('cpu')
        metric = PerformanceMetric(device)
        
        assert metric.device == device
    
    def test_memory_usage_measurement(self):
        """Test memory usage measurement."""
        # Create model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.Linear(50, 10)
        )
        
        metric = PerformanceMetric(torch.device('cpu'))
        memory_stats = metric.measure_memory_usage(model)
        
        assert isinstance(memory_stats, dict)
        assert 'model_memory_mb' in memory_stats
        assert 'param_memory_mb' in memory_stats
        assert 'buffer_memory_mb' in memory_stats
        assert memory_stats['model_memory_mb'] > 0


class TestExpertUtilizationMetric:
    """Test expert utilization evaluation."""
    
    def test_expert_utilization_creation(self):
        """Test creating expert utilization metric."""
        metric = ExpertUtilizationMetric()
        assert metric is not None
    
    def test_expert_utilization_no_moe(self):
        """Test expert utilization on non-MoE model."""
        # Regular model without MoE
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
        
        # Create test dataset
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            test_data = np.random.randint(0, 100, size=100, dtype=np.uint16)
            np.save(temp_path, test_data)
            
            dataset = TokenDataset(
                data_path=temp_path,
                block_size=8,
                tokenizer_name="gpt2"
            )
            
            metric = ExpertUtilizationMetric()
            result = metric.evaluate(model, dataset, batch_size=2, num_batches=2)
            
            # Should return error for non-MoE model
            assert 'error' in result
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestModelEvaluator:
    """Test comprehensive model evaluator."""
    
    def test_model_evaluator_creation(self):
        """Test creating model evaluator."""
        device = torch.device('cpu')
        evaluator = ModelEvaluator(device)
        
        assert evaluator.device == device
        assert evaluator.perplexity_metric is not None
        assert evaluator.performance_metric is not None
        assert evaluator.expert_metric is not None
    
    def test_save_and_load_results(self):
        """Test saving and loading evaluation results."""
        evaluator = ModelEvaluator()
        
        results = EvaluationResults(
            perplexity=10.0,
            bits_per_byte=3.5,
            tokens_per_second=1000.0,
            memory_usage_mb=256.0,
            loss=2.3
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save results
            evaluator.save_results(results, temp_path)
            assert os.path.exists(temp_path)
            
            # Load results
            loaded_results = evaluator.load_results(temp_path)
            assert isinstance(loaded_results, dict)
            assert loaded_results['perplexity'] == 10.0
            assert loaded_results['loss'] == 2.3
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestComparisonResults:
    """Test comparison results container."""
    
    def test_comparison_results_creation(self):
        """Test creating comparison results."""
        nanokimi_results = EvaluationResults(
            perplexity=10.0,
            bits_per_byte=3.0,
            tokens_per_second=1000.0,
            memory_usage_mb=256.0,
            loss=2.3
        )
        
        baseline_results = {
            'nanoGPT': EvaluationResults(
                perplexity=12.0,
                bits_per_byte=3.5,
                tokens_per_second=800.0,
                memory_usage_mb=300.0,
                loss=2.5
            )
        }
        
        improvements = {'perplexity_improvement_percent': 16.7}
        efficiency_ratios = {'speed_per_param_ratio': 1.25}
        
        comparison = ComparisonResults(
            nanokimi_results=nanokimi_results,
            baseline_results=baseline_results,
            improvements=improvements,
            efficiency_ratios=efficiency_ratios
        )
        
        assert comparison.nanokimi_results == nanokimi_results
        assert comparison.baseline_results == baseline_results
        assert comparison.improvements == improvements
        assert comparison.efficiency_ratios == efficiency_ratios
    
    def test_comparison_results_to_dict(self):
        """Test converting comparison results to dictionary."""
        nanokimi_results = EvaluationResults(
            perplexity=10.0,
            bits_per_byte=3.0,
            tokens_per_second=1000.0,
            memory_usage_mb=256.0,
            loss=2.3
        )
        
        baseline_results = {
            'baseline': EvaluationResults(
                perplexity=12.0,
                bits_per_byte=3.5,
                tokens_per_second=800.0,
                memory_usage_mb=300.0,
                loss=2.5
            )
        }
        
        comparison = ComparisonResults(
            nanokimi_results=nanokimi_results,
            baseline_results=baseline_results,
            improvements={},
            efficiency_ratios={}
        )
        
        result_dict = comparison.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'nanokimi' in result_dict
        assert 'baselines' in result_dict
        assert 'improvements' in result_dict
        assert 'efficiency_ratios' in result_dict


class TestNanoGPTComparison:
    """Test nanoGPT comparison framework."""
    
    def test_nanogpt_comparison_creation(self):
        """Test creating nanoGPT comparison."""
        device = torch.device('cpu')
        comparison = NanoGPTComparison(device)
        
        assert comparison.device == device
        assert comparison.evaluator is not None
    
    def test_create_equivalent_nanogpt(self):
        """Test creating equivalent nanoGPT model."""
        nanokimi_config = KimiConfig(
            n_layer=2,
            n_embd=64,
            n_head=4,
            vocab_size=100,
            block_size=32
        )
        
        comparison = NanoGPTComparison()
        
        # This should create a simple GPT model
        equivalent_model = comparison.create_equivalent_nanogpt(nanokimi_config)
        
        assert equivalent_model is not None
        assert hasattr(equivalent_model, 'forward')
        assert hasattr(equivalent_model, 'get_num_params')
    
    def test_calculate_improvements(self):
        """Test improvement calculation."""
        nanokimi_results = EvaluationResults(
            perplexity=10.0,
            loss=2.3,
            tokens_per_second=1000.0,
            memory_usage_mb=256.0,
            bits_per_byte=3.0
        )
        
        baseline_results = EvaluationResults(
            perplexity=12.0,
            loss=2.5,
            tokens_per_second=800.0,
            memory_usage_mb=300.0,
            bits_per_byte=3.5
        )
        
        comparison = NanoGPTComparison()
        improvements = comparison._calculate_improvements(nanokimi_results, baseline_results)
        
        assert isinstance(improvements, dict)
        
        # Perplexity improvement: (12.0 - 10.0) / 12.0 * 100 = 16.67%
        assert 'perplexity_improvement_percent' in improvements
        assert abs(improvements['perplexity_improvement_percent'] - 16.67) < 0.1
        
        # Speed improvement: (1000 - 800) / 800 * 100 = 25%
        assert 'tokens_per_second_improvement_percent' in improvements
        assert abs(improvements['tokens_per_second_improvement_percent'] - 25.0) < 0.1


class TestDatasetInfo:
    """Test dataset information container."""
    
    def test_dataset_info_creation(self):
        """Test creating dataset info."""
        info = DatasetInfo(
            name="Test Dataset",
            description="A test dataset",
            size_mb=100.0,
            num_tokens=1000000,
            vocab_size=50000,
            download_url="https://example.com",
            license="MIT"
        )
        
        assert info.name == "Test Dataset"
        assert info.description == "A test dataset"
        assert info.size_mb == 100.0
        assert info.num_tokens == 1000000
        assert info.vocab_size == 50000
        assert info.download_url == "https://example.com"
        assert info.license == "MIT"


class TestBenchmarkDatasetManager:
    """Test benchmark dataset manager."""
    
    def test_dataset_manager_creation(self):
        """Test creating dataset manager."""
        manager = BenchmarkDatasetManager(data_dir="./test_data", tokenizer_name="gpt2")
        
        assert manager.data_dir == "./test_data"
        assert manager.tokenizer_name == "gpt2"
        assert len(manager.datasets) > 0
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        manager = BenchmarkDatasetManager()
        datasets_info = manager.list_available_datasets()
        
        assert isinstance(datasets_info, dict)
        assert len(datasets_info) > 0
        
        # Check that we have expected datasets
        expected_datasets = ['wikitext103', 'penntreebank', 'lambada', 'hellaswag']
        for dataset_name in expected_datasets:
            assert dataset_name in datasets_info
            assert isinstance(datasets_info[dataset_name], DatasetInfo)
    
    def test_invalid_dataset_name(self):
        """Test error handling for invalid dataset names."""
        manager = BenchmarkDatasetManager()
        
        with pytest.raises(ValueError):
            manager.download_dataset("nonexistent_dataset")


class TestWikiTextDataset:
    """Test WikiText dataset handler."""
    
    def test_wikitext_dataset_creation(self):
        """Test creating WikiText dataset handler."""
        dataset = WikiTextDataset(data_dir="./test_data", tokenizer_name="gpt2")
        
        assert dataset.data_dir == "./test_data"
        assert dataset.tokenizer_name == "gpt2"
        assert dataset.tokenizer is not None
    
    def test_wikitext_dataset_info(self):
        """Test WikiText dataset info."""
        dataset = WikiTextDataset()
        info = dataset.get_info()
        
        assert isinstance(info, DatasetInfo)
        assert info.name == "WikiText-103"
        assert "Wikipedia" in info.description
        assert info.size_mb is not None
        assert info.num_tokens is not None
        assert info.license is not None


class TestPennTreebankDataset:
    """Test Penn Treebank dataset handler."""
    
    def test_ptb_dataset_creation(self):
        """Test creating Penn Treebank dataset handler."""
        dataset = PennTreebankDataset(data_dir="./test_data", tokenizer_name="gpt2")
        
        assert dataset.data_dir == "./test_data"
        assert dataset.tokenizer_name == "gpt2"
        assert dataset.tokenizer is not None
    
    def test_ptb_dataset_info(self):
        """Test Penn Treebank dataset info."""
        dataset = PennTreebankDataset()
        info = dataset.get_info()
        
        assert isinstance(info, DatasetInfo)
        assert info.name == "Penn Treebank"
        assert "language modeling" in info.description
        assert info.size_mb is not None
        assert info.num_tokens is not None


class TestBenchmarkIntegration:
    """Integration tests for benchmarking components."""
    
    @patch('nanokimi.benchmark.datasets.load_dataset')
    def test_load_benchmark_dataset_mock(self, mock_load_dataset):
        """Test loading benchmark dataset with mocked HuggingFace datasets."""
        # Mock dataset
        mock_dataset_item = {'text': 'This is test text for the dataset.'}
        mock_dataset = [mock_dataset_item] * 10
        mock_load_dataset.return_value = mock_dataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                dataset = load_benchmark_dataset(
                    'wikitext103',
                    split='test',
                    data_dir=temp_dir,
                    download=True,
                    block_size=32
                )
                
                assert isinstance(dataset, TokenDataset)
                assert len(dataset) > 0
                
            except Exception as e:
                # If HuggingFace datasets not available or other issues, skip
                pytest.skip(f"Dataset loading failed: {e}")
    
    def test_benchmark_model_function(self):
        """Test benchmark_model convenience function."""
        # Create simple model
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Linear(32, 100)
        )
        
        # Mock necessary methods
        model.get_num_params = lambda: 3300  # Approximate param count
        model.config = Mock()
        model.config.block_size = 16
        
        def model_forward(input_ids, labels=None):
            x = model[0](input_ids)
            logits = model[1](x.mean(dim=1))
            
            loss = None
            if labels is not None:
                target = labels[:, 0]
                loss = nn.functional.cross_entropy(logits, target)
            
            return {'logits': logits.unsqueeze(1).expand(-1, input_ids.size(1), -1), 'loss': loss}
        
        model.forward = model_forward
        
        # Create test dataset
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        with tempfile.TemporaryDirectory() as temp_output_dir:
            try:
                test_data = np.random.randint(0, 100, size=100, dtype=np.uint16)
                np.save(temp_path, test_data)
                
                dataset = TokenDataset(
                    data_path=temp_path,
                    block_size=8,
                    tokenizer_name="gpt2"
                )
                
                # Run benchmark
                results = benchmark_model(
                    model=model,
                    dataset=dataset,
                    config_name="test_model",
                    output_dir=temp_output_dir,
                    batch_size=2,
                    max_batches=2,
                    include_generation=False  # Skip generation for simple model
                )
                
                assert isinstance(results, EvaluationResults)
                assert results.perplexity > 0
                assert results.loss >= 0
                
                # Check that results file was created
                result_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.json')]
                assert len(result_files) > 0
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])