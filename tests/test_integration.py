"""
Integration tests for nanoKimi.

These tests verify that the complete system works end-to-end,
including training, inference, and benchmarking.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, Mock

from nanokimi.model import KimiModel, KimiConfig
from nanokimi.training import Trainer, TrainingConfig, create_datasets, DataConfig
from nanokimi.training.optimizer import MuonOptimizer
from nanokimi.inference import Generator, GenerationConfig
from nanokimi.benchmark import ModelEvaluator, benchmark_model
from nanokimi.training.data import TokenDataset


class TestEndToEndTraining:
    """Test complete training pipeline."""
    
    @pytest.mark.slow
    def test_mini_training_pipeline(self):
        """Test minimal end-to-end training pipeline."""
        # Create minimal model config
        model_config = KimiConfig(
            n_layer=2,
            n_embd=32,
            n_head=2,
            vocab_size=100,
            block_size=16,
            dropout=0.0,  # Disable for deterministic testing
            moe={
                "use_moe": True,
                "num_experts": 2,
                "top_k": 1,
                "expert_capacity_factor": 1.0,
                "load_balance_loss_coeff": 0.01,
                "moe_layers": [1],  # Only use MoE in layer 1
            }
        )
        
        # Create training config
        training_config = TrainingConfig(
            batch_size=2,
            micro_batch_size=1,
            max_steps=5,  # Very short training
            eval_interval=3,
            log_interval=2,
            learning_rate=1e-3,
            warmup_steps=1,
            use_wandb=False,
            mixed_precision=False,
            compile_model=False,
            dataset_name="toy",
            data_dir="./test_data",
            checkpoint_dir="./test_checkpoints",
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.data_dir = temp_dir
            training_config.checkpoint_dir = temp_dir
            
            # Create toy datasets
            data_config = DataConfig(
                tokenizer_name="gpt2",
                block_size=model_config.block_size,
                data_dir=temp_dir,
            )
            
            train_dataset, val_dataset = create_datasets(data_config, "toy")
            
            # Create trainer
            trainer = Trainer(
                model=KimiModel(model_config),
                config=training_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            
            # Record initial loss
            initial_params = {name: param.clone() for name, param in trainer.model.named_parameters()}
            
            # Run training
            trainer.train()
            
            # Check that parameters changed
            params_changed = False
            for name, param in trainer.model.named_parameters():
                if not torch.allclose(param, initial_params[name], atol=1e-6):
                    params_changed = True
                    break
            
            assert params_changed, "Model parameters should change during training"
            
            # Check that checkpoints were created
            checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
            assert len(checkpoint_files) > 0, "Checkpoints should be created"
    
    def test_training_with_different_configs(self):
        """Test training with different model configurations."""
        configs = [
            # Standard transformer (no MoE)
            KimiConfig(
                n_layer=1,
                n_embd=16,
                n_head=2,
                vocab_size=50,
                block_size=8,
                moe={"use_moe": False}
            ),
            # With MoE
            KimiConfig(
                n_layer=2,
                n_embd=16,
                n_head=2,
                vocab_size=50,
                block_size=8,
                moe={
                    "use_moe": True,
                    "num_experts": 2,
                    "top_k": 1,
                    "moe_layers": [1]
                }
            ),
        ]
        
        for i, config in enumerate(configs):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create minimal training config
                training_config = TrainingConfig(
                    batch_size=1,
                    max_steps=2,
                    eval_interval=1,
                    use_wandb=False,
                    mixed_precision=False,
                    data_dir=temp_dir,
                    checkpoint_dir=temp_dir,
                )
                
                # Create toy datasets
                data_config = DataConfig(
                    tokenizer_name="gpt2",
                    block_size=config.block_size,
                    data_dir=temp_dir,
                )
                
                train_dataset, val_dataset = create_datasets(data_config, "toy")
                
                # Create and run trainer
                trainer = Trainer(
                    model=KimiModel(config),
                    config=training_config,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                )
                
                # Should complete without errors
                trainer.train()


class TestEndToEndInference:
    """Test complete inference pipeline."""
    
    def test_generation_pipeline(self):
        """Test complete text generation pipeline."""
        # Create model
        config = KimiConfig(
            n_layer=2,
            n_embd=32,
            n_head=2,
            vocab_size=100,
            block_size=16
        )
        
        model = KimiModel(config)
        model.eval()
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.vocab_size = 100
        tokenizer.eos_token_id = 99
        tokenizer.pad_token_id = 98
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "generated text output"
        
        # Create generator
        generator = Generator(model, tokenizer)
        
        # Test different generation methods
        gen_config = GenerationConfig(
            max_new_tokens=5,
            temperature=1.0,
            do_sample=True
        )
        
        with torch.no_grad():
            # Test text generation
            generated_text = generator.generate_text("test prompt", gen_config)
            assert isinstance(generated_text, str)
            
            # Test batch generation
            prompts = ["prompt 1", "prompt 2"]
            tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2], [3, 4]]),
                'attention_mask': torch.tensor([[1, 1], [1, 1]])
            }
            tokenizer.decode.side_effect = ["output 1", "output 2"]
            
            batch_outputs = generator.batch_generate_text(prompts, gen_config)
            assert len(batch_outputs) == 2
            assert all(isinstance(output, str) for output in batch_outputs)
    
    def test_inference_with_caching(self):
        """Test inference with KV caching."""
        config = KimiConfig(
            n_layer=1,
            n_embd=16,
            n_head=2,
            vocab_size=50,
            block_size=8
        )
        
        model = KimiModel(config)
        model.eval()
        
        # Test that caching works
        input_ids = torch.randint(0, 50, (1, 3))
        
        with torch.no_grad():
            # First forward pass
            outputs1 = model(input_ids, use_cache=True)
            assert 'past_key_values' in outputs1
            assert outputs1['past_key_values'] is not None
            
            # Second forward pass with cache
            new_token = torch.randint(0, 50, (1, 1))
            outputs2 = model(
                new_token,
                past_key_values=outputs1['past_key_values'],
                use_cache=True
            )
            
            assert 'logits' in outputs2
            assert outputs2['logits'].shape == (1, 1, 50)


class TestEndToEndBenchmarking:
    """Test complete benchmarking pipeline."""
    
    def test_model_evaluation_pipeline(self):
        """Test complete model evaluation pipeline."""
        # Create model
        config = KimiConfig(
            n_layer=1,
            n_embd=16,
            n_head=2,
            vocab_size=50,
            block_size=8
        )
        
        model = KimiModel(config)
        
        # Create test dataset
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create test data
            test_data = np.random.randint(0, 50, size=100, dtype=np.uint16)
            np.save(temp_path, test_data)
            
            dataset = TokenDataset(
                data_path=temp_path,
                block_size=config.block_size,
                tokenizer_name="gpt2"
            )
            
            # Mock tokenizer
            tokenizer = Mock()
            tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            tokenizer.decode.return_value = "test generated text"
            
            # Evaluate model
            evaluator = ModelEvaluator()
            results = evaluator.evaluate_model(
                model=model,
                dataset=dataset,
                tokenizer=tokenizer,
                batch_size=2,
                max_batches=3,
                include_generation=True,
                generation_prompts=["test prompt"]
            )
            
            # Check results
            assert results.perplexity > 0
            assert results.loss >= 0
            assert results.tokens_per_second > 0
            assert results.memory_usage_mb > 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_benchmark_model_function(self):
        """Test benchmark_model convenience function."""
        # Create simple model
        config = KimiConfig(
            n_layer=1,
            n_embd=16,
            n_head=2,
            vocab_size=50,
            block_size=8
        )
        
        model = KimiModel(config)
        
        # Create test dataset
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        with tempfile.TemporaryDirectory() as temp_output_dir:
            try:
                test_data = np.random.randint(0, 50, size=80, dtype=np.uint16)
                np.save(temp_path, test_data)
                
                dataset = TokenDataset(
                    data_path=temp_path,
                    block_size=config.block_size,
                    tokenizer_name="gpt2"
                )
                
                # Run benchmark
                results = benchmark_model(
                    model=model,
                    dataset=dataset,
                    config_name="test_config",
                    output_dir=temp_output_dir,
                    batch_size=2,
                    max_batches=2,
                    include_generation=False  # Skip generation for speed
                )
                
                assert results.perplexity > 0
                assert results.tokens_per_second > 0
                
                # Check that results file was created
                result_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.json')]
                assert len(result_files) > 0
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


class TestModelCheckpointing:
    """Test model saving and loading."""
    
    def test_checkpoint_save_load(self):
        """Test saving and loading model checkpoints."""
        # Create model
        config = KimiConfig(
            n_layer=1,
            n_embd=16,
            n_head=2,
            vocab_size=50,
            block_size=8
        )
        
        model1 = KimiModel(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
            
            # Create checkpoint
            checkpoint = {
                'model_state_dict': model1.state_dict(),
                'config': config.to_dict(),
                'step': 100,
                'loss': 2.5,
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create new model and load state
            config2 = KimiConfig.from_dict(loaded_checkpoint['config'])
            model2 = KimiModel(config2)
            model2.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            # Check that models are equivalent
            for name, param1 in model1.named_parameters():
                param2 = dict(model2.named_parameters())[name]
                assert torch.allclose(param1, param2), f"Parameter {name} differs"
    
    def test_trainer_checkpoint_resume(self):
        """Test trainer checkpoint saving and resuming."""
        # Create minimal configs
        model_config = KimiConfig(
            n_layer=1,
            n_embd=16,
            n_head=2,
            vocab_size=50,
            block_size=8
        )
        
        training_config = TrainingConfig(
            batch_size=1,
            max_steps=3,
            eval_interval=2,
            save_interval=2,
            use_wandb=False,
            mixed_precision=False,
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.data_dir = temp_dir
            training_config.checkpoint_dir = temp_dir
            
            # Create toy datasets
            data_config = DataConfig(
                tokenizer_name="gpt2",
                block_size=model_config.block_size,
                data_dir=temp_dir,
            )
            
            train_dataset, val_dataset = create_datasets(data_config, "toy")
            
            # Create first trainer and train
            trainer1 = Trainer(
                model=KimiModel(model_config),
                config=training_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            
            # Train for a few steps
            trainer1.train()
            
            # Check that checkpoint was created
            checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith('checkpoint_step_')]
            assert len(checkpoint_files) > 0
            
            # Load checkpoint in new trainer
            checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
            
            trainer2 = Trainer(
                model=KimiModel(model_config),
                config=training_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            
            trainer2.load_checkpoint(checkpoint_path)
            
            # Check that state was loaded
            assert trainer2.step > 0


class TestOptimizerIntegration:
    """Test optimizer integration with models."""
    
    def test_muon_optimizer_with_moe(self):
        """Test Muon optimizer with MoE model."""
        config = KimiConfig(
            n_layer=2,
            n_embd=16,
            n_head=2,
            vocab_size=50,
            block_size=8,
            moe={
                "use_moe": True,
                "num_experts": 2,
                "top_k": 1,
                "moe_layers": [1]
            }
        )
        
        model = KimiModel(config)
        
        # Create optimizer with different learning rates for experts
        regular_params = []
        expert_params = []
        
        for name, param in model.named_parameters():
            if 'expert' in name.lower() or 'moe' in name.lower():
                expert_params.append(param)
            else:
                regular_params.append(param)
        
        param_groups = [
            {'params': regular_params, 'lr': 1e-3},
            {'params': expert_params, 'lr': 1e-3, 'expert_lr_multiplier': 2.0}
        ]
        
        optimizer = MuonOptimizer(param_groups)
        
        # Test training step
        input_ids = torch.randint(0, 50, (1, 8))
        labels = torch.randint(0, 50, (1, 8))
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert True


class TestConfigurationSystem:
    """Test configuration system integration."""
    
    def test_config_yaml_integration(self):
        """Test that configs work with YAML files."""
        import yaml
        
        # Create config dict
        config_dict = {
            'model': {
                'n_layer': 2,
                'n_head': 2,
                'n_embd': 16,
                'vocab_size': 50,
                'block_size': 8,
                'moe': {
                    'use_moe': True,
                    'num_experts': 2,
                    'top_k': 1
                }
            },
            'training': {
                'batch_size': 2,
                'max_steps': 5,
                'learning_rate': 1e-3
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            yaml_path = f.name
        
        try:
            # Load config from YAML
            with open(yaml_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Create model config
            model_config = KimiConfig(**loaded_config['model'])
            
            # Should create model successfully
            model = KimiModel(model_config)
            assert model is not None
            
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)


@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance-related integration."""
    
    def test_memory_usage_during_training(self):
        """Test memory usage tracking during training."""
        config = KimiConfig(
            n_layer=1,
            n_embd=32,
            n_head=2,
            vocab_size=100,
            block_size=16
        )
        
        model = KimiModel(config)
        optimizer = MuonOptimizer(model.parameters(), lr=1e-3)
        
        # Track memory before and after training step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # Training step
        input_ids = torch.randint(0, 100, (4, 16))
        labels = torch.randint(0, 100, (4, 16))
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            assert peak_memory >= initial_memory
        
        # Memory should be reasonable (not growing unboundedly)
        assert True  # Placeholder - actual memory checks would be more complex
    
    def test_gradient_flow_through_moe(self):
        """Test that gradients flow properly through MoE layers."""
        config = KimiConfig(
            n_layer=2,
            n_embd=16,
            n_head=2,
            vocab_size=50,
            block_size=8,
            moe={
                "use_moe": True,
                "num_experts": 2,
                "top_k": 1,
                "moe_layers": [1]
            }
        )
        
        model = KimiModel(config)
        
        # Forward pass
        input_ids = torch.randint(0, 50, (2, 8))
        labels = torch.randint(0, 50, (2, 8))
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Check that auxiliary loss is present
        assert 'auxiliary_loss' in outputs
        if outputs['auxiliary_loss'] is not None:
            total_loss = loss + outputs['auxiliary_loss']
        else:
            total_loss = loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that all parameters have gradients
        params_with_grad = 0
        params_without_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    # Check for NaN gradients
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                else:
                    params_without_grad += 1
        
        # Most parameters should have gradients
        assert params_with_grad > params_without_grad


if __name__ == "__main__":
    pytest.main([__file__])