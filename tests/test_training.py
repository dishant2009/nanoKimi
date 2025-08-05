"""
Unit tests for nanoKimi training components.

Tests the training loop, optimizer, data loading, and utilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from nanokimi.training.optimizer import MuonOptimizer, AdafactorMuon
from nanokimi.training.data import TokenDataset, TextDataProcessor, DataConfig, create_datasets
from nanokimi.training.utils import (
    WarmupCosineScheduler, WarmupLinearScheduler, WarmupConstantScheduler,
    get_lr_scheduler, compute_loss, compute_perplexity, count_parameters,
    clip_grad_norm, estimate_mfu
)
from nanokimi.training.trainer import TrainingConfig, Trainer
from nanokimi.model import KimiModel, KimiConfig


class TestMuonOptimizer:
    """Test Muon optimizer."""
    
    def test_optimizer_creation(self):
        """Test optimizer creation with different parameters."""
        # Create simple model
        model = nn.Linear(10, 5)
        
        # Test basic creation
        optimizer = MuonOptimizer(model.parameters())
        assert optimizer is not None
        
        # Test with custom parameters
        optimizer = MuonOptimizer(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            expert_lr_multiplier=2.0
        )
        assert optimizer.defaults['lr'] == 1e-3
        assert optimizer.defaults['expert_lr_multiplier'] == 2.0
    
    def test_optimizer_step(self):
        """Test optimizer step functionality."""
        # Create model and data
        model = nn.Linear(10, 1)
        optimizer = MuonOptimizer(model.parameters(), lr=1e-2)
        
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        # Initial parameters
        initial_weight = model.weight.clone()
        
        # Forward and backward pass
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        assert not torch.allclose(model.weight, initial_weight)
    
    def test_parameter_groups(self):
        """Test optimizer with different parameter groups."""
        # Create model with different parameter types
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 1)
        )
        
        # Separate parameters into groups
        regular_params = [model[0].weight, model[0].bias]
        expert_params = [model[1].weight, model[1].bias]  # Simulate expert params
        
        param_groups = [
            {'params': regular_params, 'lr': 1e-3},
            {'params': expert_params, 'lr': 1e-3, 'expert_lr_multiplier': 2.0}
        ]
        
        optimizer = MuonOptimizer(param_groups)
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['expert_lr_multiplier'] == 1.0  # Default
        assert optimizer.param_groups[1]['expert_lr_multiplier'] == 2.0
    
    def test_invalid_parameters(self):
        """Test optimizer with invalid parameters."""
        model = nn.Linear(10, 1)
        
        # Invalid learning rate
        with pytest.raises(ValueError):
            MuonOptimizer(model.parameters(), lr=-1.0)
        
        # Invalid beta parameters
        with pytest.raises(ValueError):
            MuonOptimizer(model.parameters(), betas=(1.5, 0.9))
        
        # Invalid weight decay
        with pytest.raises(ValueError):
            MuonOptimizer(model.parameters(), weight_decay=-0.1)


class TestAdafactorMuon:
    """Test AdafactorMuon optimizer."""
    
    def test_adafactor_creation(self):
        """Test AdafactorMuon optimizer creation."""
        model = nn.Linear(20, 10)
        
        optimizer = AdafactorMuon(model.parameters())
        assert optimizer is not None
        
        # Test with custom parameters
        optimizer = AdafactorMuon(
            model.parameters(),
            lr=1e-3,
            expert_lr_multiplier=1.5,
            scale_parameter=True,
            relative_step=True
        )
        assert optimizer.defaults['expert_lr_multiplier'] == 1.5
    
    def test_adafactor_step(self):
        """Test AdafactorMuon step."""
        model = nn.Linear(20, 5)
        optimizer = AdafactorMuon(model.parameters(), lr=1e-2)
        
        x = torch.randn(10, 20)
        y = torch.randn(10, 5)
        
        initial_weight = model.weight.clone()
        
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Parameters should change
        assert not torch.allclose(model.weight, initial_weight)


class TestLearningRateSchedulers:
    """Test learning rate schedulers."""
    
    def test_warmup_cosine_scheduler(self):
        """Test warmup cosine scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            max_steps=100,
            min_lr=0.1
        )
        
        # Test warmup phase
        initial_lr = scheduler.get_lr()[0]
        assert initial_lr == 0.1  # lr * (1/10) for first step
        
        # Step through warmup
        for _ in range(10):
            scheduler.step()
        
        # After warmup, should be at full lr
        warmup_end_lr = scheduler.get_lr()[0]
        assert abs(warmup_end_lr - 1.0) < 1e-6
        
        # Test cosine decay
        for _ in range(50):
            scheduler.step()
        
        decay_lr = scheduler.get_lr()[0]
        assert 0.1 < decay_lr < 1.0  # Should be between min_lr and initial_lr
    
    def test_warmup_linear_scheduler(self):
        """Test warmup linear scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=5,
            max_steps=50,
            min_lr=0.0
        )
        
        # Test warmup
        for step in range(6):
            lr = scheduler.get_lr()[0]
            expected_lr = 1.0 * step / 5 if step <= 5 else 1.0
            if step <= 5:
                assert abs(lr - expected_lr) < 1e-6
            scheduler.step()
    
    def test_warmup_constant_scheduler(self):
        """Test warmup constant scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        
        scheduler = WarmupConstantScheduler(optimizer, warmup_steps=5)
        
        # During warmup
        for step in range(5):
            lr = scheduler.get_lr()[0]
            expected_lr = 1.0 * (step + 1) / 5
            assert abs(lr - expected_lr) < 1e-6
            scheduler.step()
        
        # After warmup (constant)
        for _ in range(10):
            lr = scheduler.get_lr()[0]
            assert abs(lr - 1.0) < 1e-6
            scheduler.step()
    
    def test_get_lr_scheduler_factory(self):
        """Test learning rate scheduler factory function."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        
        # Test different scheduler types
        cosine_scheduler = get_lr_scheduler(optimizer, "cosine", 10, 100)
        assert isinstance(cosine_scheduler, WarmupCosineScheduler)
        
        linear_scheduler = get_lr_scheduler(optimizer, "linear", 10, 100)
        assert isinstance(linear_scheduler, WarmupLinearScheduler)
        
        constant_scheduler = get_lr_scheduler(optimizer, "constant", 10, 100)
        assert isinstance(constant_scheduler, WarmupConstantScheduler)
        
        # Test invalid scheduler type
        with pytest.raises(ValueError):
            get_lr_scheduler(optimizer, "invalid", 10, 100)


class TestDataComponents:
    """Test data loading and processing components."""
    
    def test_token_dataset_creation(self):
        """Test TokenDataset creation and basic functionality."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            
        try:
            # Create test data
            test_data = np.random.randint(0, 1000, size=1000, dtype=np.uint16)
            np.save(temp_path, test_data)
            
            # Create dataset
            dataset = TokenDataset(
                data_path=temp_path,
                block_size=64,
                split="test",
                tokenizer_name="gpt2"
            )
            
            assert len(dataset) > 0
            
            # Test getting an item
            item = dataset[0]
            assert 'input_ids' in item
            assert 'labels' in item
            assert item['input_ids'].shape == (64,)
            assert item['labels'].shape == (64,)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_text_data_processor(self):
        """Test text data processing."""
        config = DataConfig(tokenizer_name="gpt2", block_size=128)
        processor = TextDataProcessor(config)
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document. " * 100)
            temp_text_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_output_path = f.name
        
        try:
            # Process text file
            metadata = processor.process_text_file(temp_text_path, temp_output_path)
            
            assert 'num_tokens' in metadata
            assert 'num_chars' in metadata
            assert 'compression_ratio' in metadata
            assert metadata['num_tokens'] > 0
            
            # Check that output file was created
            assert os.path.exists(temp_output_path)
            
            # Load and verify processed data
            processed_data = np.load(temp_output_path)
            assert len(processed_data) == metadata['num_tokens']
            
        finally:
            # Clean up
            for path in [temp_text_path, temp_output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_create_datasets(self):
        """Test dataset creation factory function."""
        config = DataConfig(
            tokenizer_name="gpt2",
            block_size=64,
            data_dir="./test_data"
        )
        
        # Test toy dataset creation
        train_dataset, val_dataset = create_datasets(config, "toy")
        
        assert isinstance(train_dataset, TokenDataset)
        assert isinstance(val_dataset, TokenDataset)
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        
        # Test dataset items
        train_item = train_dataset[0]
        val_item = val_dataset[0]
        
        for item in [train_item, val_item]:
            assert 'input_ids' in item
            assert 'labels' in item
            assert item['input_ids'].shape == (config.block_size,)


class TestTrainingUtilities:
    """Test training utility functions."""
    
    def test_compute_loss(self):
        """Test loss computation functions."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        # Create test data
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test basic loss computation
        loss = compute_loss(logits, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Scalar loss
        assert loss >= 0
        
        # Test with label smoothing
        loss_smooth = compute_loss(logits, labels, label_smoothing=0.1)
        assert isinstance(loss_smooth, torch.Tensor)
        assert loss_smooth >= 0
        
        # Test with ignore index
        labels_with_ignore = labels.clone()
        labels_with_ignore[:, -2:] = -100  # Ignore last 2 tokens
        
        loss_ignore = compute_loss(logits, labels_with_ignore, ignore_index=-100)
        assert isinstance(loss_ignore, torch.Tensor)
        assert loss_ignore >= 0
    
    def test_compute_perplexity(self):
        """Test perplexity computation."""
        # Test with known loss values
        loss_tensor = torch.tensor(2.0)
        perplexity = compute_perplexity(loss_tensor)
        
        expected_ppl = torch.exp(loss_tensor)
        assert torch.allclose(perplexity, expected_ppl)
        
        # Test with batch of losses
        losses = torch.tensor([1.0, 2.0, 3.0])
        perplexities = compute_perplexity(losses)
        expected = torch.exp(losses)
        assert torch.allclose(perplexities, expected)
    
    def test_count_parameters(self):
        """Test parameter counting utility."""
        # Create model with known parameter count
        model = nn.Sequential(
            nn.Linear(10, 5),  # 10*5 + 5 = 55 params
            nn.Linear(5, 1),   # 5*1 + 1 = 6 params
        )
        
        # Total parameters
        total_params = count_parameters(model, trainable_only=False)
        assert total_params == 61  # 55 + 6
        
        # Trainable parameters (all should be trainable)
        trainable_params = count_parameters(model, trainable_only=True)
        assert trainable_params == 61
        
        # Test with non-trainable parameters
        model[0].weight.requires_grad = False
        trainable_params = count_parameters(model, trainable_only=True)
        assert trainable_params == 11  # 61 - 50 (10*5 weight)
    
    def test_clip_grad_norm(self):
        """Test gradient clipping utility."""
        # Create model and compute gradients
        model = nn.Linear(10, 1)
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        
        # Test gradient clipping
        total_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
        
        assert isinstance(total_norm, torch.Tensor)
        assert total_norm >= 0
        
        # Check that gradients are clipped if necessary
        current_norm = torch.norm(torch.stack([
            p.grad.norm() for p in model.parameters() if p.grad is not None
        ]))
        assert current_norm <= 1.0 + 1e-6  # Allow small numerical error
    
    def test_estimate_mfu(self):
        """Test MFU estimation utility."""
        # Create simple model
        model = nn.Linear(100, 50)
        
        # Estimate MFU
        mfu = estimate_mfu(
            model=model,
            batch_size=4,
            seq_len=10,
            dt=0.1  # 100ms per step
        )
        
        assert isinstance(mfu, float)
        assert 0 <= mfu <= 100  # MFU should be between 0 and 100%


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_training_config_creation(self):
        """Test TrainingConfig creation."""
        config = TrainingConfig()
        
        # Check default values
        assert config.batch_size == 32
        assert config.learning_rate == 3e-4
        assert config.max_steps == 100000
        assert config.dataset_name == "openwebtext"
        
        # Test custom values
        custom_config = TrainingConfig(
            batch_size=64,
            learning_rate=1e-3,
            max_steps=50000
        )
        
        assert custom_config.batch_size == 64
        assert custom_config.learning_rate == 1e-3
        assert custom_config.max_steps == 50000


class TestTrainerIntegration:
    """Integration tests for trainer components."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        # Create model and config
        model_config = KimiConfig(
            n_layer=2,
            n_embd=64,
            n_head=4,
            vocab_size=100,
            block_size=32
        )
        
        training_config = TrainingConfig(
            batch_size=4,
            max_steps=10,
            eval_interval=5,
            use_wandb=False,
            mixed_precision=False,
        )
        
        model = KimiModel(model_config)
        
        # Create simple dataset
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            
        try:
            # Create test data
            test_data = np.random.randint(0, 100, size=500, dtype=np.uint16)
            np.save(temp_path, test_data)
            
            dataset = TokenDataset(
                data_path=temp_path,
                block_size=model_config.block_size,
                tokenizer_name="gpt2"
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                config=training_config,
                train_dataset=dataset,
                val_dataset=dataset
            )
            
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.scheduler is not None
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('wandb.init')
    @patch('wandb.log')
    @patch('wandb.finish')
    def test_trainer_logging_setup(self, mock_wandb_finish, mock_wandb_log, mock_wandb_init):
        """Test trainer logging setup."""
        model_config = KimiConfig(
            n_layer=1,
            n_embd=32,
            n_head=2,
            vocab_size=50,
            block_size=16
        )
        
        training_config = TrainingConfig(
            batch_size=2,
            max_steps=5,
            use_wandb=True,
            wandb_project="test_project"
        )
        
        model = KimiModel(model_config)
        
        trainer = Trainer(
            model=model,
            config=training_config,
            train_dataset=None,
            val_dataset=None
        )
        
        # Check that wandb.init was called
        mock_wandb_init.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])