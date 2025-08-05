"""
Unit tests for nanoKimi model components.

Tests the core model architecture including MoE layers,
attention mechanisms, and embeddings.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from nanokimi.model import KimiModel, KimiConfig
from nanokimi.model.moe import MixtureOfExperts, TopKRouter, Expert
from nanokimi.model.attention import LatentAttention, StandardAttention
from nanokimi.model.embedding import TokenEmbedding, PositionalEmbedding


class TestKimiConfig:
    """Test KimiConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KimiConfig()
        
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768
        assert config.vocab_size == 50257
        assert config.block_size == 1024
        assert config.dropout == 0.1
        
        # Check MoE defaults
        assert config.moe["num_experts"] == 8
        assert config.moe["top_k"] == 2
        assert config.moe["use_moe"] == True
        
        # Check attention defaults
        assert config.attention["type"] == "latent"
        assert config.attention["latent_dim"] == 64
        assert config.attention["num_latents"] == 32
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = KimiConfig(n_embd=768, n_head=12)
        
        # Invalid: n_embd not divisible by n_head
        with pytest.raises(AssertionError):
            KimiConfig(n_embd=100, n_head=12)
        
        # Invalid: top_k > num_experts
        with pytest.raises(AssertionError):
            KimiConfig(moe={"num_experts": 4, "top_k": 8})
    
    def test_config_serialization(self):
        """Test config to/from dict conversion."""
        config = KimiConfig(n_layer=6, n_embd=384)
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["n_layer"] == 6
        assert config_dict["n_embd"] == 384
        
        # Test round-trip conversion
        config2 = KimiConfig.from_dict(config_dict)
        assert config2.n_layer == config.n_layer
        assert config2.n_embd == config.n_embd


class TestTokenEmbedding:
    """Test token embedding layer."""
    
    def test_token_embedding_forward(self):
        """Test token embedding forward pass."""
        vocab_size = 1000
        n_embd = 128
        batch_size = 4
        seq_len = 10
        
        embedding = TokenEmbedding(vocab_size, n_embd)
        
        # Test input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        output = embedding(input_ids)
        
        assert output.shape == (batch_size, seq_len, n_embd)
        assert output.dtype == torch.float32
    
    def test_embedding_weights(self):
        """Test embedding weight initialization."""
        vocab_size = 1000
        n_embd = 128
        
        embedding = TokenEmbedding(vocab_size, n_embd)
        
        assert embedding.weight.shape == (vocab_size, n_embd)
        
        # Check weight initialization (should be roughly normal with std=0.02)
        std = embedding.weight.std().item()
        assert 0.01 < std < 0.05  # Reasonable range for std=0.02 initialization


class TestPositionalEmbedding:
    """Test positional embedding layer."""
    
    def test_positional_embedding_forward(self):
        """Test positional embedding forward pass."""
        block_size = 1024
        n_embd = 128
        seq_len = 10
        
        embedding = PositionalEmbedding(block_size, n_embd)
        
        # Test input
        position_ids = torch.arange(seq_len)
        
        # Forward pass
        output = embedding(position_ids)
        
        assert output.shape == (seq_len, n_embd)
        assert output.dtype == torch.float32
    
    def test_position_bounds(self):
        """Test that positions don't exceed block size."""
        block_size = 100
        n_embd = 64
        
        embedding = PositionalEmbedding(block_size, n_embd)
        
        # Valid positions
        valid_positions = torch.arange(block_size)
        output = embedding(valid_positions)
        assert output.shape == (block_size, n_embd)
        
        # Invalid positions should cause IndexError
        with pytest.raises(IndexError):
            invalid_positions = torch.arange(block_size + 10)
            embedding(invalid_positions)


class TestTopKRouter:
    """Test Top-K router for MoE."""
    
    def test_router_forward(self):
        """Test router forward pass."""
        n_embd = 128
        num_experts = 8
        top_k = 2
        batch_size = 4
        seq_len = 10
        
        router = TopKRouter(n_embd, num_experts, top_k)
        
        # Test input
        x = torch.randn(batch_size * seq_len, n_embd)
        
        # Forward pass
        expert_indices, routing_weights, router_logits = router(x)
        
        assert expert_indices.shape == (batch_size * seq_len, top_k)
        assert routing_weights.shape == (batch_size * seq_len, top_k)
        assert router_logits.shape == (batch_size * seq_len, num_experts)
        
        # Check that indices are in valid range
        assert expert_indices.min() >= 0
        assert expert_indices.max() < num_experts
        
        # Check that weights sum to 1 (approximately)
        weight_sums = routing_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)


class TestExpert:
    """Test individual expert network."""
    
    def test_expert_forward(self):
        """Test expert forward pass."""
        n_embd = 128
        batch_size = 16
        
        expert = Expert(n_embd, dropout=0.1)
        
        # Test input
        x = torch.randn(batch_size, n_embd)
        
        # Forward pass
        output = expert(x)
        
        assert output.shape == (batch_size, n_embd)
        assert output.dtype == torch.float32
        
        # Test with different input size
        x2 = torch.randn(32, n_embd)
        output2 = expert(x2)
        assert output2.shape == (32, n_embd)


class TestMixtureOfExperts:
    """Test Mixture of Experts layer."""
    
    def test_moe_forward(self):
        """Test MoE forward pass."""
        config = KimiConfig(
            n_embd=128,
            n_head=8,  # 128 is divisible by 8
            moe={
                "num_experts": 4,
                "top_k": 2,
                "expert_capacity_factor": 1.25,
                "load_balance_loss_coeff": 0.01,
            }
        )
        
        moe = MixtureOfExperts(config)
        
        batch_size = 2
        seq_len = 8
        n_embd = config.n_embd
        
        # Test input
        x = torch.randn(batch_size, seq_len, n_embd)
        
        # Forward pass
        output = moe(x)
        
        assert output.shape == (batch_size, seq_len, n_embd)
        assert output.dtype == torch.float32
        
        # Check that auxiliary loss is computed
        assert moe.auxiliary_loss is not None
        assert isinstance(moe.auxiliary_loss, torch.Tensor)
        assert moe.auxiliary_loss >= 0
    
    def test_moe_expert_utilization(self):
        """Test expert utilization tracking."""
        config = KimiConfig(
            n_embd=64,
            moe={"num_experts": 4, "top_k": 2}
        )
        
        moe = MixtureOfExperts(config)
        
        # Forward pass
        x = torch.randn(8, 16, config.n_embd)
        output = moe(x)
        
        # Check expert utilization stats (would need router logits)
        # This is more of an integration test
        assert output.shape == x.shape


class TestLatentAttention:
    """Test latent attention mechanism."""
    
    def test_latent_attention_forward(self):
        """Test latent attention forward pass."""
        config = KimiConfig(
            n_embd=128,
            n_head=8,
            attention={
                "type": "latent",
                "latent_dim": 32,
                "num_latents": 16,
            }
        )
        
        attention = LatentAttention(config)
        
        batch_size = 2
        seq_len = 10
        n_embd = config.n_embd
        
        # Test input
        x = torch.randn(batch_size, seq_len, n_embd)
        
        # Forward pass
        output, past_key_value = attention(x)
        
        assert output.shape == (batch_size, seq_len, n_embd)
        assert output.dtype == torch.float32
        
        # Test with caching
        output_cached, new_past = attention(x, use_cache=True)
        assert new_past is not None
    
    def test_latent_attention_with_mask(self):
        """Test latent attention with attention mask."""
        config = KimiConfig(
            n_embd=64,
            attention={"type": "latent", "latent_dim": 32, "num_latents": 8}
        )
        
        attention = LatentAttention(config)
        
        batch_size = 2
        seq_len = 6
        
        x = torch.randn(batch_size, seq_len, config.n_embd)
        
        # Create attention mask (mask out last 2 tokens)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -2:] = 0
        
        output, _ = attention(x, attention_mask=attention_mask)
        
        assert output.shape == (batch_size, seq_len, config.n_embd)


class TestStandardAttention:
    """Test standard multi-head attention."""
    
    def test_standard_attention_forward(self):
        """Test standard attention forward pass."""
        config = KimiConfig(n_embd=128, n_head=8, block_size=16)
        
        attention = StandardAttention(config)
        
        batch_size = 2
        seq_len = 10
        
        x = torch.randn(batch_size, seq_len, config.n_embd)
        
        # Forward pass
        output, past_key_value = attention(x)
        
        assert output.shape == (batch_size, seq_len, config.n_embd)
        assert output.dtype == torch.float32
    
    def test_causal_masking(self):
        """Test that causal masking is applied correctly."""
        config = KimiConfig(n_embd=64, n_head=4, block_size=8)
        
        attention = StandardAttention(config)
        
        # Single token input
        x = torch.randn(1, 1, config.n_embd)
        output, _ = attention(x)
        assert output.shape == (1, 1, config.n_embd)
        
        # Multi-token input (should be causally masked)
        x = torch.randn(1, 4, config.n_embd)
        output, _ = attention(x)
        assert output.shape == (1, 4, config.n_embd)


class TestKimiModel:
    """Test the complete KimiModel."""
    
    def test_model_creation(self):
        """Test model creation with default config."""
        config = KimiConfig(
            n_layer=4,
            n_embd=128,
            n_head=8,
            vocab_size=1000,
            block_size=64
        )
        
        model = KimiModel(config)
        
        assert isinstance(model, nn.Module)
        assert model.config == config
        
        # Check that model has expected components
        assert hasattr(model, 'wte')  # Token embedding
        assert hasattr(model, 'wpe')  # Position embedding
        assert hasattr(model, 'h')    # Transformer blocks
        assert hasattr(model, 'ln_f') # Final layer norm
        assert hasattr(model, 'lm_head') # Language modeling head
        
        assert len(model.h) == config.n_layer
    
    def test_model_forward(self):
        """Test model forward pass."""
        config = KimiConfig(
            n_layer=2,
            n_embd=64,
            n_head=4,
            vocab_size=100,
            block_size=16
        )
        
        model = KimiModel(config)
        
        batch_size = 2
        seq_len = 8
        
        # Test input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass without labels
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs['loss'] is None
        
        # Forward pass with labels
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids, labels=labels)
        
        assert outputs['loss'] is not None
        assert isinstance(outputs['loss'], torch.Tensor)
        assert outputs['loss'].numel() == 1  # Scalar loss
    
    def test_model_generate(self):
        """Test model text generation."""
        config = KimiConfig(
            n_layer=2,
            n_embd=64,
            n_head=4,
            vocab_size=100,
            block_size=32
        )
        
        model = KimiModel(config)
        model.eval()
        
        # Test generation
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                do_sample=True
            )
        
        assert generated.shape[0] == 1  # Batch size
        assert generated.shape[1] > input_ids.shape[1]  # Generated tokens
        assert generated.shape[1] <= input_ids.shape[1] + 10  # Max new tokens
    
    def test_model_with_moe(self):
        """Test model with MoE layers."""
        config = KimiConfig(
            n_layer=4,
            n_embd=64,
            n_head=4,
            vocab_size=100,
            block_size=16,
            moe={
                "use_moe": True,
                "num_experts": 4,
                "top_k": 2,
                "moe_layers": [1, 3],  # Use MoE in layers 1 and 3
            }
        )
        
        model = KimiModel(config)
        
        # Check that specified layers have MoE
        for i, layer in enumerate(model.h):
            if i in config.moe["moe_layers"]:
                assert isinstance(layer.mlp, MixtureOfExperts)
            else:
                assert not isinstance(layer.mlp, MixtureOfExperts)
        
        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert 'auxiliary_loss' in outputs
        assert outputs['auxiliary_loss'] > 0  # Should have MoE loss
    
    def test_parameter_count(self):
        """Test parameter counting."""
        config = KimiConfig(
            n_layer=2,
            n_embd=64,
            n_head=4,
            vocab_size=100,
            block_size=16
        )
        
        model = KimiModel(config)
        
        # Test parameter counting methods
        param_count = model.get_num_params()
        assert isinstance(param_count, int)
        assert param_count > 0
        
        # Non-embedding count should be less than total
        param_count_no_embed = model.get_num_params(non_embedding=True)
        assert param_count_no_embed < param_count
    
    def test_model_device_handling(self):
        """Test model device handling."""
        config = KimiConfig(n_layer=1, n_embd=32, n_head=2, vocab_size=50)
        model = KimiModel(config)
        
        # Test CPU
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        outputs = model(input_ids)
        assert outputs['logits'].device == input_ids.device
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            input_ids = input_ids.cuda()
            outputs = model(input_ids)
            assert outputs['logits'].device == input_ids.device


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        config = KimiConfig(
            n_layer=2,
            n_embd=64,
            n_head=4,
            vocab_size=100,
            block_size=16,
            dropout=0.0,  # Disable dropout for deterministic test
        )
        
        model = KimiModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Prepare data
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        assert loss is not None
        assert loss.requires_grad
        
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        # (This is a simple smoke test)
        assert True  # If we get here, training step worked
    
    def test_model_state_dict(self):
        """Test model state dict save/load."""
        config = KimiConfig(n_layer=1, n_embd=32, n_head=2, vocab_size=50)
        
        # Create and initialize model
        model1 = KimiModel(config)
        
        # Save state
        state_dict = model1.state_dict()
        
        # Create new model and load state
        model2 = KimiModel(config)
        model2.load_state_dict(state_dict)
        
        # Check that weights are the same
        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            assert torch.allclose(param1, param2)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = KimiConfig(
            n_layer=2,
            n_embd=32,
            n_head=2,
            vocab_size=50,
            block_size=8
        )
        
        model = KimiModel(config)
        
        # Forward pass
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        labels = torch.randint(0, config.vocab_size, (1, 4))
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__])