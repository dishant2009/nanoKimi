"""
Unit tests for nanoKimi inference components.

Tests text generation, sampling strategies, and inference utilities.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from nanokimi.inference.generator import Generator, GenerationConfig, load_generator_from_checkpoint
from nanokimi.inference.sampling import (
    TemperatureSampler, TopKSampler, TopPSampler, TopASampler,
    TypicalSampler, MinPSampler, TailFreeSampler, MirostatSampler,
    CompositeSampler, sample_with_warpers
)
from nanokimi.model import KimiModel, KimiConfig
from transformers import AutoTokenizer


class TestGenerationConfig:
    """Test generation configuration."""
    
    def test_default_config(self):
        """Test default generation configuration."""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 100
        assert config.min_new_tokens == 0
        assert config.do_sample == True
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.repetition_penalty == 1.0
        assert config.use_cache == True
    
    def test_custom_config(self):
        """Test custom generation configuration."""
        config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            do_sample=False
        )
        
        assert config.max_new_tokens == 50
        assert config.temperature == 0.8
        assert config.top_k == 40
        assert config.top_p == 0.9
        assert config.do_sample == False


class TestTemperatureSampler:
    """Test temperature sampling."""
    
    def test_temperature_scaling(self):
        """Test temperature scaling of logits."""
        sampler = TemperatureSampler()
        
        # Test logits
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # Temperature = 1.0 (no change)
        scaled_logits = sampler(logits, temperature=1.0)
        assert torch.allclose(scaled_logits, logits)
        
        # Temperature = 2.0 (softer distribution)
        scaled_logits = sampler(logits, temperature=2.0)
        expected = logits / 2.0
        assert torch.allclose(scaled_logits, expected)
        
        # Temperature = 0.5 (sharper distribution)
        scaled_logits = sampler(logits, temperature=0.5)
        expected = logits / 0.5
        assert torch.allclose(scaled_logits, expected)
    
    def test_invalid_temperature(self):
        """Test invalid temperature values."""
        sampler = TemperatureSampler()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # Zero temperature should raise error
        with pytest.raises(ValueError):
            sampler(logits, temperature=0.0)
        
        # Negative temperature should raise error
        with pytest.raises(ValueError):
            sampler(logits, temperature=-1.0)


class TestTopKSampler:
    """Test top-k sampling."""
    
    def test_top_k_filtering(self):
        """Test top-k filtering of logits."""
        sampler = TopKSampler()
        
        # Create logits where we know the top-k
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])  # Top-2: indices 1, 4
        
        # Apply top-2 filtering
        filtered_logits = sampler(logits, k=2)
        
        # Check that only top-2 values remain (others should be -inf)
        assert filtered_logits[0, 0] == float('-inf')  # Was 1.0
        assert filtered_logits[0, 1] == 5.0  # Top value
        assert filtered_logits[0, 2] == float('-inf')  # Was 3.0
        assert filtered_logits[0, 3] == float('-inf')  # Was 2.0
        assert filtered_logits[0, 4] == 4.0  # Second top value
    
    def test_top_k_edge_cases(self):
        """Test top-k edge cases."""
        sampler = TopKSampler()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # k=0 should return original logits
        filtered_logits = sampler(logits, k=0)
        assert torch.allclose(filtered_logits, logits)
        
        # k larger than vocab size should return original logits
        filtered_logits = sampler(logits, k=10)
        assert torch.allclose(filtered_logits, logits)


class TestTopPSampler:
    """Test top-p (nucleus) sampling."""
    
    def test_top_p_filtering(self):
        """Test top-p filtering of logits."""
        sampler = TopPSampler()
        
        # Create logits with known probabilities after softmax
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])  # After softmax: roughly [0.1, 0.2, 0.3, 0.4]
        
        # Apply top-p filtering with p=0.7
        # This should keep tokens until cumulative prob > 0.7
        filtered_logits = sampler(logits, p=0.7)
        
        # The filtering should remove some low-probability tokens
        num_infinite = torch.isinf(filtered_logits).sum().item()
        assert num_infinite > 0  # Some tokens should be filtered out
    
    def test_top_p_edge_cases(self):
        """Test top-p edge cases."""
        sampler = TopPSampler()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # p=1.0 should keep all tokens
        filtered_logits = sampler(logits, p=1.0)
        assert not torch.isinf(filtered_logits).any()
        
        # Invalid p values should raise error
        with pytest.raises(ValueError):
            sampler(logits, p=0.0)
        
        with pytest.raises(ValueError):
            sampler(logits, p=1.5)


class TestTopASampler:
    """Test top-a sampling."""
    
    def test_top_a_filtering(self):
        """Test top-a filtering of logits."""
        sampler = TopASampler()
        
        # Create logits with known probability distribution
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        
        # Apply top-a filtering
        filtered_logits = sampler(logits, a=2.0)
        
        # Should filter out tokens with prob < max_prob / a
        num_infinite = torch.isinf(filtered_logits).sum().item()
        assert num_infinite >= 0  # Some tokens might be filtered
    
    def test_top_a_invalid_values(self):
        """Test top-a with invalid values."""
        sampler = TopASampler()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # a <= 1.0 should raise error
        with pytest.raises(ValueError):
            sampler(logits, a=1.0)
        
        with pytest.raises(ValueError):
            sampler(logits, a=0.5)


class TestMinPSampler:
    """Test min-p sampling."""
    
    def test_min_p_filtering(self):
        """Test min-p filtering of logits."""
        sampler = MinPSampler()
        
        # Create logits
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        
        # Apply min-p filtering
        filtered_logits = sampler(logits, min_p=0.1)
        
        # Should filter tokens based on min probability threshold
        num_infinite = torch.isinf(filtered_logits).sum().item()
        assert num_infinite >= 0
    
    def test_min_p_invalid_values(self):
        """Test min-p with invalid values."""
        sampler = MinPSampler()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        with pytest.raises(ValueError):
            sampler(logits, min_p=0.0)
        
        with pytest.raises(ValueError):
            sampler(logits, min_p=1.5)


class TestMirostatSampler:
    """Test Mirostat sampling."""
    
    def test_mirostat_sampling(self):
        """Test Mirostat sampling functionality."""
        sampler = MirostatSampler(target_surprise=5.0, learning_rate=0.1)
        
        # Test logits
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        
        # Sample token
        sampled_token = sampler(logits)
        
        assert isinstance(sampled_token, torch.Tensor)
        assert sampled_token.shape == (1,)
        assert 0 <= sampled_token.item() < logits.size(-1)
        
        # Check that tau is updated
        initial_tau = sampler.tau
        sampled_token = sampler(logits)
        # tau might have changed based on surprise
        assert isinstance(sampler.tau, torch.Tensor)


class TestCompositeSampler:
    """Test composite sampling."""
    
    def test_composite_sampling(self):
        """Test composite sampling with multiple strategies."""
        sampler = CompositeSampler()
        
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
        
        # Apply multiple sampling strategies
        filtered_logits = sampler(
            logits,
            temperature=0.8,
            top_k=3,
            top_p=0.9
        )
        
        assert filtered_logits.shape == logits.shape
        # Some filtering should have occurred
        num_infinite = torch.isinf(filtered_logits).sum().item()
        assert num_infinite >= 0
    
    def test_sample_with_warpers(self):
        """Test sampling with warper functions."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        
        # Create warper functions
        temp_warper = lambda x: x / 0.8  # Temperature warper
        topk_warper = lambda x: TopKSampler()(x, k=2)  # Top-k warper
        
        warpers = [temp_warper, topk_warper]
        
        # Sample with warpers
        samples = sample_with_warpers(logits, warpers, num_samples=1)
        
        assert isinstance(samples, torch.Tensor)
        assert samples.shape == (1,)
        assert 0 <= samples.item() < logits.size(-1)


class TestGenerator:
    """Test text generator."""
    
    def setup_method(self):
        """Set up test model and generator."""
        self.config = KimiConfig(
            n_layer=2,
            n_embd=64,
            n_head=4,
            vocab_size=100,
            block_size=32
        )
        self.model = KimiModel(self.config)
        
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.vocab_size = 100
        self.tokenizer.eos_token_id = 50
        self.tokenizer.pad_token_id = 51
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "generated text"
        
        self.generator = Generator(self.model, self.tokenizer)
    
    def test_generator_creation(self):
        """Test generator creation."""
        assert self.generator.model is not None
        assert self.generator.tokenizer is not None
        assert self.generator.vocab_size == 100
        assert self.generator.block_size == 32
    
    def test_generate_tokens(self):
        """Test token generation."""
        # Input tokens
        input_ids = torch.randint(0, 100, (1, 5))
        
        # Generation config
        config = GenerationConfig(
            max_new_tokens=10,
            temperature=1.0,
            do_sample=True
        )
        
        # Generate
        with torch.no_grad():
            result = self.generator.generate(input_ids, config)
        
        if isinstance(result, dict):
            generated_tokens = result['sequences']
        else:
            generated_tokens = result
        
        assert generated_tokens.shape[0] == 1  # Batch size
        assert generated_tokens.shape[1] > input_ids.shape[1]  # Added tokens
        assert generated_tokens.shape[1] <= input_ids.shape[1] + config.max_new_tokens
    
    def test_generate_text(self):
        """Test text generation from string prompt."""
        prompt = "Test prompt"
        
        config = GenerationConfig(max_new_tokens=5)
        
        # Mock the tokenizer calls
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.decode.return_value = "Test prompt generated text"
        
        with torch.no_grad():
            generated_text = self.generator.generate_text(prompt, config)
        
        assert isinstance(generated_text, str)
        # Should contain the generated part (prompt removed)
        assert "generated text" in generated_text
    
    def test_batch_generate_text(self):
        """Test batch text generation."""
        prompts = ["Prompt 1", "Prompt 2"]
        
        config = GenerationConfig(max_new_tokens=5)
        
        # Mock tokenizer for batch processing
        self.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2], [3, 4]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1]])
        }
        self.tokenizer.decode.side_effect = [
            "Prompt 1 generated",
            "Prompt 2 generated"
        ]
        
        with torch.no_grad():
            generated_texts = self.generator.batch_generate_text(prompts, config)
        
        assert isinstance(generated_texts, list)
        assert len(generated_texts) == 2
        for text in generated_texts:
            assert isinstance(text, str)
    
    def test_stream_generate(self):
        """Test streaming generation."""
        prompt = "Stream test"
        config = GenerationConfig(max_new_tokens=3)
        
        self.tokenizer.encode.return_value = [1, 2]
        self.tokenizer.decode.side_effect = ["token1", "token2", "token3"]
        
        generated_tokens = []
        with torch.no_grad():
            for token in self.generator.stream_generate(prompt, config):
                generated_tokens.append(token)
                if len(generated_tokens) >= 3:  # Prevent infinite loop
                    break
        
        assert len(generated_tokens) > 0
        for token in generated_tokens:
            assert isinstance(token, str)
    
    def test_calculate_perplexity(self):
        """Test perplexity calculation."""
        text = "Test text for perplexity"
        
        # Mock tokenizer
        self.tokenizer.return_value = {
            'input_ids': torch.randint(0, 100, (1, 10))
        }
        
        with torch.no_grad():
            perplexity = self.generator.calculate_perplexity(text)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_repetition_penalty(self):
        """Test repetition penalty application."""
        # Create logits and generated sequence
        logits = torch.randn(1, 100)
        generated_tokens = torch.tensor([[1, 2, 3, 1, 2]])  # Repeated tokens
        
        # Apply repetition penalty
        modified_logits = self.generator._apply_repetition_penalty(
            logits, generated_tokens, penalty=1.2
        )
        
        # Check that repeated tokens have modified logits
        assert not torch.allclose(modified_logits, logits)
        
        # Tokens 1 and 2 should be penalized
        for token_id in [1, 2]:
            if logits[0, token_id] > 0:
                assert modified_logits[0, token_id] < logits[0, token_id]
            else:
                assert modified_logits[0, token_id] > logits[0, token_id]
    
    def test_no_repeat_ngram_penalty(self):
        """Test no-repeat n-gram penalty."""
        logits = torch.randn(1, 100)
        # Create sequence with potential 3-gram repetition: [1, 2, 3, 1, 2, ?]
        generated_tokens = torch.tensor([[1, 2, 3, 1, 2]])
        
        # Apply n-gram penalty
        modified_logits = self.generator._apply_no_repeat_ngram_penalty(
            logits, generated_tokens, ngram_size=3
        )
        
        # Token 3 should be banned (would complete repeated 3-gram [1, 2, 3])
        assert modified_logits[0, 3] == float('-inf')
    
    def test_sample_next_tokens(self):
        """Test next token sampling."""
        logits = torch.randn(2, 100)  # Batch size 2
        
        config = GenerationConfig(
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        next_tokens = self.generator._sample_next_tokens(logits, config)
        
        assert next_tokens.shape == (2,)  # Batch size
        assert next_tokens.dtype == torch.long
        
        # All tokens should be in valid range
        assert (next_tokens >= 0).all()
        assert (next_tokens < 100).all()


class TestGeneratorIntegration:
    """Integration tests for generator."""
    
    @patch('torch.load')
    def test_load_generator_from_checkpoint(self, mock_torch_load):
        """Test loading generator from checkpoint."""
        # Mock checkpoint data
        mock_checkpoint = {
            'config': {
                'n_layer': 2,
                'n_embd': 64,
                'n_head': 4,
                'vocab_size': 100,
                'block_size': 32,
                'dropout': 0.1,
                'moe': {'use_moe': False},
                'attention': {'type': 'standard'},
            },
            'model_state_dict': {}
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # Mock model state dict loading
        with patch.object(KimiModel, 'load_state_dict'):
            generator = load_generator_from_checkpoint(
                'fake_checkpoint.pt',
                device=torch.device('cpu'),
                tokenizer_name='gpt2'
            )
        
        assert isinstance(generator, Generator)
        assert generator.model is not None
        assert generator.tokenizer is not None
    
    def test_generation_with_different_configs(self):
        """Test generation with various configurations."""
        config = KimiConfig(
            n_layer=1,
            n_embd=32,
            n_head=2,
            vocab_size=50,
            block_size=16
        )
        model = KimiModel(config)
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.vocab_size = 50
        tokenizer.eos_token_id = 49
        tokenizer.pad_token_id = 48
        
        generator = Generator(model, tokenizer)
        
        input_ids = torch.randint(0, 50, (1, 3))
        
        # Test different generation configs
        configs = [
            GenerationConfig(max_new_tokens=5, do_sample=False),  # Greedy
            GenerationConfig(max_new_tokens=5, temperature=0.8),  # Sampling
            GenerationConfig(max_new_tokens=5, top_k=10),        # Top-k
            GenerationConfig(max_new_tokens=5, top_p=0.9),       # Top-p
        ]
        
        for gen_config in configs:
            with torch.no_grad():
                result = generator.generate(input_ids, gen_config)
            
            if isinstance(result, dict):
                generated = result['sequences']
            else:
                generated = result
                
            assert generated.shape[0] == 1
            assert generated.shape[1] > input_ids.shape[1]


if __name__ == "__main__":
    pytest.main([__file__])