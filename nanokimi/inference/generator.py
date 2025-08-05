"""
Text generation and inference for nanoKimi.

This module implements efficient text generation with various
sampling strategies and optimization techniques.
"""

import time
from typing import List, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..model import KimiModel, KimiConfig
from .sampling import TopKSampler, TopPSampler, TemperatureSampler


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    # Generation parameters
    max_new_tokens: int = 100
    min_new_tokens: int = 0
    
    # Sampling parameters
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Special tokens
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    # Generation control
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # Performance
    use_cache: bool = True
    batch_size: int = 1
    
    # Output control
    return_dict_in_generate: bool = True
    output_scores: bool = False
    output_attentions: bool = False


class Generator:
    """
    Text generator for nanoKimi models.
    
    Provides efficient text generation with various sampling strategies,
    caching, and optimization techniques.
    """
    
    def __init__(
        self,
        model: KimiModel,
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        # Load tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        self.tokenizer = tokenizer
        
        # Sampling strategies
        self.temperature_sampler = TemperatureSampler()
        self.top_k_sampler = TopKSampler()
        self.top_p_sampler = TopPSampler()
        
        # Model info
        self.vocab_size = model.config.vocab_size
        self.block_size = model.config.block_size
        
    def generate(
        self,
        input_ids: torch.Tensor,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate text from input tokens.
        
        Args:
            input_ids: Input token tensor of shape (batch_size, seq_len)
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated tokens or generation results dictionary
        """
        if config is None:
            config = GenerationConfig(**kwargs)
        else:
            # Override config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Validate inputs
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, input_length = input_ids.shape
        
        # Setup special tokens
        eos_token_id = config.eos_token_id or self.tokenizer.eos_token_id
        pad_token_id = config.pad_token_id or self.tokenizer.pad_token_id
        
        # Generation state
        past_key_values = None
        generated_tokens = input_ids.clone()
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generation loop
        generation_start_time = time.time()
        scores_list = [] if config.output_scores else None
        
        self.model.eval()
        
        with torch.no_grad():
            for step in range(config.max_new_tokens):
                # Prepare model inputs
                if config.use_cache and past_key_values is not None:
                    # Only pass the last token for cached generation
                    model_inputs = generated_tokens[:, -1:]
                else:
                    model_inputs = generated_tokens
                
                # Forward pass
                outputs = self.model(
                    model_inputs,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache,
                )
                
                logits = outputs["logits"][:, -1, :]  # (batch_size, vocab_size)
                
                if config.use_cache:
                    past_key_values = outputs.get("past_key_values")
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, generated_tokens, config.repetition_penalty
                    )
                
                # Apply no-repeat n-gram penalty
                if config.no_repeat_ngram_size > 0:
                    logits = self._apply_no_repeat_ngram_penalty(
                        logits, generated_tokens, config.no_repeat_ngram_size
                    )
                
                # Sample next tokens
                if config.do_sample:
                    next_tokens = self._sample_next_tokens(logits, config)
                else:
                    next_tokens = torch.argmax(logits, dim=-1)
                
                # Store scores if requested
                if config.output_scores:
                    scores_list.append(logits)
                
                # Update generated sequence
                generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(1)], dim=1)
                
                # Check for finished sequences
                if eos_token_id is not None:
                    is_finished |= (next_tokens == eos_token_id)
                
                # Stop if all sequences are finished
                if config.min_new_tokens <= step and is_finished.all():
                    break
                
                # Stop if we've reached the maximum sequence length
                if generated_tokens.size(1) >= self.block_size:
                    break
        
        generation_time = time.time() - generation_start_time
        
        # Prepare output
        if config.return_dict_in_generate:
            result = {
                "sequences": generated_tokens,
                "generation_time": generation_time,
                "tokens_per_second": config.max_new_tokens / generation_time,
            }
            
            if config.output_scores:
                result["scores"] = torch.stack(scores_list, dim=1)
            
            return result
        else:
            return generated_tokens
    
    def _sample_next_tokens(
        self, 
        logits: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Sample next tokens using specified sampling strategy.
        
        Args:
            logits: Model logits of shape (batch_size, vocab_size)
            config: Generation configuration
            
        Returns:
            Sampled token ids of shape (batch_size,)
        """
        # Apply temperature
        if config.temperature != 1.0:
            logits = self.temperature_sampler(logits, config.temperature)
        
        # Apply top-k filtering
        if config.top_k is not None:
            logits = self.top_k_sampler(logits, config.top_k)
        
        # Apply top-p filtering
        if config.top_p is not None:
            logits = self.top_p_sampler(logits, config.top_p)
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_tokens
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated_tokens: torch.Tensor, 
        penalty: float
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits.
        
        Args:
            logits: Model logits
            generated_tokens: Previously generated tokens
            penalty: Repetition penalty factor
            
        Returns:
            Modified logits with repetition penalty
        """
        if penalty == 1.0:
            return logits
        
        batch_size = logits.size(0)
        
        for i in range(batch_size):
            # Get unique tokens in the sequence
            unique_tokens = torch.unique(generated_tokens[i])
            
            # Apply penalty
            for token in unique_tokens:
                if logits[i, token] > 0:
                    logits[i, token] /= penalty
                else:
                    logits[i, token] *= penalty
        
        return logits
    
    def _apply_no_repeat_ngram_penalty(
        self, 
        logits: torch.Tensor, 
        generated_tokens: torch.Tensor, 
        ngram_size: int
    ) -> torch.Tensor:
        """
        Apply no-repeat n-gram penalty.
        
        Args:
            logits: Model logits
            generated_tokens: Previously generated tokens
            ngram_size: Size of n-grams to check
            
        Returns:
            Modified logits
        """
        if ngram_size <= 0:
            return logits
        
        batch_size, seq_len = generated_tokens.shape
        
        for i in range(batch_size):
            # Check for potential n-gram repetitions
            if seq_len >= ngram_size:
                # Get the last (ngram_size - 1) tokens
                prefix = generated_tokens[i, -(ngram_size - 1):].tolist()
                
                # Find all n-grams that start with this prefix
                for j in range(seq_len - ngram_size + 1):
                    ngram = generated_tokens[i, j:j + ngram_size].tolist()
                    if ngram[:-1] == prefix:
                        # Prevent repeating this n-gram
                        banned_token = ngram[-1]
                        logits[i, banned_token] = float('-inf')
        
        return logits
    
    def generate_text(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate text from string prompt.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate tokens
        outputs = self.generate(input_ids, config, **kwargs)
        
        if isinstance(outputs, dict):
            generated_tokens = outputs["sequences"]
        else:
            generated_tokens = outputs
        
        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_tokens[0], 
            skip_special_tokens=True
        )
        
        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        return generated_text
    
    def batch_generate_text(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text strings
        """
        # Tokenize all prompts
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.block_size // 2,
        )
        
        input_ids = tokenized["input_ids"].to(self.device)
        
        # Update config for batch processing
        if config is None:
            config = GenerationConfig(**kwargs)
        config.batch_size = len(prompts)
        
        # Generate tokens
        outputs = self.generate(input_ids, config)
        
        if isinstance(outputs, dict):
            generated_tokens = outputs["sequences"]
        else:
            generated_tokens = outputs
        
        # Decode all generated texts
        generated_texts = []
        for i, prompt in enumerate(prompts):
            generated_text = self.tokenizer.decode(
                generated_tokens[i],
                skip_special_tokens=True
            )
            
            # Remove prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ):
        """
        Stream generated tokens one by one.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        if config is None:
            config = GenerationConfig(**kwargs)
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = input_ids.clone()
        
        past_key_values = None
        
        self.model.eval()
        
        with torch.no_grad():
            for step in range(config.max_new_tokens):
                # Prepare model inputs
                if config.use_cache and past_key_values is not None:
                    model_inputs = generated_tokens[:, -1:]
                else:
                    model_inputs = generated_tokens
                
                # Forward pass
                outputs = self.model(
                    model_inputs,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache,
                )
                
                logits = outputs["logits"][:, -1, :]
                
                if config.use_cache:
                    past_key_values = outputs.get("past_key_values")
                
                # Sample next token
                if config.do_sample:
                    next_token = self._sample_next_tokens(logits, config)
                else:
                    next_token = torch.argmax(logits, dim=-1)
                
                # Update sequence
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)
                
                # Decode and yield new token
                new_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield new_text
                
                # Check for stopping conditions
                if (config.eos_token_id is not None and 
                    next_token[0] == config.eos_token_id):
                    break
                
                if generated_tokens.size(1) >= self.block_size:
                    break
    
    def calculate_perplexity(
        self,
        text: str,
        stride: int = 512,
    ) -> float:
        """
        Calculate perplexity of text under the model.
        
        Args:
            text: Input text
            stride: Stride for sliding window evaluation
            
        Returns:
            Perplexity score
        """
        # Tokenize text
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        seq_len = input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + self.block_size, seq_len)
                trg_len = end_loc - prev_end_loc
                
                input_batch = input_ids[:, begin_loc:end_loc]
                target_ids = input_ids[:, begin_loc:end_loc].clone()
                target_ids[:, :-trg_len] = -100
                
                outputs = self.model(input_batch, labels=target_ids)
                neg_log_likelihood = outputs["loss"]
                
                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
                
                if end_loc == seq_len:
                    break
        
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()


def load_generator_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    tokenizer_name: str = "gpt2",
) -> Generator:
    """
    Load generator from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        tokenizer_name: Name of tokenizer to use
        
    Returns:
        Loaded generator
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model config
    config = KimiConfig.from_dict(checkpoint["config"])
    model = KimiModel(config)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create generator
    generator = Generator(model, tokenizer, device)
    
    return generator