"""
Sampling strategies for text generation in nanoKimi.

This module implements various sampling techniques for controlling
text generation quality and diversity.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class TemperatureSampler:
    """
    Temperature-based sampling.
    
    Controls randomness in generation by scaling logits.
    Higher temperature = more random, lower = more deterministic.
    """
    
    def __call__(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits of shape (..., vocab_size)
            temperature: Temperature parameter (> 0)
            
        Returns:
            Temperature-scaled logits
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        return logits / temperature


class TopKSampler:
    """
    Top-K sampling.
    
    Only considers the top K most likely tokens at each step.
    """
    
    def __call__(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply top-k filtering to logits.
        
        Args:
            logits: Model logits of shape (..., vocab_size)
            k: Number of top tokens to keep
            
        Returns:
            Filtered logits with only top-k tokens
        """
        if k <= 0:
            return logits
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
        
        # Create mask for top-k tokens
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_values)
        
        return mask


class TopPSampler:
    """
    Top-P (nucleus) sampling.
    
    Samples from the smallest set of tokens whose cumulative probability
    exceeds the threshold P.
    """
    
    def __call__(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Apply top-p (nucleus) filtering to logits.
        
        Args:
            logits: Model logits of shape (..., vocab_size)
            p: Cumulative probability threshold (0 < p <= 1)
            
        Returns:
            Filtered logits with nucleus sampling
        """
        if not 0 < p <= 1:
            raise ValueError("p must be between 0 and 1")
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Create mask for tokens to remove (cumulative probability > p)
        sorted_indices_to_remove = cumulative_probs > p
        
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create final mask
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        
        # Set removed tokens to -inf
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        return filtered_logits


class TopASampler:
    """
    Top-A sampling.
    
    Dynamically selects tokens based on the maximum probability,
    filtering out tokens with probability less than max_prob / A.
    """
    
    def __call__(self, logits: torch.Tensor, a: float) -> torch.Tensor:
        """
        Apply top-a filtering to logits.
        
        Args:
            logits: Model logits of shape (..., vocab_size)
            a: Top-a parameter (> 1)
            
        Returns:
            Filtered logits with top-a sampling
        """
        if a <= 1:
            raise ValueError("a must be greater than 1")
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get maximum probability
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
        
        # Create mask for tokens below threshold
        threshold = max_prob / a
        mask = probs < threshold
        
        # Apply mask
        filtered_logits = logits.clone()
        filtered_logits[mask] = float('-inf')
        
        return filtered_logits


class TypicalSampler:
    """
    Typical sampling.
    
    Samples tokens with typical information content,
    filtering out both very high and very low probability tokens.
    """
    
    def __call__(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Apply typical sampling to logits.
        
        Args:
            logits: Model logits of shape (..., vocab_size)
            tau: Typical sampling parameter (0 < tau < 1)
            
        Returns:
            Filtered logits with typical sampling
        """
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1")
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Calculate information content: -log(p)
        info_content = -torch.log(probs + 1e-10)
        
        # Calculate entropy (expected information content)
        entropy = torch.sum(probs * info_content, dim=-1, keepdim=True)
        
        # Calculate absolute deviation from entropy
        deviation = torch.abs(info_content - entropy)
        
        # Sort by deviation
        sorted_deviations, sorted_indices = torch.sort(deviation, dim=-1)
        sorted_probs = torch.gather(probs, -1, sorted_indices)
        
        # Find cumulative probability threshold
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        last_ind = torch.searchsorted(cumulative_probs, tau, right=False)
        
        # Create mask
        mask = torch.zeros_like(probs, dtype=torch.bool)
        for i in range(probs.size(0)):
            indices_to_keep = sorted_indices[i, :last_ind[i] + 1]
            mask[i, indices_to_keep] = True
        
        # Apply mask
        filtered_logits = logits.clone()
        filtered_logits[~mask] = float('-inf')
        
        return filtered_logits


class MinPSampler:
    """
    Min-P sampling.
    
    Filters tokens based on a minimum probability relative to the
    most likely token.
    """
    
    def __call__(self, logits: torch.Tensor, min_p: float) -> torch.Tensor:
        """
        Apply min-p filtering to logits.
        
        Args:
            logits: Model logits of shape (..., vocab_size)
            min_p: Minimum probability threshold (0 < min_p <= 1)
            
        Returns:
            Filtered logits with min-p sampling
        """
        if not 0 < min_p <= 1:
            raise ValueError("min_p must be between 0 and 1")
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get maximum probability
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
        
        # Calculate threshold
        threshold = max_prob * min_p
        
        # Create mask
        mask = probs < threshold
        
        # Apply mask
        filtered_logits = logits.clone()
        filtered_logits[mask] = float('-inf')
        
        return filtered_logits


class TailFreeSampler:
    """
    Tail-free sampling.
    
    Removes the tail of the distribution based on the second derivative
    of the cumulative distribution function.
    """
    
    def __call__(self, logits: torch.Tensor, z: float) -> torch.Tensor:
        """
        Apply tail-free sampling to logits.
        
        Args:
            logits: Model logits of shape (..., vocab_size)
            z: Tail-free parameter (0 < z < 1)
            
        Returns:
            Filtered logits with tail-free sampling
        """
        if not 0 < z < 1:
            raise ValueError("z must be between 0 and 1")
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Convert to probabilities
        probs = F.softmax(sorted_logits, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        # Calculate second derivative (finite differences)
        # d²/dx² CDF ≈ CDF[i+1] - 2*CDF[i] + CDF[i-1]
        second_derivative = torch.zeros_like(cumulative_probs)
        if cumulative_probs.size(-1) >= 3:
            second_derivative[..., 1:-1] = (
                cumulative_probs[..., 2:] - 
                2 * cumulative_probs[..., 1:-1] + 
                cumulative_probs[..., :-2]
            )
        
        # Find where to cut off (where second derivative becomes small)
        cutoff = torch.sum(torch.abs(second_derivative) > z, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(probs, dtype=torch.bool)
        for i in range(probs.size(0)):
            if cutoff[i] > 0:
                mask[i, :cutoff[i]] = True
            else:
                mask[i, 0] = True  # Keep at least one token
        
        # Apply mask to original logits
        indices_to_remove = ~mask.scatter(-1, sorted_indices, mask)
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        return filtered_logits


class MirostatSampler:
    """
    Mirostat sampling.
    
    Maintains a target perplexity by dynamically adjusting the sampling
    temperature based on recent performance.
    """
    
    def __init__(self, target_surprise: float = 5.0, learning_rate: float = 0.1):
        """
        Initialize Mirostat sampler.
        
        Args:
            target_surprise: Target surprise value (related to perplexity)
            learning_rate: Learning rate for adaptation
        """
        self.target_surprise = target_surprise
        self.learning_rate = learning_rate
        self.tau = 1.0  # Initial temperature
    
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Mirostat sampling to logits.
        
        Args:
            logits: Model logits of shape (..., vocab_size)
            
        Returns:
            Sampled token indices
        """
        # Convert to probabilities
        probs = F.softmax(logits / self.tau, dim=-1)
        
        # Sample token
        sampled_token = torch.multinomial(probs, num_samples=1)
        
        # Calculate observed surprise
        token_prob = probs.gather(-1, sampled_token)
        observed_surprise = -torch.log2(token_prob)
        
        # Update tau based on surprise error
        surprise_error = observed_surprise - self.target_surprise
        self.tau = self.tau - self.learning_rate * surprise_error
        self.tau = torch.clamp(self.tau, min=0.1, max=10.0)  # Keep tau in reasonable range
        
        return sampled_token.squeeze(-1)


class CompositeSampler:
    """
    Composite sampler that combines multiple sampling strategies.
    
    Applies sampling methods in sequence to progressively filter
    the token distribution.
    """
    
    def __init__(self):
        self.temperature_sampler = TemperatureSampler()
        self.top_k_sampler = TopKSampler()
        self.top_p_sampler = TopPSampler()
        self.top_a_sampler = TopASampler()
        self.typical_sampler = TypicalSampler()
        self.min_p_sampler = MinPSampler()
        self.tail_free_sampler = TailFreeSampler()
    
    def __call__(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        top_a: Optional[float] = None,
        typical_p: Optional[float] = None,
        min_p: Optional[float] = None,
        tail_free_z: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply composite sampling to logits.
        
        Args:
            logits: Model logits
            temperature: Temperature for scaling
            top_k: Top-k parameter
            top_p: Top-p parameter
            top_a: Top-a parameter
            typical_p: Typical sampling parameter
            min_p: Min-p parameter
            tail_free_z: Tail-free parameter
            
        Returns:
            Filtered logits
        """
        filtered_logits = logits.clone()
        
        # Apply filters in order
        if tail_free_z is not None:
            filtered_logits = self.tail_free_sampler(filtered_logits, tail_free_z)
        
        if typical_p is not None:
            filtered_logits = self.typical_sampler(filtered_logits, typical_p)
        
        if top_a is not None:
            filtered_logits = self.top_a_sampler(filtered_logits, top_a)
        
        if min_p is not None:
            filtered_logits = self.min_p_sampler(filtered_logits, min_p)
        
        if top_k is not None:
            filtered_logits = self.top_k_sampler(filtered_logits, top_k)
        
        if top_p is not None:
            filtered_logits = self.top_p_sampler(filtered_logits, top_p)
        
        if temperature is not None:
            filtered_logits = self.temperature_sampler(filtered_logits, temperature)
        
        return filtered_logits


def sample_with_warpers(
    logits: torch.Tensor,
    warpers: list,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Sample tokens using a list of logit warpers.
    
    Args:
        logits: Model logits
        warpers: List of sampling functions
        num_samples: Number of samples to draw
        
    Returns:
        Sampled token indices
    """
    # Apply all warpers in sequence
    for warper in warpers:
        if callable(warper):
            logits = warper(logits)
    
    # Sample from the final distribution
    probs = F.softmax(logits, dim=-1)
    samples = torch.multinomial(probs, num_samples=num_samples)
    
    if num_samples == 1:
        samples = samples.squeeze(-1)
    
    return samples