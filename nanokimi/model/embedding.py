"""
Embedding layers for nanoKimi.

This module implements token and positional embeddings
with various encoding strategies.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional weight sharing."""
    
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.weight = nn.Parameter(torch.randn(vocab_size, n_embd))
        
        # Initialize embeddings
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of token embedding.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Token embeddings of shape (batch_size, seq_len, n_embd)
        """
        return F.embedding(input_ids, self.weight)


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding."""
    
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.weight = nn.Parameter(torch.randn(block_size, n_embd))
        
        # Initialize positional embeddings
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of positional embedding.
        
        Args:
            position_ids: Position indices of shape (seq_len,) or (batch_size, seq_len)
            
        Returns:
            Positional embeddings of shape (seq_len, n_embd) or (batch_size, seq_len, n_embd)
        """
        return F.embedding(position_ids, self.weight)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".
    
    This creates fixed sinusoidal embeddings that don't require learning
    and can generalize to longer sequences than seen during training.
    """
    
    def __init__(self, n_embd: int, max_len: int = 5000):
        super().__init__()
        self.n_embd = n_embd
        
        # Create sinusoidal embeddings
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * 
                            (-math.log(10000.0) / n_embd))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of sinusoidal positional embedding.
        
        Args:
            position_ids: Position indices
            
        Returns:
            Sinusoidal positional embeddings
        """
        return self.pe[:, position_ids, :]


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer.
    
    This applies rotational position encoding directly to the
    query and key vectors in attention.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position embeddings
        self._update_cos_sin_cache(max_seq_len)
    
    def _update_cos_sin_cache(self, seq_len: int):
        """Update cached cosine and sine values."""
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple:
        """
        Apply rotary positional embedding to queries and keys.
        
        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, n_heads, seq_len, head_dim)
            seq_len: Sequence length
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to input tensor."""
        # Split x into pairs for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        rotated = torch.stack([
            x1 * cos[..., ::2] - x2 * sin[..., ::2],
            x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        ], dim=-1)
        
        return rotated.flatten(-2)


class ALiBiEmbedding(nn.Module):
    """
    Attention with Linear Biases (ALiBi) from "Train Short, Test Long".
    
    This adds linear biases to attention scores based on position
    distance, allowing for better length generalization.
    """
    
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        
        # Compute slopes for each head
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = self._get_slopes(2*closest_power_of_2)[0::2][:n_heads-closest_power_of_2]
            return torch.tensor(slopes_a + slopes_b)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate ALiBi bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            ALiBi bias tensor of shape (n_heads, seq_len, seq_len)
        """
        # Create position distance matrix
        positions = torch.arange(seq_len, device=self.slopes.device)
        distance_matrix = positions[None, :] - positions[:, None]
        
        # Apply slopes to create bias matrix
        bias = distance_matrix[None, :, :] * self.slopes[:, None, None]
        
        return bias


class CompositionalEmbedding(nn.Module):
    """
    Compositional embedding that combines multiple embedding types.
    
    This allows flexible combination of token, positional, and other
    embedding types for experimentation.
    """
    
    def __init__(
        self, 
        vocab_size: int,
        n_embd: int,
        block_size: int,
        embedding_types: list = ['token', 'positional'],
        positional_type: str = 'learned'
    ):
        super().__init__()
        self.embedding_types = embedding_types
        self.n_embd = n_embd
        
        # Token embeddings
        if 'token' in embedding_types:
            self.token_emb = TokenEmbedding(vocab_size, n_embd)
        
        # Positional embeddings
        if 'positional' in embedding_types:
            if positional_type == 'learned':
                self.pos_emb = PositionalEmbedding(block_size, n_embd)
            elif positional_type == 'sinusoidal':
                self.pos_emb = SinusoidalPositionalEmbedding(n_embd)
            else:
                raise ValueError(f"Unknown positional type: {positional_type}")
        
        # Layer norm for embedding sum
        if len(embedding_types) > 1:
            self.ln_emb = nn.LayerNorm(n_embd)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass combining multiple embedding types.
        
        Args:
            input_ids: Token indices
            position_ids: Optional position indices
            
        Returns:
            Combined embeddings
        """
        embeddings = []
        
        # Token embeddings
        if 'token' in self.embedding_types:
            embeddings.append(self.token_emb(input_ids))
        
        # Positional embeddings
        if 'positional' in self.embedding_types:
            if position_ids is None:
                seq_len = input_ids.size(-1)
                position_ids = torch.arange(seq_len, device=input_ids.device)
            embeddings.append(self.pos_emb(position_ids))
        
        # Combine embeddings
        if len(embeddings) == 1:
            output = embeddings[0]
        else:
            output = sum(embeddings)
            if hasattr(self, 'ln_emb'):
                output = self.ln_emb(output)
        
        return output