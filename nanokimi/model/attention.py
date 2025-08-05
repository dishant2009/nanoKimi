"""
Latent attention implementation for nanoKimi.

This module implements the revolutionary latent attention mechanism
that enhances context understanding through latent space projections.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class LatentAttention(nn.Module):
    """
    Latent attention mechanism for enhanced context understanding.
    
    This attention variant projects inputs into a latent space with
    learnable latent tokens, enabling more efficient long-range
    dependencies and better context modeling.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.latent_dim = config.attention.get("latent_dim", 64)
        self.num_latents = config.attention.get("num_latents", 32)
        self.dropout = config.dropout
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        
        # Cross-attention from inputs to latents
        self.to_latent_q = nn.Linear(self.n_embd, self.latent_dim, bias=False)
        self.to_latent_k = nn.Linear(self.n_embd, self.latent_dim, bias=False)
        self.to_latent_v = nn.Linear(self.n_embd, self.latent_dim, bias=False)
        
        # Self-attention within latent space
        self.latent_to_latent = MultiHeadAttention(
            dim=self.latent_dim,
            heads=max(1, self.latent_dim // 64),
            dropout=self.dropout
        )
        
        # Cross-attention from latents back to inputs
        self.from_latent_q = nn.Linear(self.n_embd, self.latent_dim, bias=False)
        self.from_latent_k = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.from_latent_v = nn.Linear(self.latent_dim, self.n_embd, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize latents
        nn.init.normal_(self.latents, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of latent attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            past_key_value: Optional cached key-value pairs
            use_cache: Whether to cache key-value pairs
            
        Returns:
            Tuple of (output, new_past_key_value)
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand latents for batch
        latents = repeat(self.latents, 'n d -> b n d', b=batch_size)
        
        # Phase 1: Cross-attention from inputs to latents
        q_latent = self.to_latent_q(x)  # (B, T, latent_dim)
        k_latent = self.to_latent_k(x)  # (B, T, latent_dim)
        v_latent = self.to_latent_v(x)  # (B, T, latent_dim)
        
        # Attention: latents attend to inputs
        latent_attended = self._cross_attention(
            latents,  # queries
            k_latent,  # keys from input
            v_latent,  # values from input
            attention_mask
        )
        
        # Phase 2: Self-attention within latent space
        latent_processed = self.latent_to_latent(latent_attended)
        
        # Phase 3: Cross-attention from latents back to inputs
        q_output = self.from_latent_q(x)  # (B, T, latent_dim)
        k_output = self.from_latent_k(latent_processed)  # (B, num_latents, latent_dim)
        v_output = self.from_latent_v(latent_processed)  # (B, num_latents, n_embd)
        
        # Attention: inputs attend to processed latents
        output = self._cross_attention(
            q_output,  # queries from input
            k_output,  # keys from latents
            v_output,  # values from latents
            None  # No mask needed for latent attention
        )
        
        # Final projection
        output = self.out_proj(output)
        output = self.dropout_layer(output)
        
        # Handle caching (simplified for latent attention)
        new_past_key_value = None
        if use_cache:
            # Cache the processed latents for next iteration
            new_past_key_value = (latent_processed, latent_processed)
        
        return output, new_past_key_value
    
    def _cross_attention(
        self, 
        queries: torch.Tensor,
        keys: torch.Tensor, 
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cross-attention between queries, keys, and values.
        
        Args:
            queries: Query tensor
            keys: Key tensor  
            values: Value tensor
            attention_mask: Optional attention mask
            
        Returns:
            Attention output
        """
        # Compute attention scores
        scale = math.sqrt(queries.size(-1))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            mask_expanded = attention_mask.unsqueeze(1)  # (B, 1, T)
            mask_expanded = mask_expanded.expand(-1, queries.size(1), -1)  # (B, num_queries, T)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, values)
        
        return output


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention for latent space processing."""
    
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % heads == 0, "dim must be divisible by heads"
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of same shape
        """
        b, n, d = x.shape
        
        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class SparseAttention(nn.Module):
    """
    Sparse attention pattern for efficient long-range dependencies.
    
    Implements local + global sparse attention patterns similar to
    Longformer and BigBird.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.window_size = config.attention.get("window_size", 256)
        self.global_size = config.attention.get("global_size", 32)
        
        # Standard attention components
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with sparse attention patterns.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            past_key_value: Optional cached key-value pairs
            use_cache: Whether to cache key-value pairs
            
        Returns:
            Tuple of (output, new_past_key_value)
        """
        B, T, C = x.size()
        
        # Generate Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) 
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle cached key-value pairs
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        
        new_past_key_value = (k, v) if use_cache else None
        
        # Apply sparse attention pattern
        if T <= self.window_size:
            # Use full attention for short sequences
            y = self._full_attention(q, k, v, attention_mask)
        else:
            # Use sparse attention for long sequences
            y = self._sparse_attention(q, k, v, attention_mask)
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y, new_past_key_value
    
    def _full_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard full attention computation."""
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        
        return y
    
    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sparse attention with local + global patterns.
        
        This is a simplified implementation. In practice, you'd want
        to use more efficient sparse attention kernels.
        """
        B, H, T, D = q.shape
        
        # Create sparse attention mask
        sparse_mask = self._create_sparse_mask(T)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply sparse mask
        att = att.masked_fill(sparse_mask == 0, float('-inf'))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        
        return y
    
    def _create_sparse_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create sparse attention mask with local + global patterns.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Sparse attention mask
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # Local attention (sliding window)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = True
        
        # Global attention (first few tokens attend to all, all attend to first few)
        global_tokens = min(self.global_size, seq_len)
        mask[:global_tokens, :] = True  # Global tokens attend to all
        mask[:, :global_tokens] = True  # All tokens attend to global tokens
        
        return mask.to(device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))


class StandardAttention(nn.Module):
    """
    Standard multi-head attention mechanism.
    
    This is the classic transformer attention as described in
    "Attention Is All You Need" with causal masking for language modeling.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.dropout = config.dropout
        self.block_size = config.block_size
        
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        
        # Combined QKV projection
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.block_size, self.block_size))
            .view(1, 1, self.block_size, self.block_size),
            persistent=False
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of standard attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            past_key_value: Optional cached key-value pairs for generation
            use_cache: Whether to return key-value pairs for caching
            
        Returns:
            Tuple of (output, new_past_key_value)
        """
        B, T, C = x.size()
        
        # Calculate Q, K, V for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle cached key-value pairs for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        
        # Store key-value pairs for next iteration if caching
        present_key_value = (k, v) if use_cache else None
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask
        seq_len = k.size(-2)
        if seq_len <= self.block_size:
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
            att = att.masked_fill(causal_mask == 0, float('-inf'))
        else:
            # For sequences longer than block_size, create mask on the fly
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=att.device))
            att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand mask to match attention scores shape
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        
        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        
        return y, present_key_value