"""
Flash Attention integration for nanoKimi.

This module provides efficient attention implementations using Flash Attention
for improved memory usage and speed during training and inference.
"""

import math
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import flash attention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    warnings.warn(
        "Flash Attention not available. Install with: pip install flash-attn --no-build-isolation",
        ImportWarning
    )

# Try to import xFormers for alternative efficient attention
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class FlashAttention(nn.Module):
    """
    Flash Attention implementation with fallbacks.
    
    This module provides efficient attention computation using Flash Attention
    when available, with fallbacks to other efficient implementations.
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        causal: bool = True,
        use_flash: bool = True,
        block_size: int = 1024,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.causal = causal
        self.use_flash = use_flash and FLASH_ATTENTION_AVAILABLE
        self.block_size = block_size
        
        # Query, Key, Value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        
        # Dropout layer for non-flash attention
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Flash attention parameters
        self.flash_config = {
            'dropout_p': dropout if self.training else 0.0,
            'softmax_scale': None,  # Will use 1/sqrt(head_dim)
            'causal': causal,
        }
        
        # Register causal mask for fallback
        if causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(block_size, block_size))
                .view(1, 1, block_size, block_size),
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
        Forward pass with Flash Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            past_key_value: Optional cached key-value pairs
            use_cache: Whether to cache key-value pairs
            
        Returns:
            Tuple of (output, new_past_key_value)
        """
        B, T, C = x.size()
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        # Handle cached key-value pairs
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Concatenate past and current
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
            # Update sequence length
            T = k.size(1)
        
        new_past_key_value = (k, v) if use_cache else None
        
        # Choose attention implementation
        if self.use_flash and self._can_use_flash(q, k, v, attention_mask):
            y = self._flash_attention(q, k, v, attention_mask)
        elif XFORMERS_AVAILABLE and self._can_use_xformers(q, k, v, attention_mask):
            y = self._xformers_attention(q, k, v, attention_mask)
        else:
            y = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape output
        y = y.view(B, T, C)
        
        # Apply output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        
        return y, new_past_key_value
    
    def _can_use_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> bool:
        """Check if Flash Attention can be used."""
        if not FLASH_ATTENTION_AVAILABLE:
            return False
        
        # Flash attention requirements
        device_ok = q.device.type == 'cuda'
        dtype_ok = q.dtype in [torch.float16, torch.bfloat16]
        
        # Flash attention doesn't support arbitrary attention masks
        mask_ok = attention_mask is None or self._is_causal_mask(attention_mask)
        
        # Head dimension should be supported (usually <= 128)
        head_dim_ok = self.head_dim <= 128
        
        return device_ok and dtype_ok and mask_ok and head_dim_ok
    
    def _can_use_xformers(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> bool:
        """Check if xFormers attention can be used."""
        if not XFORMERS_AVAILABLE:
            return False
        
        # xFormers is more flexible with dtypes and masks
        device_ok = q.device.type in ['cuda', 'cpu']
        
        return device_ok
    
    def _is_causal_mask(self, attention_mask: torch.Tensor) -> bool:
        """Check if attention mask is causal (lower triangular)."""
        if attention_mask is None:
            return True
        
        # Simple heuristic: check if mask has causal pattern
        seq_len = attention_mask.size(-1)
        if seq_len <= 1:
            return True
        
        # Check if it's lower triangular
        expected_causal = torch.tril(torch.ones_like(attention_mask[-1, -1]))
        return torch.allclose(attention_mask[-1, -1], expected_causal)
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention using Flash Attention."""
        # Flash attention expects (batch_size, seq_len, n_heads, head_dim)
        # Our tensors are already in this format
        
        # Update dropout probability based on training mode
        dropout_p = self.dropout if self.training else 0.0
        
        if attention_mask is not None and not self._is_causal_mask(attention_mask):
            # For non-causal masks, use variable length attention
            return self._flash_attention_varlen(q, k, v, attention_mask, dropout_p)
        else:
            # Standard flash attention for causal case
            return flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=self.causal,
            )
    
    def _flash_attention_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        dropout_p: float,
    ) -> torch.Tensor:
        """Flash attention with variable length sequences."""
        batch_size, seq_len = q.shape[:2]
        
        # Convert attention mask to cumulative sequence lengths
        # This is a simplified version - real implementation would be more complex
        if attention_mask.dim() == 2:
            # Batch attention mask: (batch_size, seq_len)
            attention_lengths = attention_mask.sum(dim=1)
        else:
            # Assume full length sequences for now
            attention_lengths = torch.full((batch_size,), seq_len, device=q.device)
        
        # Create cumulative sequence lengths
        cu_seqlens = torch.cat([
            torch.zeros(1, device=q.device, dtype=torch.int32),
            attention_lengths.cumsum(dim=0, dtype=torch.int32)
        ])
        
        max_seqlen = attention_lengths.max().item()
        
        # Flatten sequences
        q_flat = q.reshape(-1, self.n_head, self.head_dim)
        k_flat = k.reshape(-1, self.n_head, self.head_dim)
        v_flat = v.reshape(-1, self.n_head, self.head_dim)
        
        # Apply flash attention
        output_flat = flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=self.causal,
        )
        
        # Reshape back
        output = output_flat.reshape(batch_size, seq_len, self.n_head, self.head_dim)
        
        return output
    
    def _xformers_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention using xFormers."""
        # Reshape for xFormers: (batch_size, seq_len, n_heads, head_dim)
        # Our tensors are already in this format
        
        # Create attention bias from mask
        attn_bias = None
        if attention_mask is not None:
            # Convert mask to bias
            attn_bias = torch.zeros_like(attention_mask, dtype=q.dtype)
            attn_bias.masked_fill_(attention_mask == 0, float('-inf'))
        elif self.causal:
            # Create causal mask
            seq_len = q.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
            attn_bias = torch.zeros(seq_len, seq_len, device=q.device, dtype=q.dtype)
            attn_bias.masked_fill_(causal_mask == 0, float('-inf'))
        
        # Apply xFormers memory efficient attention
        output = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        
        return output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fallback to standard attention implementation."""
        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask
        if self.causal:
            seq_len = q.size(-2)
            if seq_len <= self.causal_mask.size(-1):
                causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
                att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand to match attention scores shape
                mask_expanded = attention_mask.unsqueeze(1).unsqueeze(1)
                att = att.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v
        
        # Reshape back to (batch_size, seq_len, n_heads, head_dim)
        y = y.transpose(1, 2)
        
        return y


class FlashMHA(nn.Module):
    """
    Multi-Head Attention with Flash Attention optimizations.
    
    This is a drop-in replacement for standard multi-head attention
    that automatically uses the most efficient available implementation.
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        causal: bool = True,
        block_size: int = 1024,
        use_flash: bool = True,
        use_rope: bool = False,
        rope_base: int = 10000,
    ):
        super().__init__()
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope
        
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        # Core attention mechanism
        self.attention = FlashAttention(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            causal=causal,
            use_flash=use_flash,
            block_size=block_size,
        )
        
        # Rotary Position Embedding (optional)
        if use_rope:
            from .embedding import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_seq_len=block_size,
                base=rope_base,
            )
        else:
            self.rope = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Flash Attention and optional RoPE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            past_key_value: Optional cached key-value pairs
            use_cache: Whether to cache key-value pairs
            
        Returns:
            Tuple of (output, new_past_key_value)
        """
        if self.use_rope:
            return self._forward_with_rope(x, attention_mask, past_key_value, use_cache)
        else:
            return self.attention(x, attention_mask, past_key_value, use_cache)
    
    def _forward_with_rope(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with Rotary Position Embedding."""
        B, T, C = x.size()
        
        # Get Q, K, V from attention layer
        qkv = self.attention.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        # Apply RoPE to queries and keys
        seq_len = k.size(1)
        if past_key_value is not None:
            seq_len += past_key_value[0].size(1)
        
        q, k = self.rope(q, k, seq_len)
        
        # Handle cached key-value pairs
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
        
        new_past_key_value = (k, v) if use_cache else None
        
        # Use Flash Attention implementation
        if self.attention._can_use_flash(q, k, v, attention_mask):
            y = self.attention._flash_attention(q, k, v, attention_mask)
        elif self.attention._can_use_xformers(q, k, v, attention_mask):
            y = self.attention._xformers_attention(q, k, v, attention_mask)
        else:
            y = self.attention._standard_attention(q, k, v, attention_mask)
        
        # Reshape and apply output projection
        y = y.view(B, T, C)
        y = self.attention.resid_dropout(self.attention.c_proj(y))
        
        return y, new_past_key_value


def get_attention_implementation(
    n_embd: int,
    n_head: int,
    dropout: float = 0.0,
    causal: bool = True,
    block_size: int = 1024,
    attention_type: str = "auto",
    **kwargs
) -> nn.Module:
    """
    Factory function to get the best available attention implementation.
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        dropout: Dropout probability
        causal: Whether to use causal attention
        block_size: Maximum sequence length
        attention_type: Type of attention ("auto", "flash", "standard", "latent")
        **kwargs: Additional arguments
        
    Returns:
        Attention module instance
    """
    if attention_type == "auto":
        # Automatically choose best implementation
        if FLASH_ATTENTION_AVAILABLE:
            attention_type = "flash"
        elif XFORMERS_AVAILABLE:
            attention_type = "xformers"
        else:
            attention_type = "standard"
    
    if attention_type == "flash":
        return FlashMHA(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            causal=causal,
            block_size=block_size,
            use_flash=True,
            **kwargs
        )
    elif attention_type == "xformers":
        return FlashMHA(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            causal=causal,
            block_size=block_size,
            use_flash=False,  # Will use xFormers
            **kwargs
        )
    elif attention_type == "latent":
        from .attention import LatentAttention
        return LatentAttention(kwargs.get('config'))
    else:
        # Standard attention
        from .attention import StandardAttention
        return StandardAttention(kwargs.get('config'))


def is_flash_attention_available() -> bool:
    """Check if Flash Attention is available."""
    return FLASH_ATTENTION_AVAILABLE


def is_xformers_available() -> bool:
    """Check if xFormers is available."""
    return XFORMERS_AVAILABLE


def get_attention_info() -> dict:
    """Get information about available attention implementations."""
    return {
        'flash_attention': FLASH_ATTENTION_AVAILABLE,
        'xformers': XFORMERS_AVAILABLE,
        'recommended': 'flash' if FLASH_ATTENTION_AVAILABLE else 'xformers' if XFORMERS_AVAILABLE else 'standard',
    }