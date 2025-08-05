"""
Core transformer implementation with Kimi-K2 innovations.

This module implements the main KimiModel class with integrated MoE layers,
latent attention, and support for the Muon optimizer.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import LatentAttention
from .moe import MixtureOfExperts
from .embedding import TokenEmbedding, PositionalEmbedding


@dataclass
class KimiConfig:
    """Configuration class for Kimi-K2 model."""
    
    # Model architecture
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    dropout: float = 0.1
    
    # Mixture of Experts configuration
    moe: Dict[str, Any] = field(default_factory=lambda: {
        "num_experts": 8,
        "top_k": 2,
        "expert_capacity_factor": 1.25,
        "load_balance_loss_coeff": 0.01,
        "use_moe": True,
        "moe_layers": [3, 7, 11],  # Which layers to use MoE in
    })
    
    # Latent attention configuration  
    attention: Dict[str, Any] = field(default_factory=lambda: {
        "type": "latent",
        "latent_dim": 64,
        "num_latents": 32,
        "use_flash_attention": True,
    })
    
    # Training configuration
    training: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "learning_rate": 3e-4,
        "optimizer": "muon",
        "warmup_steps": 1000,
        "max_steps": 100000,
        "grad_clip": 1.0,
        "weight_decay": 0.1,
    })

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.moe["top_k"] <= self.moe["num_experts"], "top_k must be <= num_experts"
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "KimiConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "dropout": self.dropout,
            "moe": self.moe,
            "attention": self.attention,
            "training": self.training,
        }


class LayerNorm(nn.Module):
    """Layer normalization with bias."""
    
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    """Standard MLP block for non-MoE layers."""
    
    def __init__(self, config: KimiConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class KimiBlock(nn.Module):
    """
    A single transformer block with optional MoE and latent attention.
    """
    
    def __init__(self, config: KimiConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Layer normalization
        self.ln_1 = LayerNorm(config.n_embd)
        self.ln_2 = LayerNorm(config.n_embd)
        
        # Attention mechanism
        if config.attention["type"] == "latent":
            self.attn = LatentAttention(config)
        elif config.attention.get("use_flash_attention", False):
            from .flash_attention import get_attention_implementation
            self.attn = get_attention_implementation(
                n_embd=config.n_embd,
                n_head=config.n_head,
                dropout=config.dropout,
                causal=True,
                block_size=config.block_size,
                attention_type="auto",
                config=config
            )
        else:
            # Fallback to standard multi-head attention
            self.attn = StandardAttention(config)
        
        # MLP or MoE layer
        use_moe = (config.moe["use_moe"] and 
                  layer_idx in config.moe.get("moe_layers", []))
        
        if use_moe:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = MLP(config)
            
        self.use_moe = use_moe

    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            past_key_value: Optional cached key-value pairs for fast generation
            use_cache: Whether to cache key-value pairs
            
        Returns:
            Tuple of (output, new_past_key_value, auxiliary_loss)
        """
        # Attention with residual connection
        attn_output, new_past_key_value = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_output
        
        # MLP/MoE with residual connection
        mlp_output = self.mlp(self.ln_2(x))
        
        # Handle MoE auxiliary loss
        auxiliary_loss = None
        if self.use_moe and hasattr(self.mlp, 'auxiliary_loss'):
            auxiliary_loss = self.mlp.auxiliary_loss
            
        if isinstance(mlp_output, tuple):
            mlp_output, auxiliary_loss = mlp_output
            
        x = x + mlp_output
        
        return x, new_past_key_value, auxiliary_loss


class StandardAttention(nn.Module):
    """Standard multi-head attention fallback."""
    
    def __init__(self, config: KimiConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Handle cached key-value pairs
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        
        new_past_key_value = (k, v) if use_cache else None
        
        # Attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        seq_len = k.size(-2)
        if seq_len <= self.causal_mask.size(-1):
            att = att.masked_fill(
                self.causal_mask[:, :, :T, :seq_len] == 0, float('-inf')
            )
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs
        
        # Output projection
        y = self.c_proj(y)
        
        return y, new_past_key_value


class KimiModel(nn.Module):
    """
    Main Kimi-K2 model implementation.
    
    Features:
    - Mixture of Experts (MoE) layers
    - Latent attention mechanism
    - Standard transformer backbone
    - Support for autoregressive generation
    """
    
    def __init__(self, config: KimiConfig):
        super().__init__()
        self.config = config
        
        # Token and positional embeddings
        self.wte = TokenEmbedding(config.vocab_size, config.n_embd)
        self.wpe = PositionalEmbedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            KimiBlock(config, layer_idx) for layer_idx in range(config.n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights (optional optimization)
        self.lm_head.weight = self.wte.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            past_key_values: Optional cached key-value pairs
            use_cache: Whether to cache key-value pairs for generation
            labels: Optional labels for computing loss
            
        Returns:
            Dictionary containing logits, loss, and auxiliary information
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        if past_key_values is not None and len(past_key_values) > 0:
            # Adjust position indices for cached generation
            pos = pos + past_key_values[0][0].size(-2)
            
        tok_emb = self.wte(input_ids)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)       # (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        new_past_key_values = []
        total_auxiliary_loss = 0.0
        
        for i, block in enumerate(self.h):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            x, new_past_key_value, auxiliary_loss = block(
                x,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                new_past_key_values.append(new_past_key_value)
                
            if auxiliary_loss is not None:
                total_auxiliary_loss += auxiliary_loss
        
        # Final layer norm and language modeling head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add auxiliary loss from MoE layers
            if total_auxiliary_loss > 0:
                loss = loss + total_auxiliary_loss
        
        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_past_key_values if use_cache else None,
            "auxiliary_loss": total_auxiliary_loss,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting token sequence
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter  
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token sequence
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_ids, use_cache=False)
                logits = outputs["logits"]
                
                # Get logits for the last token
                logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Apply top-p filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set logits to -inf for removed tokens
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check if we've hit the maximum sequence length
                if input_ids.size(1) >= self.config.block_size:
                    break
        
        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params