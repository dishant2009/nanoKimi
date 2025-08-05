"""
Mixture of Experts (MoE) implementation for nanoKimi.

This module implements a sparse MoE layer with top-k routing,
load balancing, and efficient expert selection mechanisms.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Expert(nn.Module):
    """Individual expert network in MoE layer."""
    
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert network."""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TopKRouter(nn.Module):
    """Top-K router for selecting experts."""
    
    def __init__(self, n_embd: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts.
        
        Args:
            x: Input tensor of shape (batch_size * seq_len, n_embd)
            
        Returns:
            Tuple of (expert_indices, expert_weights, router_logits)
        """
        # Compute router logits
        router_logits = self.gate(x)  # (batch_size * seq_len, num_experts)
        
        # Get top-k experts and their weights
        routing_weights, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return expert_indices, routing_weights, router_logits


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts layer.
    
    This implementation includes:
    - Top-k expert routing
    - Load balancing loss
    - Efficient expert computation
    - Support for training and inference modes
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe["num_experts"]
        self.top_k = config.moe["top_k"]
        self.expert_capacity_factor = config.moe["expert_capacity_factor"]
        self.load_balance_loss_coeff = config.moe["load_balance_loss_coeff"]
        self.n_embd = config.n_embd
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(config.n_embd, config.dropout) 
            for _ in range(self.num_experts)
        ])
        
        # Router for expert selection
        self.router = TopKRouter(config.n_embd, self.num_experts, self.top_k)
        
        # Track auxiliary loss
        self.auxiliary_loss = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Reshape input for routing
        x_flat = x.view(-1, n_embd)  # (batch_size * seq_len, n_embd)
        
        # Route tokens to experts
        expert_indices, routing_weights, router_logits = self.router(x_flat)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process tokens for each expert choice (top-k routing)
        for k in range(self.top_k):
            for expert_idx in range(self.num_experts):
                # Find tokens that route to this expert in position k
                token_mask = (expert_indices[:, k] == expert_idx)
                
                if token_mask.sum() == 0:
                    continue
                
                # Get tokens for this expert
                expert_tokens = x_flat[token_mask]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)
                
                # Get routing weights for these tokens
                weights = routing_weights[token_mask, k:k+1]
                
                # Apply routing weights and accumulate
                weighted_output = expert_output * weights
                output[token_mask] += weighted_output
        
        # Compute auxiliary loss for load balancing
        self.auxiliary_loss = self._compute_load_balance_loss(router_logits)
        
        # Reshape output back to original shape
        output = output.view(batch_size, seq_len, n_embd)
        
        return output
    
    def _compute_load_balance_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert utilization.
        
        Args:
            router_logits: Router output logits
            
        Returns:
            Load balancing loss scalar
        """
        # Compute expert probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Fraction of tokens assigned to each expert
        expert_counts = router_probs.sum(dim=0)
        total_tokens = router_logits.size(0)
        expert_fractions = expert_counts / total_tokens
        
        # Expected fraction (uniform distribution)
        expected_fraction = 1.0 / self.num_experts
        
        # Compute load balance loss (coefficient of variation)
        mean_fraction = expert_fractions.mean()
        variance = ((expert_fractions - mean_fraction) ** 2).mean()
        cv_squared = variance / (mean_fraction ** 2 + 1e-8)
        
        return self.load_balance_loss_coeff * cv_squared

    def expert_utilization_stats(self, router_logits: torch.Tensor) -> dict:
        """
        Compute expert utilization statistics for monitoring.
        
        Args:
            router_logits: Router output logits
            
        Returns:
            Dictionary of utilization statistics
        """
        with torch.no_grad():
            router_probs = F.softmax(router_logits, dim=-1)
            expert_counts = router_probs.sum(dim=0)
            total_tokens = router_logits.size(0)
            
            utilization = expert_counts / total_tokens
            
            return {
                "expert_utilization": utilization.cpu().numpy(),
                "min_utilization": utilization.min().item(),
                "max_utilization": utilization.max().item(),
                "std_utilization": utilization.std().item(),
                "num_active_experts": (expert_counts > 0.01 * total_tokens).sum().item(),
            }


class SwitchMoE(nn.Module):
    """
    Switch Transformer style MoE layer (top-1 routing).
    
    Simpler alternative to sparse MoE with top-1 routing.
    Can be more efficient for some use cases.
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe["num_experts"]
        self.expert_capacity_factor = config.moe.get("expert_capacity_factor", 1.25)
        self.load_balance_loss_coeff = config.moe.get("load_balance_loss_coeff", 0.01)
        self.n_embd = config.n_embd
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(config.n_embd, config.dropout) 
            for _ in range(self.num_experts)
        ])
        
        # Router (top-1)
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
        # Track auxiliary loss
        self.auxiliary_loss = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Switch MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Reshape for processing
        x_flat = x.view(-1, n_embd)
        
        # Compute routing
        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-1 routing
        expert_indices = torch.argmax(router_probs, dim=-1)
        expert_weights = router_probs.gather(1, expert_indices.unsqueeze(1)).squeeze(1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx)
            
            if expert_mask.sum() == 0:
                continue
                
            # Get tokens for this expert
            expert_tokens = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Apply expert weights
            weighted_output = expert_output * expert_weights[expert_mask].unsqueeze(1)
            output[expert_mask] = weighted_output
        
        # Compute load balance loss
        self.auxiliary_loss = self._compute_load_balance_loss(router_probs)
        
        # Reshape back
        output = output.view(batch_size, seq_len, n_embd)
        
        return output
        
    def _compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss for Switch MoE."""
        # Expert assignment frequencies
        expert_counts = router_probs.sum(dim=0)
        total_tokens = router_probs.size(0)
        
        # Target: uniform distribution
        target_count = total_tokens / self.num_experts
        
        # L2 loss between actual and target counts
        loss = ((expert_counts - target_count) ** 2).sum()
        return self.load_balance_loss_coeff * loss / (target_count ** 2)


def create_moe_layer(config, moe_type: str = "sparse"):
    """
    Factory function to create MoE layer based on type.
    
    Args:
        config: Model configuration
        moe_type: Type of MoE ("sparse" or "switch")
        
    Returns:
        MoE layer instance
    """
    if moe_type == "sparse":
        return MixtureOfExperts(config)
    elif moe_type == "switch":
        return SwitchMoE(config)
    else:
        raise ValueError(f"Unknown MoE type: {moe_type}")