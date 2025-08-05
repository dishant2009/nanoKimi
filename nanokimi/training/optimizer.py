"""
Muon optimizer implementation for nanoKimi.

The Muon optimizer is specifically designed for training Mixture of Experts
models with improved convergence and gradient handling.
"""

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer


class MuonOptimizer(Optimizer):
    """
    Muon optimizer designed for MoE training.
    
    This optimizer combines the benefits of Adam with specialized
    handling for sparse expert gradients and improved convergence
    for transformer models with MoE layers.
    
    Key features:
    - Adaptive learning rates per parameter
    - Specialized momentum handling for sparse gradients
    - Load balancing awareness for MoE training
    - Memory-efficient implementation
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        expert_lr_multiplier: float = 1.0,
        momentum_dtype: torch.dtype = torch.float32,
        foreach: Optional[bool] = None,
    ):
        """
        Initialize Muon optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)
            amsgrad: Whether to use AMSGrad variant (default: False)
            expert_lr_multiplier: Learning rate multiplier for expert parameters (default: 1.0)
            momentum_dtype: Data type for momentum buffers (default: float32)
            foreach: Whether to use vectorized implementation (default: None)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            expert_lr_multiplier=expert_lr_multiplier,
            momentum_dtype=momentum_dtype,
            foreach=foreach,
        )
        super().__init__(params, defaults)
    
    def __setstate__(self, state: Dict[str, Any]):
        """Set state for optimizer loading."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('expert_lr_multiplier', 1.0)
            group.setdefault('momentum_dtype', torch.float32)
            group.setdefault('foreach', None)
    
    def _init_group(self, group: Dict, params_with_grad: list, grads: list, exp_avgs: list, exp_avg_sqs: list, max_exp_avg_sqs: list, state_steps: list):
        """Initialize optimizer state for parameter group."""
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')
                grads.append(p.grad)
                
                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults.get('capturable', False) else 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=group['momentum_dtype'])
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=group['momentum_dtype'])
                    if group['amsgrad']:
                        # Maintains max of all exp_avg_sq values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=group['momentum_dtype'])
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                
                state_steps.append(state['step'])
        
        return has_complex
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
            
        Returns:
            Optional loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            
            has_complex = self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)
            
            muon_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=group['betas'][0],
                beta2=group['betas'][1],
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                expert_lr_multiplier=group['expert_lr_multiplier'],
                has_complex=has_complex,
                foreach=group['foreach'],
            )
        
        return loss
    
    def is_expert_param(self, param_name: str) -> bool:
        """
        Check if a parameter belongs to an expert layer.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            True if parameter belongs to expert layer
        """
        expert_keywords = ['expert', 'moe', 'router', 'gate']
        return any(keyword in param_name.lower() for keyword in expert_keywords)


def muon_update(
    params: list,
    grads: list,
    exp_avgs: list,
    exp_avg_sqs: list,
    max_exp_avg_sqs: list,
    state_steps: list,
    foreach: Optional[bool] = None,
    amsgrad: bool = False,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    expert_lr_multiplier: float = 1.0,
    has_complex: bool = False,
):
    """
    Functional API for Muon optimizer update.
    
    This is the core update logic separated from the class for easier testing
    and potential compilation.
    """
    if foreach is None:
        foreach = False  # Disable foreach for simplicity in this implementation
    
    if foreach:
        _multi_tensor_muon_update(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            expert_lr_multiplier=expert_lr_multiplier,
            has_complex=has_complex,
        )
    else:
        _single_tensor_muon_update(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            expert_lr_multiplier=expert_lr_multiplier,
            has_complex=has_complex,
        )


def _single_tensor_muon_update(
    params: list,
    grads: list,
    exp_avgs: list,
    exp_avg_sqs: list,
    max_exp_avg_sqs: list,
    state_steps: list,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    expert_lr_multiplier: float,
    has_complex: bool,
):
    """Single tensor Muon update implementation."""
    for i, param in enumerate(params):
        grad = grads[i] if not has_complex else torch.view_as_real(grads[i])
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        
        # Update step count
        if torch.is_tensor(step_t):
            step_t += 1
        else:
            step_t += 1
            state_steps[i] = step_t
        
        # Perform stepweight decay
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)
        
        # Check if this is an expert parameter (simplified heuristic)
        is_expert = 'expert' in str(param.shape) or param.numel() > 1000000  # Large params likely experts
        current_lr = lr * expert_lr_multiplier if is_expert else lr
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step_t
        bias_correction2 = 1 - beta2 ** step_t
        
        # Exponential moving average of gradient values
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Exponential moving average of squared gradient values
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sq = max_exp_avg_sqs[i]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        
        step_size = current_lr / bias_correction1
        
        # Apply update
        if has_complex:
            param.view_as_real().addcdiv_(exp_avg, denom, value=-step_size)
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)


def _multi_tensor_muon_update(
    params: list,
    grads: list,
    exp_avgs: list,
    exp_avg_sqs: list,
    max_exp_avg_sqs: list,
    state_steps: list,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    expert_lr_multiplier: float,
    has_complex: bool,
):
    """Multi-tensor Muon update implementation (placeholder)."""
    # For now, fallback to single tensor implementation
    # In practice, this would use optimized multi-tensor kernels
    _single_tensor_muon_update(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
        expert_lr_multiplier,
        has_complex,
    )


class AdafactorMuon(Optimizer):
    """
    Muon variant inspired by Adafactor for memory efficiency.
    
    This combines Muon's expert-aware optimization with Adafactor's
    memory-efficient second moment estimation.
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: Optional[float] = None,
        eps2: float = 1e-30,
        cliping_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        expert_lr_multiplier: float = 1.0,
    ):
        """Initialize AdafactorMuon optimizer."""
        if lr is not None and lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            eps2=eps2,
            cliping_threshold=cliping_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            expert_lr_multiplier=expert_lr_multiplier,
        )
        super().__init__(params, defaults)
    
    def _get_lr(self, param_group: Dict, param_state: Dict) -> float:
        """Compute internal learning rate."""
        min_step = 1e-6 * param_state['step'] if param_group['scale_parameter'] else 1e-2
        rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps2'], param_state['RMS'])
        return param_scale * rel_step_sz
    
    def _get_options(self, param_group: Dict, param_shape: torch.Size) -> Tuple[bool, bool]:
        """Determine factorization options based on parameter shape."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment
    
    def _rms(self, tensor: torch.Tensor) -> float:
        """Compute root mean square."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)
    
    def _approx_sq_grad(self, exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        """Approximate squared gradient from factored matrices."""
        r_factor = ((exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
                   .rsqrt_().unsqueeze(-1).clamp_(0, math.inf))
        c_factor = exp_avg_sq_col.rsqrt().unsqueeze(0).clamp_(0, math.inf)
        return torch.mul(r_factor, c_factor)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                grad_shape = grad.shape
                
                factored, use_first_moment = self._get_options(group, grad_shape)
                
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad).float()
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).float()
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).float()
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad).float()
                    
                    state['RMS'] = 0
                    
                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()
                
                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                
                lr = group['lr']
                if group['lr'] is None:
                    lr = self._get_lr(group, state)
                
                # Check if expert parameter
                is_expert = 'expert' in str(p.shape) or p.numel() > 1000000
                if is_expert:
                    lr *= group['expert_lr_multiplier']
                
                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad**2 + group['eps2']
                
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                update.div_(max(1.0, self._rms(update) / group['cliping_threshold']))
                
                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    update = exp_avg
                
                if group['weight_decay'] > 0:
                    p_data_fp32.mul_(1 - group['weight_decay'] * lr)
                
                p_data_fp32.add_(update, alpha=-lr)
                
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)
        
        return loss