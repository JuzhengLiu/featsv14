import torch
from torch.optim.optimizer import Optimizer
import math

class CAdamW(Optimizer):
    """
    Implements Cautious AdamW algorithm.
    Paper: Cautious Optimizers: Improving Training with One Line of Code
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 1. Weight Decay (Decoupled like AdamW)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                # 2. Update Momentum (First and Second moments)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # --- 3. Cautious Modification (The "One Line") ---
                # 仅在动量与梯度方向一致时更新
                # Mask: (exp_avg * grad > 0)
                mask = (exp_avg * grad > 0).float()
                
                # Scaling: 保持总更新能量守恒 (Scale by 1/mask_mean)
                mask_ratio = mask.mean()
                # 避免除以0
                if mask_ratio > 1e-6:
                    mask.div_(mask_ratio)
                
                # Apply update
                # p = p - lr * (exp_avg / denom) * mask
                p.add_((exp_avg / denom) * mask, alpha=-step_size)

        return loss