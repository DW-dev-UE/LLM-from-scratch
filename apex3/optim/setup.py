from __future__ import annotations

import torch
import torch.nn as nn

from .muon import Muon


def split_params(model: nn.Module):
    muon, adam_decay, adam_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or ".router." in name:
            adam_no_decay.append(p)
        elif name.startswith(("embed.", "head.")):
            adam_decay.append(p)
        else:
            muon.append(p)
    return muon, adam_decay, adam_no_decay


class OptimizerSet:
    def __init__(self, model, lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, use_muon=True, muon_momentum_dtype: torch.dtype = torch.float32, muon_check: bool = False):
        muon_p, decay_p, no_decay_p = split_params(model)
        self.optimizers: list[torch.optim.Optimizer] = []
        self.muon: Muon | None = None
        if use_muon:
            self.muon = Muon(muon_p, lr=lr, weight_decay=weight_decay, momentum_dtype=muon_momentum_dtype, check=muon_check)
            self.optimizers.append(self.muon)
        else:
            decay_p = decay_p + muon_p
        groups = [{"params": decay_p, "weight_decay": weight_decay}, {"params": no_decay_p, "weight_decay": 0.0}]
        groups = [g for g in groups if g["params"]]
        self.optimizers.append(torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=True))

    def set_lr(self, lr: float) -> None:
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = lr

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)


def wsd_lr(tokens: float, max_lr: float, warmup_tokens: float, total_tokens: float, decay_frac: float = 0.15, final_ratio: float = 0.0) -> float:
    if tokens < warmup_tokens:
        return max_lr * tokens / warmup_tokens
    decay_start = total_tokens * (1 - decay_frac)
    if tokens < decay_start:
        return max_lr
    frac = min(1.0, (tokens - decay_start) / max(1.0, total_tokens - decay_start))
    return max_lr * (1 - frac * (1 - final_ratio))


def router_bias_rate(tokens: float, total_tokens: float, rate: float, freeze_frac: float = 0.03, late_rate: float = -1.0, late_from: float = 0.0) -> float:
    if tokens >= total_tokens * (1 - freeze_frac):
        return 0.0
    return late_rate if late_rate >= 0 and tokens >= late_from else rate


def batch_tokens(tokens: float, start: float, end: float, ramp_tokens: float) -> float:
    if ramp_tokens <= 0 or tokens >= ramp_tokens:
        return end
    return start + (end - start) * tokens / ramp_tokens
