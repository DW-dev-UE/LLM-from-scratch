from __future__ import annotations

import warnings

import torch
import torch.nn as nn

IGNORE_INDEX = -100
_TRITON = []


def _triton_ops():
    if not _TRITON:
        try:
            from . import ce_triton
        except Exception as e:
            warnings.warn(f"Triton CE unavailable ({e}); using the torch path")
            ce_triton = None
        _TRITON.append(ce_triton)
    return _TRITON[0]


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, labels, z_coef, chunk, impl):
        valid = labels != IGNORE_INDEX
        n_valid = valid.sum().clamp(min=1).float()
        grad_h = torch.empty_like(h)
        grad_w = torch.zeros(w.shape, dtype=torch.float32, device=w.device) if ctx.needs_input_grad[1] else None
        ce_sum = torch.zeros((), dtype=torch.float32, device=h.device)
        z_sum = torch.zeros_like(ce_sum)
        ops = _triton_ops() if impl == "triton" and h.is_cuda else None
        if ops is not None:
            buf = h.new_empty(min(chunk, h.size(0)), w.size(0))
            g = None
            for s in range(0, h.size(0), chunk):
                hc, vc = h[s : s + chunk], valid[s : s + chunk].float()
                g = torch.mm(hc, w.t(), out=buf[: hc.size(0)])
                lse, target = ops.ce_grad_(g, labels[s : s + chunk], n_valid, z_coef, IGNORE_INDEX)
                ce_sum += ((lse - target) * vc).sum()
                z_sum += (lse.square() * vc).sum()
                torch.mm(g, w, out=grad_h[s : s + chunk])
                if grad_w is not None:
                    ops.accumulate_wgrad(grad_w, g.t(), hc)
            del buf, g
            return _FusedLinearCE._finish(ctx, w, grad_h, grad_w, ce_sum, z_sum, n_valid, z_coef)
        for s in range(0, h.size(0), chunk):
            hc, vc = h[s : s + chunk], valid[s : s + chunk].float()
            y = labels[s : s + chunk].clamp(min=0).unsqueeze(1)
            logits = (hc @ w.t()).float()
            lse = torch.logsumexp(logits, dim=-1)
            ce_sum += ((lse - logits.gather(1, y).squeeze(1)) * vc).sum()
            z_sum += (lse.square() * vc).sum()
            g = logits.sub_(lse.unsqueeze(1)).exp_()
            g.mul_((1 + 2 * z_coef * lse).unsqueeze(1))
            g.scatter_add_(1, y, torch.full_like(y, -1, dtype=g.dtype))
            g.mul_((vc / n_valid).unsqueeze(1))
            g = g.to(h.dtype)
            grad_h[s : s + chunk] = g @ w
            if grad_w is not None:
                grad_w += (g.t() @ hc).float()
        return _FusedLinearCE._finish(ctx, w, grad_h, grad_w, ce_sum, z_sum, n_valid, z_coef)

    @staticmethod
    def _finish(ctx, w, grad_h, grad_w, ce_sum, z_sum, n_valid, z_coef):
        ctx.save_for_backward(grad_h, grad_w.to(w.dtype) if grad_w is not None else None)
        ce, z = ce_sum / n_valid, z_sum / n_valid
        ctx.mark_non_differentiable(ce, z)
        return ce + z_coef * z, ce, z

    @staticmethod
    def backward(ctx, d_loss, _d_ce, _d_z):
        grad_h, grad_w = ctx.saved_tensors
        gh = grad_h * d_loss.to(grad_h.dtype)
        gw = grad_w * d_loss.to(grad_w.dtype) if grad_w is not None else None
        return gh, gw, None, None, None, None


@torch.no_grad()
def _ce_no_grad(h, w, labels, z_coef, chunk):
    valid = labels != IGNORE_INDEX
    n_valid = valid.sum().clamp(min=1).float()
    ce_sum = torch.zeros((), dtype=torch.float32, device=h.device)
    z_sum = torch.zeros_like(ce_sum)
    for s in range(0, h.size(0), chunk):
        vc = valid[s : s + chunk].float()
        y = labels[s : s + chunk].clamp(min=0).unsqueeze(1)
        logits = (h[s : s + chunk] @ w.t()).float()
        lse = torch.logsumexp(logits, dim=-1)
        ce_sum += ((lse - logits.gather(1, y).squeeze(1)) * vc).sum()
        z_sum += (lse.square() * vc).sum()
    ce, z = ce_sum / n_valid, z_sum / n_valid
    return ce + z_coef * z, ce, z


def fused_linear_cross_entropy(h, w, labels, z_coef: float, chunk: int = 1024, impl: str = "torch"):
    if torch.is_grad_enabled() and (h.requires_grad or w.requires_grad):
        return _FusedLinearCE.apply(h, w, labels, z_coef, chunk, impl)
    return _ce_no_grad(h, w, labels, z_coef, chunk)


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, z_coef: float, chunk: int, tied: bool = False, impl: str = "torch"):
        super().__init__()
        self.z_coef, self.chunk, self.impl = z_coef, chunk, impl
        self.weight = None if tied else nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, h: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None = None):
        w = self.weight if weight is None else weight
        return fused_linear_cross_entropy(h, w, labels, self.z_coef, self.chunk, self.impl)

    def logits(self, h: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        return h @ (self.weight if weight is None else weight).t()
