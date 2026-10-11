from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).to(x.dtype)


def _swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (F.silu(g.float()) * u.float()).to(g.dtype)


rms_norm_fn = _rms_norm
rope_fn = _rope
swiglu_fn = _swiglu


def enable_compile() -> None:
    global rms_norm_fn, rope_fn, swiglu_fn
    rms_norm_fn = torch.compile(_rms_norm, dynamic=True)
    rope_fn = torch.compile(_rope, dynamic=True)
    swiglu_fn = torch.compile(_swiglu, dynamic=True)


class _WeightedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u, w):
        ctx.save_for_backward(g, u, w)
        return (F.silu(g.float()) * u.float() * w.unsqueeze(-1)).to(g.dtype)

    @staticmethod
    def backward(ctx, dout):
        g, u, w = ctx.saved_tensors
        gf, uf, d = g.float(), u.float(), dout.float()
        sig = torch.sigmoid(gf)
        silu = gf * sig
        dw = (d * silu * uf).sum(-1)
        da = d * w.unsqueeze(-1)
        du = da * silu
        dg = da * uf * sig * (1 + gf * (1 - sig))
        return dg.to(g.dtype), du.to(u.dtype), dw.to(w.dtype)


def weighted_swiglu(g: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _WeightedSwiGLU.apply(g, u, w)


def make_linear(in_features: int, out_features: int, impl: str = "torch") -> nn.Module:
    if impl == "te":
        import transformer_engine.pytorch as te

        return te.Linear(in_features, out_features, bias=False, device=torch.get_default_device())
    lin = nn.Linear(in_features, out_features, bias=False)
    if impl == "fp8_rowwise":
        from . import fp8

        return fp8.to_float8_linear(lin)
    if impl != "torch":
        raise ValueError(f"linear: torch, te or fp8_rowwise, not {impl!r}")
    return lin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_fn(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, linear: str = "torch"):
        super().__init__()
        self.w_gate = make_linear(d_model, hidden, linear)
        self.w_up = make_linear(d_model, hidden, linear)
        self.w_down = make_linear(hidden, d_model, linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(swiglu_fn(self.w_gate(x), self.w_up(x)))


class RopeCache:
    def __init__(self, head_dim: int):
        self.head_dim = head_dim
        self._tables: dict = {}

    def get(self, theta: float, scale: float, length: int, device: torch.device):
        key = (theta, scale, device)
        table = self._tables.get(key)
        if table is None or table[0].size(0) < length:
            exponent = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float64) / self.head_dim
            inv_freq = 1.0 / (theta**exponent)
            pos = torch.arange(length, device=device, dtype=torch.float64) / scale
            angles = torch.outer(pos, inv_freq)
            table = (angles.cos().float(), angles.sin().float())
            self._tables[key] = table
        return table
