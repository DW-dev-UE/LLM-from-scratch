from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from ..precision import precision_context
from . import layers as L

_RECOMPUTING = False


@contextlib.contextmanager
def recompute_guard():
    global _RECOMPUTING
    prev, _RECOMPUTING = _RECOMPUTING, True
    try:
        yield
    finally:
        _RECOMPUTING = prev


class Router(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts, self.top_k = cfg.n_experts, cfg.top_k
        self.route_scale, self.aux_alpha = cfg.route_scale, cfg.seq_aux_alpha
        self.weight = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model))
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))
        self.register_buffer("load_counts", torch.zeros(cfg.n_experts), persistent=False)

    def forward(self, h: torch.Tensor, batch: int, seq: int):
        scores = torch.sigmoid(F.linear(h.float(), self.weight.float()))
        idx = torch.topk(scores + self.expert_bias, self.top_k, dim=-1, sorted=False).indices
        gate = scores.gather(1, idx)
        gate = gate / (gate.sum(-1, keepdim=True) + 1e-20) * self.route_scale
        selected = torch.zeros_like(scores).scatter_(1, idx, 1.0)
        if self.training and not _RECOMPUTING:
            with torch.no_grad():
                self.load_counts += selected.sum(0)
        aux = scores.new_zeros(())
        if self.training and self.aux_alpha > 0:
            f = selected.view(batch, seq, -1).sum(1) * (self.n_experts / (self.top_k * seq))
            p = (scores / scores.sum(-1, keepdim=True)).view(batch, seq, -1).mean(1)
            aux = self.aux_alpha * (f * p).sum(-1).mean()
        return idx, gate, aux


GROUPED_IMPLS = ("grouped_mm", "fp8_grouped")


class GroupedExperts(nn.Module):
    def __init__(self, n_experts: int, d_model: int, hidden: int, impl: str):
        super().__init__()
        if impl not in GROUPED_IMPLS + ("loop",):
            raise ValueError(f"experts: grouped_mm, fp8_grouped or loop, not {impl!r}")
        if impl == "fp8_grouped" and (d_model % 16 or hidden % 16):
            raise ValueError(f"fp8_grouped needs d_model and expert_ffn divisible by 16: {d_model}, {hidden}")
        self.impl = impl
        self.w_gate = nn.Parameter(torch.empty(n_experts, hidden, d_model))
        self.w_up = nn.Parameter(torch.empty(n_experts, hidden, d_model))
        self.w_down = nn.Parameter(torch.empty(n_experts, d_model, hidden))

    def forward(self, x: torch.Tensor, counts: torch.Tensor, slot_w: torch.Tensor) -> torch.Tensor:
        if self.impl in GROUPED_IMPLS:
            offs = torch.cumsum(counts, 0, dtype=torch.int32)
            if self.impl == "fp8_grouped":
                from . import fp8

                def mm(a, w):
                    return fp8.grouped_mm(a, w.to(a.dtype).transpose(-2, -1), offs)
            else:
                def mm(a, w):
                    return torch._grouped_mm(a, w.to(a.dtype).transpose(-2, -1), offs=offs)

            return mm(L.weighted_swiglu(mm(x, self.w_gate), mm(x, self.w_up), slot_w), self.w_down)
        outs, start = [], 0
        for e, n in enumerate(counts.tolist()):
            xe, we = x[start : start + n], slot_w[start : start + n]
            a = L.weighted_swiglu(xe @ self.w_gate[e].to(x.dtype).t(), xe @ self.w_up[e].to(x.dtype).t(), we)
            outs.append(a @ self.w_down[e].to(x.dtype).t())
            start += n
        return torch.cat(outs)


class TEExperts(nn.Module):
    def __init__(self, n_experts: int, d_model: int, hidden: int):
        super().__init__()
        import transformer_engine.pytorch as te

        dev = torch.get_default_device()
        self.gate = te.GroupedLinear(n_experts, d_model, hidden, bias=False, device=dev)
        self.up = te.GroupedLinear(n_experts, d_model, hidden, bias=False, device=dev)
        self.down = te.GroupedLinear(n_experts, hidden, d_model, bias=False, device=dev)

    def forward(self, x: torch.Tensor, counts: torch.Tensor, slot_w: torch.Tensor) -> torch.Tensor:
        m = counts.tolist()
        return self.down(L.weighted_swiglu(self.gate(x, m), self.up(x, m), slot_w), m)


class TEStackedExperts(nn.Module):
    def __init__(self, n_experts: int, d_model: int, hidden: int):
        super().__init__()
        import transformer_engine.pytorch as te

        dev = torch.get_default_device()
        self.w_gate = nn.Parameter(torch.empty(n_experts, hidden, d_model))
        self.w_up = nn.Parameter(torch.empty(n_experts, hidden, d_model))
        self.w_down = nn.Parameter(torch.empty(n_experts, d_model, hidden))
        self.gate = te.GroupedLinear(n_experts, d_model, hidden, bias=False, device=dev)
        self.up = te.GroupedLinear(n_experts, d_model, hidden, bias=False, device=dev)
        self.down = te.GroupedLinear(n_experts, hidden, d_model, bias=False, device=dev)
        for mod in (self.gate, self.up, self.down):
            for e in range(n_experts):
                del mod._parameters[f"weight{e}"]

    def _bind(self) -> None:
        for mod, w in ((self.gate, self.w_gate), (self.up, self.w_up), (self.down, self.w_down)):
            for e, v in enumerate(w.unbind(0)):
                mod.__dict__[f"weight{e}"] = v

    def forward(self, x: torch.Tensor, counts: torch.Tensor, slot_w: torch.Tensor) -> torch.Tensor:
        m = counts.tolist()
        self._bind()
        return self.down(L.weighted_swiglu(self.gate(x, m), self.up(x, m), slot_w), m)


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig, experts: str = "grouped_mm", linear: str = "torch", align: int = 16, chunk: int = 0, precision: str = "bf16"):
        super().__init__()
        self.n_experts, self.align, self.chunk, self.precision = cfg.n_experts, align, chunk, precision
        self.router = Router(cfg)
        if experts == "te":
            self.experts = TEExperts(cfg.n_experts, cfg.d_model, cfg.expert_ffn)
        elif experts == "te_stacked":
            self.experts = TEStackedExperts(cfg.n_experts, cfg.d_model, cfg.expert_ffn)
        else:
            self.experts = GroupedExperts(cfg.n_experts, cfg.d_model, cfg.expert_ffn, experts)
        self.shared = L.SwiGLU(cfg.d_model, cfg.shared_ffn, linear) if cfg.shared_ffn > 0 else None

    def forward(self, x: torch.Tensor):
        B, S, d = x.shape
        h = x.reshape(-1, d)
        idx, gate, aux = self.router(h, B, S)
        T = h.size(0)
        if self.chunk and T > self.chunk:
            grad = torch.is_grad_enabled()
            parts = []
            for s in range(0, T, self.chunk):
                args = (h[s : s + self.chunk], idx[s : s + self.chunk], gate[s : s + self.chunk])
                parts.append(checkpoint(self._routed, *args, use_reentrant=False, context_fn=self._contexts) if grad else self._routed(*args))
            y = torch.cat(parts)
        else:
            y = self._routed(h, idx, gate)
        if self.shared is not None:
            y = y + self.shared(h)
        return y.view(B, S, d), aux

    def _contexts(self):
        return contextlib.nullcontext(), precision_context(self.precision)

    def _routed(self, h: torch.Tensor, idx: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        T, K = idx.shape
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        e_sorted = flat[order]
        counts = torch.zeros(self.n_experts, dtype=torch.long, device=h.device).scatter_add_(0, flat, torch.ones_like(flat))
        padded = (counts + self.align - 1) // self.align * self.align
        dest = (torch.cumsum(padded, 0) - padded)[e_sorted] + (torch.arange(T * K, device=h.device) - (torch.cumsum(counts, 0) - counts)[e_sorted])
        tok = order // K
        static = isinstance(self.experts, GroupedExperts) and self.experts.impl in GROUPED_IMPLS
        M = -(-(T * K + self.n_experts * (self.align - 1)) // self.align) * self.align if static else int(padded.sum())
        xs = h.new_zeros(M, h.size(1)).index_copy(0, dest, h.index_select(0, tok))
        ws = gate.new_zeros(M).index_copy(0, dest, gate.reshape(-1).index_select(0, order))
        ys = self.experts(xs, padded, ws)
        row = torch.empty_like(dest).index_copy_(0, order, dest)
        return ys.index_select(0, row).view(T, K, -1).float().sum(1).to(h.dtype)


_LAST_HOT: dict[int, int] = {}


@torch.no_grad()
def update_router_biases(model: nn.Module, rate: float, mode: str = "sign"):
    routers = [m for m in model.modules() if isinstance(m, Router)]
    if not routers:
        return {}, None
    counts = torch.stack([r.load_counts for r in routers])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
    mean = counts.mean(-1, keepdim=True)
    if rate > 0:
        if mode == "sign":
            delta = rate * torch.sign(mean - counts)
        elif mode == "proportional":
            delta = rate * ((mean - counts) / mean.clamp(min=1)).clamp(-1, 1)
        else:
            raise ValueError(f"router_bias_update: sign or proportional, not {mode}")
        for r, dl in zip(routers, delta):
            r.expert_bias.add_(dl)
    for r in routers:
        r.load_counts.zero_()
    ratio = counts.max(-1).values / mean.squeeze(-1).clamp(min=1)
    bias = torch.stack([r.expert_bias for r in routers])
    hot = counts.argmax(-1).tolist()
    switched = [h != _LAST_HOT[id(r)] for h, r in zip(hot, routers) if id(r) in _LAST_HOT]
    for h, r in zip(hot, routers):
        _LAST_HOT[id(r)] = h
    stats = {"moe/load_max_over_mean": ratio.max().item(), "moe/load_max_over_mean_avg": ratio.mean().item(), "moe/zero_load_experts": int((counts == 0).sum().item()),
             "moe/bias_range": (bias.max(-1).values - bias.min(-1).values).max().item(), "moe/load_cv_avg": (counts.std(-1) / mean.squeeze(-1).clamp(min=1)).mean().item(), "moe/hot_switch": sum(switched) / len(switched) if switched else 0.0}
    return stats, counts.cpu()


def load_report(model: nn.Module, counts: torch.Tensor) -> str:
    routers = [m for m in model.modules() if isinstance(m, Router)]
    mean = counts.mean(-1).clamp(min=1)
    ratio, hot = (counts.max(-1).values / mean).tolist(), counts.argmax(-1).tolist()
    return " ".join(f"L{i}:{q:.2f}@{h}({r.expert_bias[h].item():+.2f})" for i, (q, h, r) in enumerate(zip(ratio, hot, routers)))
