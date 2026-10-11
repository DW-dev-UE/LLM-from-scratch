from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from . import layers as L


@dataclass
class AttnMeta:
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch: int
    seq: int
    cu_seqlens_k: torch.Tensor | None = None


def build_attn_meta(position_ids: torch.Tensor) -> AttnMeta:
    B, S = position_ids.shape
    pos = position_ids.reshape(-1)
    starts = torch.nonzero(pos == 0, as_tuple=False).squeeze(-1)
    cu = torch.cat([starts, starts.new_tensor([pos.numel()])]).to(torch.int32)
    return AttnMeta(pos, cu, S, B, S, cu.clone())


def _flash(q, k, v, meta: AttnMeta, window: int, impl: str) -> torch.Tensor:
    if impl == "fa3":
        from flash_attn_interface import flash_attn_varlen_func
    else:
        from flash_attn import flash_attn_varlen_func
    window_size = (window - 1, 0) if window > 0 else (-1, -1)
    cu_k = meta.cu_seqlens if meta.cu_seqlens_k is None else meta.cu_seqlens_k
    out = flash_attn_varlen_func(q, k, v, meta.cu_seqlens, cu_k, meta.max_seqlen, meta.max_seqlen, causal=True, window_size=window_size)
    return out[0] if isinstance(out, tuple) else out


def _sdpa(q, k, v, meta: AttnMeta, window: int) -> torch.Tensor:
    B, S = meta.batch, meta.seq
    H, D, Hk = q.size(1), q.size(2), k.size(1)
    q = q.view(B, S, H, D).transpose(1, 2)
    k = k.view(B, S, Hk, D).transpose(1, 2).repeat_interleave(H // Hk, dim=1)
    v = v.view(B, S, Hk, D).transpose(1, 2).repeat_interleave(H // Hk, dim=1)
    doc = torch.cumsum(meta.position_ids.view(B, S) == 0, dim=1)
    i = torch.arange(S, device=q.device)
    mask = (doc[:, :, None] == doc[:, None, :]) & (i[:, None] >= i[None, :])
    if window > 0:
        mask = mask & (i[:, None] - i[None, :] < window)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
    return out.transpose(1, 2).reshape(B * S, H, D)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, linear: str = "torch"):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.is_global = cfg.is_global(layer_idx)
        self.window = -1 if self.is_global else cfg.window
        self.theta = cfg.rope_theta_global if self.is_global else cfg.rope_theta_local
        self.rope_scale = cfg.rope_scale_global if self.is_global else 1.0
        d, q_dim, kv_dim = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
        self.wq = L.make_linear(d, q_dim, linear)
        self.wk = L.make_linear(d, kv_dim, linear)
        self.wv = L.make_linear(d, kv_dim, linear)
        self.wo = L.make_linear(q_dim, d, linear)
        self.q_norm = L.RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = L.RMSNorm(cfg.head_dim, cfg.norm_eps)

    @property
    def rope_key(self) -> tuple[float, float]:
        return (self.theta, self.rope_scale)

    def forward(self, x: torch.Tensor, meta: AttnMeta, rope: tuple[torch.Tensor, torch.Tensor], impl: str):
        T = meta.batch * meta.seq
        q = self.q_norm(self.wq(x).view(T, self.n_heads, self.head_dim))
        k = self.k_norm(self.wk(x).view(T, self.n_kv, self.head_dim))
        v = self.wv(x).view(T, self.n_kv, self.head_dim)
        cos, sin = rope
        q, k = L.rope_fn(q, cos, sin), L.rope_fn(k, cos, sin)
        if impl == "sdpa":
            o = _sdpa(q, k, v, meta, self.window)
        else:
            o = _flash(q, k, v, meta, self.window, impl)
        return self.wo(o.reshape(meta.batch, meta.seq, self.n_heads * self.head_dim))
