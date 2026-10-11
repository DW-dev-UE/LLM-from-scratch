from __future__ import annotations

import contextlib
import functools
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..act_offload import mark_layer
from ..config import ImplConfig, ModelConfig
from ..precision import precision_context
from . import layers as L
from .attention import Attention, AttnMeta, build_attn_meta
from .loss import LMHead
from .moe import MoE, Router, recompute_guard


@contextlib.contextmanager
def _recompute(precision: str):
    with recompute_guard(), precision_context(precision):
        yield


def _recompute_contexts(precision: str):
    return contextlib.nullcontext(), _recompute(precision)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int, impl: ImplConfig):
        super().__init__()
        self.attn_impl = impl.attn
        self.recompute = impl.recompute == "full"
        self.precision = impl.precision
        self._fwd = torch.compile(self._forward) if impl.compile_blocks else self._forward
        self.is_moe = cfg.is_moe_layer(idx)
        self.attn_norm = L.RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, idx, impl.linear)
        self.ffn_norm = L.RMSNorm(cfg.d_model, cfg.norm_eps)
        if self.is_moe:
            self.ffn = MoE(cfg, impl.experts, impl.linear, impl.moe_align, impl.moe_chunk, impl.precision)
        else:
            self.ffn = L.SwiGLU(cfg.d_model, cfg.dense_ffn, impl.linear)

    def _forward(self, x: torch.Tensor, meta: AttnMeta, rope):
        x = x + self.attn(self.attn_norm(x), meta, rope, self.attn_impl)
        h = self.ffn_norm(x)
        if self.is_moe:
            y, aux = self.ffn(h)
        else:
            y, aux = self.ffn(h), x.new_zeros((), dtype=torch.float32)
        return x + y, aux

    def forward(self, x: torch.Tensor, meta: AttnMeta, rope):
        if self.recompute and self.training and torch.is_grad_enabled():
            return checkpoint(self._fwd, x, meta, rope, use_reentrant=False, context_fn=functools.partial(_recompute_contexts, self.precision))
        return self._fwd(x, meta, rope)


class Apex3(nn.Module):
    def __init__(self, cfg: ModelConfig, impl: ImplConfig | None = None):
        super().__init__()
        impl = impl or ImplConfig()
        self.cfg, self.impl = cfg, impl
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg, i, impl) for i in range(cfg.n_layers))
        self.norm = L.RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = LMHead(cfg.d_model, cfg.vocab_size, cfg.z_loss, impl.ce_chunk, tied=cfg.tie_embeddings, impl=impl.ce_impl)
        self.rope = L.RopeCache(cfg.head_dim)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, position_ids: torch.Tensor) -> dict:
        B, S = input_ids.shape
        meta = build_attn_meta(position_ids)
        ropes = {}
        for key in {layer.attn.rope_key for layer in self.layers}:
            cos, sin = self.rope.get(*key, S, input_ids.device)
            ropes[key] = (cos[meta.position_ids], sin[meta.position_ids])
        x = self.embed(input_ids)
        aux = torch.zeros((), dtype=torch.float32, device=x.device)
        for i, layer in enumerate(self.layers):
            mark_layer(i)
            x, a = layer(x, meta, ropes[layer.attn.rope_key])
            aux = aux + a
        mark_layer(len(self.layers))
        h = self.norm(x).reshape(B * S, -1)
        loss, ce, z = self.head(h, labels.reshape(-1), self.embed.weight if self.cfg.tie_embeddings else None)
        return {"loss": loss + aux, "ce": ce, "z": z, "aux": aux.detach()}

    @torch.no_grad()
    def init_weights(self) -> None:
        std = self.cfg.init_std
        out_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if p.ndim == 1:
                nn.init.ones_(p)
            elif name.endswith(("wo.weight", "w_down.weight", "w_down")) or ".down.weight" in name:
                nn.init.normal_(p, 0.0, out_std)
            else:
                nn.init.normal_(p, 0.0, std)
        for m in self.modules():
            if isinstance(m, Router):
                m.expert_bias.zero_()
                m.load_counts.zero_()
