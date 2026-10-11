from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

from ..config import ImplConfig, ModelConfig
from ..model import Apex3
from ..model import layers as L
from ..model.moe import Router


class Backend(Protocol):
    def generate(self, ids: list[int], stop: set[int], max_new_tokens: int) -> list[int]:
        ...


def load_model(cfg: ModelConfig, ckpt: str, device: str = "cuda", dtype=torch.bfloat16, experts: str = "grouped_mm") -> Apex3:
    import torch.distributed.checkpoint as dcp

    with torch.device(device):
        model = Apex3(cfg, ImplConfig(attn="sdpa", experts=experts))
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=ckpt)
    model.load_state_dict(state["model"])
    for m in model.modules():
        if not isinstance(m, Router):
            for p in m.parameters(recurse=False):
                p.data = p.data.to(dtype)
    return model.eval()


class TorchBackend:
    def __init__(self, model: Apex3, max_len: int = 131_072, temperature: float = 0.7, top_p: float = 0.95, seed: int = 0, prefill_chunk: int = 1024):
        self.model, self.cfg = model, model.cfg
        self.max_len, self.temperature, self.top_p, self.chunk = max_len, temperature, top_p, prefill_chunk
        w = model.embed.weight
        self.device, self.dtype = w.device, w.dtype
        c = self.cfg
        shape = (c.n_layers, max_len, c.n_kv_heads, c.head_dim)
        self.k = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.v = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.ropes = {key: model.rope.get(*key, max_len, self.device) for key in {b.attn.rope_key for b in model.layers}}
        self.cached: list[int] = []
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.capture_layers: tuple[int, ...] = ()
        self.layer_states: torch.Tensor | None = None
        self.turn_states: torch.Tensor | None = None

    @torch.no_grad()
    def hidden(self, ids: list[int], start: int) -> torch.Tensor:
        m, n = self.model, len(ids)
        end = start + n
        pos = torch.arange(start, end, device=self.device)
        masks: dict[int, tuple[int, torch.Tensor]] = {}
        caps = []
        x = m.embed(torch.tensor(ids, device=self.device))
        for li, block in enumerate(m.layers):
            a = block.attn
            h = block.attn_norm(x)
            q = a.q_norm(a.wq(h).view(n, a.n_heads, a.head_dim))
            k = a.k_norm(a.wk(h).view(n, a.n_kv, a.head_dim))
            v = a.wv(h).view(n, a.n_kv, a.head_dim)
            cos, sin = self.ropes[a.rope_key]
            q, k = L.rope_fn(q, cos[pos], sin[pos]), L.rope_fn(k, cos[pos], sin[pos])
            self.k[li, start:end], self.v[li, start:end] = k, v
            if a.window not in masks:
                lo = 0 if a.window < 0 else max(0, start - a.window + 1)
                kpos = torch.arange(lo, end, device=self.device)
                mask = kpos[None, :] <= pos[:, None]
                if a.window > 0:
                    mask &= pos[:, None] - kpos[None, :] < a.window
                masks[a.window] = (lo, mask)
            lo, mask = masks[a.window]
            rep = a.n_heads // a.n_kv
            kc = self.k[li, lo:end].repeat_interleave(rep, dim=1).transpose(0, 1)
            vc = self.v[li, lo:end].repeat_interleave(rep, dim=1).transpose(0, 1)
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kc[None], vc[None], attn_mask=mask)
            x = x + a.wo(o[0].transpose(0, 1).reshape(n, a.n_heads * a.head_dim))
            h = block.ffn_norm(x)
            x = x + (block.ffn(h[None])[0][0] if block.is_moe else block.ffn(h))
            if li in self.capture_layers:
                caps.append(x)
        self.layer_states = torch.stack(caps, 1) if caps else None
        return m.norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        m = self.model
        return m.head.logits(h, m.embed.weight if self.cfg.tie_embeddings else None).float()

    def sample(self, logits: torch.Tensor) -> int:
        if self.temperature <= 0:
            return int(logits.argmax())
        probs = torch.softmax(logits / self.temperature, dim=-1)
        if self.top_p < 1:
            p, idx = probs.sort(descending=True)
            p = p * (p.cumsum(0) - p < self.top_p)
            return int(idx[torch.multinomial(p, 1, generator=self.gen)])
        return int(torch.multinomial(probs, 1, generator=self.gen))

    @torch.no_grad()
    def generate(self, ids: list[int], stop: set[int], max_new_tokens: int) -> list[int]:
        if len(ids) + max_new_tokens > self.max_len:
            raise ValueError(f"{len(ids)} + {max_new_tokens} tokens exceed the cache ({self.max_len})")
        p = 0
        for a, b in zip(self.cached, ids):
            if a != b:
                break
            p += 1
        p = min(p, len(ids) - 1)
        self.cached = self.cached[:p]
        for s in range(p, len(ids), self.chunk):
            h = self.hidden(ids[s : s + self.chunk], s)
        turn = [self.layer_states[-1:]] if self.capture_layers else []
        self.cached = list(ids)
        out: list[int] = []
        for _ in range(max_new_tokens):
            t = self.sample(self.logits(h[-1]))
            out.append(t)
            if t in stop or len(out) == max_new_tokens:
                break
            h = self.hidden([t], len(self.cached))
            self.cached.append(t)
            if self.capture_layers:
                turn.append(self.layer_states)
        self.turn_states = torch.cat(turn) if turn else None
        return out
