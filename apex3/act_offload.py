from __future__ import annotations

import contextlib
from collections import defaultdict

import torch
import torch.nn as nn

_ACTIVE: "ActivationOffload | None" = None


def mark_layer(i: int) -> None:
    if _ACTIVE is not None:
        _ACTIVE.layer = i


class _Saved:
    __slots__ = ("cpu", "shape", "dtype", "layer", "gpu", "event", "used")

    def __init__(self, cpu: torch.Tensor, shape: torch.Size, dtype: torch.dtype, layer: int):
        self.cpu, self.shape, self.dtype, self.layer = cpu, shape, dtype, layer
        self.gpu, self.event, self.used = None, None, False


class ActivationOffload:
    def __init__(self, model: nn.Module, capacity_bytes: int, min_bytes: int = 1 << 20, align: int = 512, select=None, ahead: int = 2, max_pending_bytes: int = 4 << 30):
        from .offload import pinned_empty

        self.arena = pinned_empty(capacity_bytes, torch.uint8)
        self.capacity, self.min_bytes, self.align = capacity_bytes, min_bytes, align
        self.select, self.ahead = select, ahead
        self.pending: list[tuple[torch.cuda.Event, torch.Tensor]] = []
        self.pending_bytes, self.max_pending_bytes = 0, max_pending_bytes
        self.param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self.d2h, self.h2d = torch.cuda.Stream(), torch.cuda.Stream()
        self.offset, self.layer = 0, -1
        self.by_layer: dict[int, list[_Saved]] = defaultdict(list)
        self.fetched: set[int] = set()
        self.stats = {"offloaded_bytes": 0, "offloaded_tensors": 0, "kept_bytes": 0, "overflow_bytes": 0}

    @contextlib.contextmanager
    def forward(self):
        global _ACTIVE
        cur = torch.cuda.current_stream()
        self.d2h.wait_stream(cur)
        self.d2h.wait_stream(self.h2d)
        self.offset, self.layer = 0, -1
        self.by_layer.clear()
        self.fetched.clear()
        self.stats = dict.fromkeys(self.stats, 0)
        self._release()
        _ACTIVE = self
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield self
        finally:
            _ACTIVE = None

    def _pack(self, t: torch.Tensor):
        n = t.numel() * t.element_size()
        if (not t.is_cuda or n < self.min_bytes or t.untyped_storage().data_ptr() in self.param_storages or (self.select is not None and not self.select(t))):
            self.stats["kept_bytes"] += n if t.is_cuda else 0
            return t
        start = -(-self.offset // self.align) * self.align
        if start + n > self.capacity:
            self.stats["overflow_bytes"] += n
            return t
        self.offset = start + n
        src = t if t.is_contiguous() else t.contiguous()
        cpu = self.arena[start : start + n].view(t.dtype).view(t.shape)
        self.d2h.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.d2h):
            cpu.copy_(src, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        self.pending.append((done, src))
        self.pending_bytes += n
        self._release(wait_above=self.max_pending_bytes)
        saved = _Saved(cpu, t.shape, t.dtype, self.layer)
        self.by_layer[self.layer].append(saved)
        self.stats["offloaded_bytes"] += n
        self.stats["offloaded_tensors"] += 1
        return saved

    def _release(self, wait_above: int | None = None) -> None:
        while self.pending:
            done, src = self.pending[0]
            if not done.query():
                if wait_above is None or self.pending_bytes <= wait_above:
                    break
                done.synchronize()
            self.pending.pop(0)
            self.pending_bytes -= src.numel() * src.element_size()

    def _copy_back(self, s: _Saved) -> None:
        cur = torch.cuda.current_stream()
        s.gpu = torch.empty(s.shape, dtype=s.dtype, device="cuda")
        self.h2d.wait_stream(self.d2h)
        self.h2d.wait_stream(cur)
        with torch.cuda.stream(self.h2d):
            s.gpu.copy_(s.cpu, non_blocking=True)
            s.event = torch.cuda.Event()
            s.event.record()

    def _fetch(self, layer: int) -> None:
        if layer in self.fetched or layer not in self.by_layer:
            return
        self.fetched.add(layer)
        for s in self.by_layer[layer]:
            if s.gpu is None and not s.used:
                self._copy_back(s)

    def _unpack(self, saved):
        if isinstance(saved, torch.Tensor):
            return saved
        if saved.gpu is None:
            self._copy_back(saved)
        saved.used = True
        torch.cuda.current_stream().wait_event(saved.event)
        out, saved.gpu = saved.gpu, None
        self._release()
        for ahead in range(saved.layer - 1, saved.layer - 1 - self.ahead, -1):
            self._fetch(ahead)
        return out
