from __future__ import annotations

import contextlib
import math
import os
import re
import threading
import warnings
import weakref
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from .config import ImplConfig, ModelConfig
from .model import Apex3
from .model.moe import Router
from .optim.muon import orthogonalize_all
from .optim.setup import split_params
from .precision import fp8_weights_context

_TYPESTR = {torch.float32: "<f4", torch.bfloat16: "<i2"}
_NUMPY = {torch.float32: np.float32, torch.bfloat16: np.int16, torch.uint8: np.uint8}
_HOST_REGISTER_MAPPED = 2
_TE_EXPERT = re.compile(r"^(.*\.experts)\.(gate|up|down)\.weight(\d+)$")

_ACCUM_SRC = r"""
#include <torch/extension.h>
#include <cstring>

void bf16_accumulate(torch::Tensor acc, torch::Tensor stage, bool fresh, int64_t threads) {
  TORCH_CHECK(acc.scalar_type() == torch::kFloat32 && stage.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(acc.is_contiguous() && stage.is_contiguous() && acc.numel() == stage.numel());
  float* a = acc.data_ptr<float>();
  const uint16_t* s = reinterpret_cast<const uint16_t*>(stage.data_ptr<at::BFloat16>());
  const int64_t n = acc.numel();
  pybind11::gil_scoped_release no_gil;
  if (fresh) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < n; ++i) {
      uint32_t b = uint32_t(s[i]) << 16; float f; std::memcpy(&f, &b, 4); a[i] = f;
    }
  } else {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < n; ++i) {
      uint32_t b = uint32_t(s[i]) << 16; float f; std::memcpy(&f, &b, 4); a[i] += f;
    }
  }
}
"""
_accum_ext = None


def bf16_accumulate_kernel():
    global _accum_ext
    if _accum_ext is None:
        from torch.utils.cpp_extension import load_inline

        try:
            _accum_ext = load_inline("apex3_bf16_accumulate", cpp_sources=_ACCUM_SRC, functions=["bf16_accumulate"], extra_cflags=["-O3", "-fopenmp", "-mcpu=native"], extra_ldflags=["-fopenmp"])
        except Exception as e:
            warnings.warn(f"CPU accumulation kernel not built ({e}); using torch add_ (slower)")
            _accum_ext = False
    return _accum_ext.bf16_accumulate if _accum_ext else None


class _Mapped:
    def __init__(self, cpu: torch.Tensor):
        self.cpu = cpu
        self.__cuda_array_interface__ = {"shape": (cpu.numel(),), "typestr": _TYPESTR[cpu.dtype], "data": (cpu.data_ptr(), False), "version": 3, "strides": None}


def pinned_empty(numel: int, dtype: torch.dtype) -> torch.Tensor:
    arr = np.empty(max(numel, 1), dtype=_NUMPY[dtype])
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostRegister(arr.ctypes.data, arr.nbytes, _HOST_REGISTER_MAPPED)
    if err != cudart.cudaError.success:
        raise RuntimeError(f"cudaHostRegister of {arr.nbytes / 2**30:.1f} GiB failed: {err}")
    weakref.finalize(arr, cudart.cudaHostUnregister, arr.ctypes.data)
    cpu = torch.from_numpy(arr)
    return cpu.view(dtype) if dtype == torch.bfloat16 else cpu


def host_mapped(numel: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    cpu = pinned_empty(numel, dtype)
    try:
        dev = torch.as_tensor(_Mapped(cpu), device="cuda")
    except RuntimeError:
        warnings.warn("registered memory is not visible to torch; using pin_memory() (rounded up to a power of two)")
        cpu = torch.empty(numel, dtype=dtype, pin_memory=True)
        dev = torch.as_tensor(_Mapped(cpu), device="cuda")
    return cpu, dev.view(dtype) if dtype == torch.bfloat16 else dev


def _construct(cfg: ModelConfig, impl: ImplConfig, weights_ctx) -> Apex3:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"), weights_ctx:
            model = Apex3(cfg, impl)
    finally:
        torch.set_default_dtype(prev)
    for m in model.modules():
        if isinstance(m, Router):
            m.float()
    return model


def build_single_gpu_model(cfg: ModelConfig, impl: ImplConfig, fp8_weights: bool = False) -> tuple[Apex3, dict[str, torch.Tensor] | None]:
    model = _construct(cfg, impl, contextlib.nullcontext())
    model.init_weights()
    if not fp8_weights:
        return model, None
    init = {n: p.detach().cpu() for n, p in model.named_parameters()}
    del model
    torch.cuda.empty_cache()
    return _construct(cfg, impl, fp8_weights_context(impl.precision)), init


def _is_quantized(p: torch.Tensor) -> bool:
    return type(p).__module__.startswith("transformer_engine")


def _layout(params: list[torch.Tensor]) -> tuple[list[int], int]:
    offsets, n = [], 0
    for p in params:
        offsets.append(n)
        n += p.numel()
    return offsets, n


def canonical_views(names: list[str], offsets: list[int], shapes: list[torch.Size], flat: torch.Tensor) -> dict:
    out, groups = {}, defaultdict(list)
    for name, off, shape in zip(names, offsets, shapes):
        m = _TE_EXPERT.match(name)
        if m:
            groups[(m[1], m[2])].append((int(m[3]), off, shape))
        else:
            out[name] = flat[off : off + math.prod(shape)].view(shape)
    for (prefix, kind), items in groups.items():
        items.sort()
        e0, off0, shape = items[0]
        size = math.prod(shape)
        if [e for e, _, _ in items] != list(range(len(items))) or any(off != off0 + e * size or s != shape for e, off, s in items):
            raise RuntimeError(f"{prefix}.{kind}: expert weights are not consecutive in registration order")
        out[f"{prefix}.w_{kind}"] = flat[off0 : off0 + len(items) * size].view(len(items), *shape)
    return out


class OffloadOptimizer:
    def __init__(self, model: nn.Module, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95), eps: float = 1e-8, use_muon: bool = True, muon_momentum_dtype: torch.dtype = torch.float32,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, bucket_numel: int = 1 << 28, init: dict[str, torch.Tensor] | None = None, accumulate: str = "gpu", gpu_share: float = 0.0):
        if accumulate not in ("gpu", "cpu"):
            raise ValueError(f"accumulate: gpu or cpu, not {accumulate}")
        if not 0.0 <= gpu_share <= 1.0:
            raise ValueError(f"gpu_share must be in [0, 1], not {gpu_share}")
        muon_p, decay_p, no_decay_p = split_params(model)
        if not use_muon:
            decay_p, muon_p = decay_p + muon_p, []
        self.muon = None
        self.lr, self.weight_decay, self.betas, self.eps = lr, weight_decay, betas, eps
        self.mu, self.nesterov, self.ns_steps, self.bucket_numel = momentum, nesterov, ns_steps, bucket_numel
        self.adam_step = 0

        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        kind = {id(p): "muon" for p in muon_p}
        kind.update({id(p): "decay" for p in decay_p})
        kind.update({id(p): "no_decay" for p in no_decay_p})
        self.kind = [kind[id(p)] for p in self.params]
        self.muon_idx = [i for i, k in enumerate(self.kind) if k == "muon"]
        self.adam_idx = [i for i, k in enumerate(self.kind) if k != "muon"]
        shapes = [p.shape for p in self.params]

        offsets, n = _layout(self.params)
        self._master_cpu, master = host_mapped(n, torch.float32)
        self._acc_cpu, acc = host_mapped(n, torch.float32)
        self.master = [master[o : o + p.numel()].view(p.shape) for o, p in zip(offsets, self.params)]
        self.acc = [acc[o : o + p.numel()].view(p.shape) for o, p in zip(offsets, self.params)]
        with torch.no_grad():
            if init is None:
                if any(_is_quantized(p) for p in self.params):
                    raise ValueError("the model holds FP8 weights: pass init= from build_single_gpu_model")
                for w, p in zip(self.master, self.params):
                    w.copy_(p)
            else:
                for name, o, p in zip(self.names, offsets, self.params):
                    self._master_cpu[o : o + p.numel()].view(p.shape).copy_(init[name])
                for w, p in zip(self.master, self.params):
                    p.copy_(w)

        m_params = [self.params[i] for i in self.muon_idx]
        m_off, m_n = _layout(m_params)
        self._mom_cpu, mom = host_mapped(max(m_n, 1), muon_momentum_dtype)
        mom.zero_()
        self.mom = {i: mom[o : o + p.numel()].view(p.shape) for i, o, p in zip(self.muon_idx, m_off, m_params)}

        a_params = [self.params[i] for i in self.adam_idx]
        a_off, a_n = _layout(a_params)
        self._m_cpu, m1 = host_mapped(max(a_n, 1), torch.float32)
        self._v_cpu, m2 = host_mapped(max(a_n, 1), torch.float32)
        m1.zero_()
        m2.zero_()
        self.exp_avg = {i: m1[o : o + p.numel()].view(p.shape) for i, o, p in zip(self.adam_idx, a_off, a_params)}
        self.exp_avg_sq = {i: m2[o : o + p.numel()].view(p.shape) for i, o, p in zip(self.adam_idx, a_off, a_params)}

        self._ckpt_model = canonical_views(self.names, offsets, shapes, self._master_cpu)
        self._ckpt_mom = canonical_views([self.names[i] for i in self.muon_idx], m_off, [shapes[i] for i in self.muon_idx], self._mom_cpu)
        a_names, a_shapes = [self.names[i] for i in self.adam_idx], [shapes[i] for i in self.adam_idx]
        self._ckpt_m = canonical_views(a_names, a_off, a_shapes, self._m_cpu)
        self._ckpt_v = canonical_views(a_names, a_off, a_shapes, self._v_cpu)
        torch.cuda.synchronize()

        self.stream = torch.cuda.Stream()
        self.fresh = [True] * len(self.params)
        self.accumulate, self.jobs = accumulate, []
        self.on_cpu = [False] * len(self.params)
        if accumulate == "cpu":
            gpu_left = gpu_share * n
            for i, p in enumerate(self.params):
                self.on_cpu[i] = gpu_left <= 0
                gpu_left -= 0 if self.on_cpu[i] else p.numel()
            cpu_params = [p for p, c in zip(self.params, self.on_cpu) if c]
            stage = {dt: pinned_empty(sum(p.numel() for p in cpu_params if p.dtype == dt), dt) for dt in {p.dtype for p in cpu_params}}
            used = dict.fromkeys(stage, 0)
            self.stage = []
            for p, c in zip(self.params, self.on_cpu):
                if not c:
                    self.stage.append(None)
                    continue
                self.stage.append(stage[p.dtype][used[p.dtype] : used[p.dtype] + p.numel()].view(p.shape))
                used[p.dtype] += p.numel()
            self.acc_cpu = [self._acc_cpu[o : o + p.numel()].view(p.shape) for o, p in zip(offsets, self.params)]
            self.free = [threading.Event() for _ in self.params]
            for e in self.free:
                e.set()
            self.pool = ThreadPoolExecutor(max_workers=1)
            torch.set_num_threads(max(1, (os.cpu_count() or 1) - 2))
            self.kernel, self.kernel_threads = bf16_accumulate_kernel(), max(1, (os.cpu_count() or 1) // 2)
        for i, p in enumerate(self.params):
            p.register_post_accumulate_grad_hook(self._hook(i))

    def _hook(self, i: int):
        def on_grad(p: torch.Tensor) -> None:
            g, p.grad = p.grad, None
            fresh, self.fresh[i] = self.fresh[i], False
            on_cpu = self.on_cpu[i]
            if on_cpu:
                self.free[i].wait()
                self.free[i].clear()
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                if on_cpu:
                    self.stage[i].copy_(g, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record()
                elif fresh:
                    self.acc[i].copy_(g)
                else:
                    self.acc[i].add_(g)
            g.record_stream(self.stream)
            if on_cpu:
                self.jobs.append(self.pool.submit(self._cpu_add, i, done, fresh))

        return on_grad

    def _cpu_add(self, i: int, done: torch.cuda.Event, fresh: bool) -> None:
        done.synchronize()
        if self.kernel is not None and self.stage[i].dtype == torch.bfloat16:
            self.kernel(self.acc_cpu[i], self.stage[i], fresh, self.kernel_threads)
        elif fresh:
            self.acc_cpu[i].copy_(self.stage[i])
        else:
            self.acc_cpu[i].add_(self.stage[i])
        self.free[i].set()

    def _drain(self) -> None:
        torch.cuda.current_stream().wait_stream(self.stream)
        for job in self.jobs:
            job.result()
        self.jobs.clear()

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def zero_grad(self) -> None:
        self.fresh = [True] * len(self.params)

    def _live(self, idx: list[int]) -> list[int]:
        return [i for i in idx if not self.fresh[i]]

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        self._drain()
        accs = [self.acc[i] for i in self._live(list(range(len(self.params))))]
        if not accs:
            return torch.zeros((), device="cuda")
        total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(accs)))
        torch._foreach_mul_(accs, torch.clamp(max_norm / (total + 1e-6), max=1.0))
        return total

    @torch.no_grad()
    def step(self) -> None:
        self._drain()
        bucket, size = [], 0
        for i in self._live(self.muon_idx):
            bucket.append(i)
            size += self.params[i].numel()
            if size >= self.bucket_numel:
                self._muon(bucket)
                bucket, size = [], 0
        if bucket:
            self._muon(bucket)
        live = self._live(self.adam_idx)
        if live:
            self.adam_step += 1
        for i in live:
            self._adam(i)

    def _muon(self, bucket: list[int]) -> None:
        lr, wd, mu = self.lr, self.weight_decay, self.mu
        us = []
        for i in bucket:
            g, buf = self.acc[i], self.mom[i]
            buf.mul_(mu).add_(g)
            us.append(g.add(buf.to(g.dtype), alpha=mu) if self.nesterov else buf)
        outs, _ = orthogonalize_all(us, self.ns_steps, self.bucket_numel)
        for i, o in zip(bucket, outs):
            p, w = self.params[i], self.master[i]
            w.mul_(1 - lr * wd)
            w.add_(o.to(w.dtype), alpha=-lr * 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1])))
            p.copy_(w)

    def _adam(self, i: int) -> None:
        (b1, b2), t = self.betas, self.adam_step
        wd = self.weight_decay if self.kind[i] == "decay" else 0.0
        g, m, v, w = self.acc[i], self.exp_avg[i], self.exp_avg_sq[i], self.master[i]
        w.mul_(1 - self.lr * wd)
        m.lerp_(g, 1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        denom = (v.sqrt() / math.sqrt(1 - b2**t)).add_(self.eps)
        w.addcdiv_(m, denom, value=-self.lr / (1 - b1**t))
        self.params[i].copy_(w)

    def checkpoint_state(self, model: nn.Module) -> dict:
        self._drain()
        torch.cuda.synchronize()
        params = {id(p) for p in self.params}
        named_params = {n for n, p in model.named_parameters() if id(p) in params}
        buffers = {k: v for k, v in model.state_dict(keep_vars=True).items() if k not in named_params and not k.endswith("_extra_state")}
        return {"model": {**self._ckpt_model, **{k: v.data for k, v in buffers.items()}}, "optim": {"muon_momentum": self._ckpt_mom, "adam_exp_avg": self._ckpt_m, "adam_exp_avg_sq": self._ckpt_v, "adam_step": torch.tensor(self.adam_step)}}

    @torch.no_grad()
    def after_load(self, state: dict) -> None:
        self.adam_step = int(state["optim"]["adam_step"])
        self.refresh_params()

    @torch.no_grad()
    def refresh_params(self) -> None:
        for w, p in zip(self.master, self.params):
            p.copy_(w)
        torch.cuda.synchronize()
