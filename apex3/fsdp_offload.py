from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from .config import ImplConfig, ModelConfig
from .model import Apex3
from .model.moe import Router
from .offload import pinned_empty
from .optim.muon import orthogonalize_all
from .optim.setup import split_params
from .parallel import apply_fsdp, build_mesh


def build_fsdp_offload_model(cfg: ModelConfig, impl: ImplConfig, world: int, param_dtype=torch.bfloat16, reduce_dtype: torch.dtype | None = None) -> Apex3:
    if reduce_dtype is None:
        reduce_dtype = torch.bfloat16 if world == 2 else torch.float32
    prev = torch.get_default_dtype()
    torch.set_default_dtype(param_dtype)
    try:
        with torch.device("meta"):
            model = Apex3(cfg, impl)
    finally:
        torch.set_default_dtype(prev)
    for m in model.modules():
        if isinstance(m, Router):
            m.float()
    apply_fsdp(model, build_mesh(world, shard_size=world), reduce_dtype=reduce_dtype)
    model.to_empty(device="cuda")
    model.init_weights()
    return model


def _views(flat: torch.Tensor, offsets: list[int], shapes: list[torch.Size]) -> list[torch.Tensor]:
    return [flat[o : o + math.prod(s)].view(s) for o, s in zip(offsets, shapes)]


class FSDPOffloadOptimizer:
    def __init__(self, model: nn.Module, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95), eps: float = 1e-8, use_muon: bool = True, muon_momentum_dtype: torch.dtype = torch.float32, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, bucket_numel: int = 1 << 27, cpu_threads: int | None = None):
        muon_p, decay_p, no_decay_p = split_params(model)
        if not use_muon:
            decay_p, muon_p = decay_p + muon_p, []
        self.muon = None
        self.lr, self.weight_decay, self.betas, self.eps = lr, weight_decay, betas, eps
        self.mu, self.nesterov, self.ns_steps, self.bucket_numel = momentum, nesterov, ns_steps, bucket_numel
        self.adam_step, self.clip_coef = 0, 1.0
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        torch.set_num_threads(cpu_threads or max(1, (os.cpu_count() or 1) // local_world))

        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        for n, p in named:
            if not isinstance(p, DTensor) or p.device_mesh.ndim != 1:
                raise ValueError(f"{n}: needs FSDP2 parameters on a 1D mesh (one node)")
        named.sort(key=lambda item: item[1].dtype != torch.bfloat16)
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        kind = {id(p): "muon" for p in muon_p}
        kind.update({id(p): "decay" for p in decay_p})
        kind.update({id(p): "no_decay" for p in no_decay_p})
        self.kind = [kind[id(p)] for p in self.params]
        self.muon_idx = [i for i, k in enumerate(self.kind) if k == "muon"]
        self.adam_idx = [i for i, k in enumerate(self.kind) if k != "muon"]

        local = [p.to_local() for p in self.params]
        shapes = [t.shape for t in local]
        offsets, self.n = self._layout(local)
        self.n16 = sum(t.numel() for t in local if t.dtype == torch.bfloat16)
        self._master, self._acc = pinned_empty(self.n, torch.float32), pinned_empty(self.n, torch.float32)
        self._stage16 = pinned_empty(self.n16, torch.bfloat16)
        self._stage32 = pinned_empty(self.n - self.n16, torch.float32)
        self.master, self.acc = _views(self._master, offsets, shapes), _views(self._acc, offsets, shapes)
        self.stage = []
        for o, t in zip(offsets, local):
            flat, base = (self._stage16, 0) if t.dtype == torch.bfloat16 else (self._stage32, self.n16)
            self.stage.append(flat[o - base : o - base + t.numel()].view(t.shape))
        for w, t in zip(self.master, local):
            w.copy_(t)

        m_off, m_n = self._layout([local[i] for i in self.muon_idx])
        self._mom = pinned_empty(m_n, muon_momentum_dtype)
        self._mom.zero_()
        self.mom = dict(zip(self.muon_idx, _views(self._mom, m_off, [shapes[i] for i in self.muon_idx])))
        a_off, a_n = self._layout([local[i] for i in self.adam_idx])
        self._m, self._v = pinned_empty(a_n, torch.float32), pinned_empty(a_n, torch.float32)
        self._m.zero_()
        self._v.zero_()
        a_shapes = [shapes[i] for i in self.adam_idx]
        self.exp_avg = dict(zip(self.adam_idx, _views(self._m, a_off, a_shapes)))
        self.exp_avg_sq = dict(zip(self.adam_idx, _views(self._v, a_off, a_shapes)))

        self.stream = torch.cuda.Stream()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.job = None
        self.fresh = True
        self.got = [False] * len(self.params)
        self.process_group = dist.new_group(backend="gloo")
        self.cpu_mesh = DeviceMesh.from_group(self.process_group, "cpu")
        torch.cuda.synchronize()

    @staticmethod
    def _layout(tensors: list[torch.Tensor]) -> tuple[list[int], int]:
        offsets, n = [], 0
        for t in tensors:
            offsets.append(n)
            n += t.numel()
        return offsets, n

    def _wait(self) -> None:
        if self.job is not None:
            self.job.result()
            self.job = None

    def after_backward(self) -> None:
        self._wait()
        torch.cuda.synchronize()
        missing = set()
        with torch.cuda.stream(self.stream):
            for i, p in enumerate(self.params):
                if p.grad is None:
                    missing.add(i)
                    continue
                g = p.grad.to_local()
                self.stage[i].copy_(g, non_blocking=True)
                g.record_stream(self.stream)
            done = torch.cuda.Event()
            done.record()
        for i, p in enumerate(self.params):
            p.grad = None
            self.got[i] = self.got[i] or i not in missing
        fresh, self.fresh = self.fresh, False
        self.job = self.pool.submit(self._accumulate, done, fresh, missing)

    def _accumulate(self, done: torch.cuda.Event, fresh: bool, missing: set[int]) -> None:
        done.synchronize()
        for i in missing:
            self.stage[i].zero_()
        for acc, stage in ((self._acc[: self.n16], self._stage16[: self.n16]), (self._acc[self.n16 : self.n], self._stage32[: self.n - self.n16])):
            if not acc.numel():
                continue
            if fresh:
                acc.copy_(stage)
            else:
                acc.add_(stage)

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def zero_grad(self) -> None:
        self.fresh = True
        self.got = [False] * len(self.params)

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        self._wait()
        if all(self.got):
            sq = torch.linalg.vector_norm(self._acc[: self.n]).double() ** 2
        else:
            sq = torch.zeros((), dtype=torch.float64)
            for i in range(len(self.params)):
                if self.got[i]:
                    sq += torch.linalg.vector_norm(self.acc[i]).double() ** 2
        total = sq.to("cuda")
        dist.all_reduce(total)
        total = total.sqrt().float()
        self.clip_coef = min(1.0, max_norm / (total.item() + 1e-6))
        return total

    @torch.no_grad()
    def step(self) -> None:
        self._wait()
        bucket, size = [], 0
        for i in self.muon_idx:
            if not self.got[i]:
                continue
            bucket.append(i)
            size += self.params[i].numel()
            if size >= self.bucket_numel:
                self._muon(bucket)
                bucket, size = [], 0
        if bucket:
            self._muon(bucket)
        live = [i for i in self.adam_idx if self.got[i]]
        if live:
            self.adam_step += 1
        for i in live:
            self._adam(i)
        torch.cuda.synchronize()
        self.clip_coef = 1.0

    def _grad(self, i: int) -> torch.Tensor:
        g = self.acc[i].to("cuda", non_blocking=True)
        return g.mul_(self.clip_coef) if self.clip_coef != 1.0 else g

    def _muon(self, bucket: list[int]) -> None:
        lr, wd, mu = self.lr, self.weight_decay, self.mu
        us, bufs = [], []
        for i in bucket:
            p, g = self.params[i], self._grad(i)
            buf = self.mom[i].to("cuda", non_blocking=True)
            buf.mul_(mu).add_(g)
            u = g.add(buf.to(g.dtype), alpha=mu) if self.nesterov else buf
            us.append(DTensor.from_local(u, p.device_mesh, p.placements, run_check=False, shape=p.shape, stride=p.stride()))
            bufs.append(buf)
        outs, _ = orthogonalize_all(us, self.ns_steps, self.bucket_numel)
        for i, o, buf in zip(bucket, outs, bufs):
            p = self.params[i]
            w = self.master[i].to("cuda", non_blocking=True)
            w.mul_(1 - lr * wd)
            w.add_(o.to_local().to(w.dtype), alpha=-lr * 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1])))
            self.master[i].copy_(w, non_blocking=True)
            self.mom[i].copy_(buf, non_blocking=True)
            p.to_local().copy_(w)

    def _adam(self, i: int) -> None:
        (b1, b2), t = self.betas, self.adam_step
        wd = self.weight_decay if self.kind[i] == "decay" else 0.0
        g = self._grad(i)
        m = self.exp_avg[i].to("cuda", non_blocking=True)
        v = self.exp_avg_sq[i].to("cuda", non_blocking=True)
        w = self.master[i].to("cuda", non_blocking=True)
        w.mul_(1 - self.lr * wd)
        m.lerp_(g, 1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        denom = (v.sqrt() / math.sqrt(1 - b2**t)).add_(self.eps)
        w.addcdiv_(m, denom, value=-self.lr / (1 - b1**t))
        for cpu, gpu in ((self.exp_avg[i], m), (self.exp_avg_sq[i], v), (self.master[i], w)):
            cpu.copy_(gpu, non_blocking=True)
        self.params[i].to_local().copy_(w)

    def _dtensor(self, i: int, shard: torch.Tensor) -> DTensor:
        p = self.params[i]
        return DTensor.from_local(shard, self.cpu_mesh, p.placements, run_check=False, shape=p.shape, stride=p.stride())

    def checkpoint_state(self, model: nn.Module) -> dict:
        self._wait()
        torch.cuda.synchronize()
        params = set(self.names)
        buffers = {k: v for k, v in model.state_dict().items() if k not in params and not k.endswith("_extra_state")}
        return {"model": {**{n: self._dtensor(i, self.master[i]) for i, n in enumerate(self.names)}, **buffers},
                "optim": {"muon_momentum": {self.names[i]: self._dtensor(i, m) for i, m in self.mom.items()}, "adam_exp_avg": {self.names[i]: self._dtensor(i, m) for i, m in self.exp_avg.items()}, "adam_exp_avg_sq": {self.names[i]: self._dtensor(i, v) for i, v in self.exp_avg_sq.items()}, "adam_step": torch.tensor(self.adam_step)}}

    @torch.no_grad()
    def after_load(self, state: dict) -> None:
        self.adam_step = int(state["optim"]["adam_step"])
        for w, p in zip(self.master, self.params):
            p.to_local().copy_(w.to("cuda", non_blocking=True))
        torch.cuda.synchronize()
