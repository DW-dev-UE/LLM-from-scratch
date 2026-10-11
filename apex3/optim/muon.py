from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if transposed else X


def _local_slice(full: torch.Tensor, ref: DTensor) -> torch.Tensor:
    local = full
    for mesh_dim, placement in enumerate(ref.placements):
        if isinstance(placement, Shard):
            n = ref.device_mesh.size(mesh_dim)
            r = ref.device_mesh.get_local_rank(mesh_dim)
            chunks = torch.chunk(local, n, dim=placement.dim)
            local = chunks[r] if r < len(chunks) else local.narrow(placement.dim, 0, 0)
    return local.contiguous()


def orthogonalize(u: torch.Tensor, steps: int) -> torch.Tensor:
    if not isinstance(u, DTensor):
        return newton_schulz5(u, steps)
    mesh, placements = u.device_mesh, u.placements
    if u.ndim == 3 and all(p.is_replicate() or p == Shard(0) for p in placements):
        local = newton_schulz5(u.to_local(), steps)
    else:
        local = _local_slice(newton_schulz5(u.full_tensor(), steps), u)
    assert local.shape == u.to_local().shape, (local.shape, u.to_local().shape)
    return DTensor.from_local(local, mesh, placements, run_check=False, shape=u.shape, stride=u.stride())


def row_sizes(m: int, world: int) -> list[int]:
    c = -(-m // world)
    return [max(0, min(c, m - r * c)) for r in range(world)]


def owners(n: int, world: int, start: int = 0) -> list[int]:
    return [(start + j) % world for j in range(n)]


def pack_to_owners(pieces: list[torch.Tensor], own: list[int], world: int) -> tuple[torch.Tensor, list[int]]:
    by_dst = [[p.reshape(-1) for p, o in zip(pieces, own) if o == d] for d in range(world)]
    flat = [t for ts in by_dst for t in ts]
    return torch.cat(flat), [sum(t.numel() for t in ts) for ts in by_dst]


def unpack_at_owner(recv: torch.Tensor, k: int, rows: list[int], n: int) -> torch.Tensor:
    parts = torch.split(recv, [k * r * n for r in rows])
    return torch.cat([p.view(k, r, n) for p, r in zip(parts, rows)], dim=1)


def pack_from_owner(out: torch.Tensor, rows: list[int]) -> tuple[torch.Tensor, list[int]]:
    k, _, n = out.shape
    parts = torch.split(out, rows, dim=1)
    return torch.cat([p.reshape(-1) for p in parts]), [k * r * n for r in rows]


def unpack_results(recv: torch.Tensor, own: list[int], world: int, my_rows: int, n: int) -> list[torch.Tensor]:
    counts = [own.count(o) for o in range(world)]
    parts = torch.split(recv, [c * my_rows * n for c in counts])
    by_owner = [p.view(c, my_rows, n) for p, c in zip(parts, counts)]
    taken = [0] * world
    out = []
    for o in own:
        out.append(by_owner[o][taken[o]])
        taken[o] += 1
    return out


def _all_to_all(send: torch.Tensor, send_sizes: list[int], recv_sizes: list[int], group) -> torch.Tensor:
    recv = send.new_empty(sum(recv_sizes))
    dist.all_to_all_single(recv, send.contiguous(), recv_sizes, send_sizes, group=group)
    return recv


def _row_sharded(u) -> bool:
    if not isinstance(u, DTensor) or u.ndim != 2:
        return False
    if not all(isinstance(p, (Shard, Replicate)) for p in u.placements):
        return False
    shards = [p for p in u.placements if isinstance(p, Shard)]
    return len(shards) == 1 and shards[0].dim == 0


def _orthogonalize_owned(us: list[DTensor], steps: int, start: int) -> list[torch.Tensor]:
    ref = us[0]
    mesh = ref.device_mesh
    dim = next(i for i, p in enumerate(ref.placements) if isinstance(p, Shard))
    group, world, rank = mesh.get_group(dim), mesh.size(dim), mesh.get_local_rank(dim)
    m, n = ref.shape
    rows = row_sizes(m, world)
    own = owners(len(us), world, start)
    k = own.count(rank)
    send, send_sizes = pack_to_owners([u.to_local().to(torch.bfloat16) for u in us], own, world)
    recv = _all_to_all(send, send_sizes, [k * r * n for r in rows], group)
    full = newton_schulz5(unpack_at_owner(recv, k, rows, n), steps) if k else recv.new_empty(0, m, n)
    send, send_sizes = pack_from_owner(full, rows)
    recv = _all_to_all(send, send_sizes, [own.count(o) * rows[rank] * n for o in range(world)], group)
    return unpack_results(recv, own, world, rows[rank], n)


def orthogonalize_all(us: list[torch.Tensor], steps: int, bucket_numel: int = 1 << 26, check: bool = False) -> tuple[list[torch.Tensor], float | None]:
    out: list[torch.Tensor | None] = [None] * len(us)
    plain, owned = defaultdict(list), defaultdict(list)
    for i, u in enumerate(us):
        if not isinstance(u, DTensor):
            plain[tuple(u.shape)].append(i)
        elif _row_sharded(u):
            owned[(tuple(u.shape), u.device_mesh, tuple(u.placements))].append(i)
        else:
            out[i] = orthogonalize(u, steps)
    for shape, idx in plain.items():
        per = max(1, bucket_numel // math.prod(shape))
        for b in range(0, len(idx), per):
            chunk = idx[b : b + per]
            for i, r in zip(chunk, newton_schulz5(torch.stack([us[i].to(torch.bfloat16) for i in chunk]), steps)):
                out[i] = r
    err, start = 0.0, 0
    for (shape, mesh, placements), idx in owned.items():
        world = mesh.size(next(d for d, p in enumerate(placements) if isinstance(p, Shard)))
        per = max(1, bucket_numel * world // math.prod(shape))
        for b in range(0, len(idx), per):
            chunk = idx[b : b + per]
            pieces = _orthogonalize_owned([us[i] for i in chunk], steps, start)
            start += len(chunk)
            for i, local in zip(chunk, pieces):
                u = us[i]
                if check:
                    ref = _local_slice(newton_schulz5(u.full_tensor(), steps), u)
                    if ref.numel():
                        err = max(err, float((ref.float() - local.float()).abs().max()))
                out[i] = DTensor.from_local(local.contiguous(), u.device_mesh, u.placements, run_check=False, shape=u.shape, stride=u.stride())
    if not check:
        return out, None
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([err], device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        err = float(t)
    return out, err


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=4e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5, momentum_dtype: torch.dtype = torch.float32, bucket_numel: int = 1 << 26, check: bool = False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.momentum_dtype, self.bucket_numel, self.check = momentum_dtype, bucket_numel, check
        self.check_error: float | None = None

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
            params, us = [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, dtype=self.momentum_dtype)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                us.append(g.add(buf.to(g.dtype), alpha=mu) if group["nesterov"] else buf)
                params.append(p)
            outs, err = orthogonalize_all(us, group["ns_steps"], self.bucket_numel, self.check)
            if err is not None:
                self.check_error = max(self.check_error or 0.0, err)
            for p, o in zip(params, outs):
                scale = 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))
                p.mul_(1 - lr * wd)
                p.add_(o.to(p.dtype), alpha=-lr * scale)
