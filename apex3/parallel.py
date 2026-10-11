from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except ImportError:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    return rank, world, local


def build_mesh(world: int, shard_size: int = 8) -> DeviceMesh:
    shard = min(shard_size, world)
    if world % shard:
        raise ValueError(f"world size {world} is not a multiple of shard size {shard}")
    if world == shard:
        return init_device_mesh("cuda", (world,), mesh_dim_names=("shard",))
    return init_device_mesh("cuda", (world // shard, shard), mesh_dim_names=("replicate", "shard"))


def apply_fsdp(model, mesh: DeviceMesh, reduce_dtype: torch.dtype = torch.float32):
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype)
    mp_fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    for layer in model.layers:
        if layer.is_moe:
            fully_shard(layer.ffn.router, mesh=mesh, mp_policy=mp_fp32)
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    if not model.cfg.tie_embeddings:
        fully_shard(model.embed, mesh=mesh, mp_policy=mp)
        fully_shard(model.head, mesh=mesh, mp_policy=mp, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def set_grad_sync(model, is_last_microbatch: bool) -> None:
    if hasattr(model, "set_requires_all_reduce"):
        model.set_requires_all_reduce(is_last_microbatch)


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_max(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t
