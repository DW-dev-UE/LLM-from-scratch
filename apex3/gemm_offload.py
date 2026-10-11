from __future__ import annotations

import contextlib

import torch
import torch.fx as fx
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch.utils.checkpoint import CheckpointPolicy

from .config import ModelConfig

MODES = ("none", "hooks", "compiler")
PARTS = ("all", "experts", "shared")


def _gemm_ops() -> tuple:
    ops = [torch.ops.aten.mm.default, torch.ops.aten._grouped_mm.default]
    for name in ("_scaled_mm", "_scaled_mm_v2", "_scaled_grouped_mm", "_scaled_grouped_mm_v2"):
        packet = getattr(torch.ops.aten, name, None)
        default = getattr(packet, "default", None) if packet is not None else None
        if default is not None:
            ops.append(default)
    return tuple(ops)


_GEMMS = _gemm_ops()


def ffn_widths(cfg: ModelConfig, part: str = "all") -> tuple[int, ...]:
    experts = {cfg.expert_ffn} if cfg.n_experts else set()
    dense = {w for w in (cfg.shared_ffn, cfg.dense_ffn) if w}
    widths = {"all": experts | dense, "experts": experts, "shared": dense}[part]
    attn = {cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim}
    clash = widths & attn
    if clash:
        raise ValueError(f"FFN widths {sorted(clash)} equal an attention projection width: cannot tell them apart")
    return tuple(sorted(widths))


class MarkFFNGemmsSaved(CustomGraphPass):
    def __init__(self, widths: tuple[int, ...], offload_in_graph: bool):
        self.widths, self.offload_in_graph = tuple(widths), offload_in_graph

    def __call__(self, graph: fx.Graph) -> None:
        backward: set[fx.Node] = set()
        for n in graph.nodes:
            if (n.op == "placeholder" and n.name.startswith("tangents")) or any(a in backward for a in n.all_input_nodes):
                backward.add(n)
        for n in graph.nodes:
            if n in backward or n.op != "call_function" or n.target not in _GEMMS:
                continue
            val = n.meta.get("val")
            if isinstance(val, torch.Tensor) and val.dim() == 2 and val.shape[-1] in self.widths:
                n.meta["recompute"] = CheckpointPolicy.MUST_SAVE
                if self.offload_in_graph:
                    n.meta["should_offload"] = True

    def uuid(self) -> str:
        return f"apex3.gemm_offload.v2:{self.widths}:{self.offload_in_graph}:{','.join(str(op) for op in _GEMMS)}"


def install(cfg: ModelConfig, mode: str, part: str = "all") -> None:
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    if mode not in MODES or part not in PARTS:
        raise ValueError(f"gemm offload mode {mode!r} / part {part!r}, not one of {MODES} / {PARTS}")
    inductor_config.joint_custom_pre_pass = (None if mode == "none" else MarkFFNGemmsSaved(ffn_widths(cfg, part), offload_in_graph=mode == "compiler"))
    on = mode == "compiler"
    functorch_config.enable_activation_offloading = on
    functorch_config.activation_offload_separate_stream = on
    functorch_config.activation_offload_sink_wait = on
    functorch_config.activation_reload_prefetch = on
    functorch_config.activation_offload_cpu_gpu_bw = 300.0 if on else 50.0


def selector(cfg: ModelConfig, part: str = "all"):
    widths = frozenset(ffn_widths(cfg, part))

    def select(t: torch.Tensor) -> bool:
        return t.dim() == 2 and t.dtype == torch.bfloat16 and t.shape[-1] in widths

    return select


def compiler_pool(mode: str):
    if mode != "compiler":
        return contextlib.nullcontext()
    from torch._functorch._activation_offloading.offload_ops import pinned_memory_pool

    return pinned_memory_pool()
