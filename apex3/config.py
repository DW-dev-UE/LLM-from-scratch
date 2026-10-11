from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 204_800
    d_model: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 4
    head_dim: int = 128
    global_every: int = 6
    window: int = 1024
    rope_theta_local: float = 1e4
    rope_theta_global: float = 1e6
    rope_scale_global: float = 1.0
    n_dense_layers: int = 1
    dense_ffn: int = 10_240
    n_experts: int = 128
    expert_ffn: int = 640
    top_k: int = 8
    shared_ffn: int = 5_120
    route_scale: float = 1.0
    router_bias_rate: float = 1e-3
    router_bias_update: str = "sign"
    router_bias_rate_late: float = -1.0
    router_bias_late_tokens: float = 0.0
    seq_aux_alpha: float = 1e-4
    norm_eps: float = 1e-6
    z_loss: float = 1e-4
    tie_embeddings: bool = False
    init_std: float = 0.02

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0
        assert self.global_every >= 1
        if self.n_experts:
            assert 0 < self.top_k <= self.n_experts

    def is_global(self, i: int) -> bool:
        return (i + 1) % self.global_every == 0

    def is_moe_layer(self, i: int) -> bool:
        return self.n_experts > 0 and i >= self.n_dense_layers

    @property
    def n_global_layers(self) -> int:
        return sum(self.is_global(i) for i in range(self.n_layers))

    @property
    def n_moe_layers(self) -> int:
        return sum(self.is_moe_layer(i) for i in range(self.n_layers))


@dataclass(frozen=True)
class ImplConfig:
    attn: str = "fa2"
    experts: str = "grouped_mm"
    linear: str = "torch"
    recompute: str = "none"
    precision: str = "bf16"
    compile_blocks: bool = False
    moe_align: int = 16
    ce_chunk: int = 1024
    ce_impl: str = "torch"
    moe_chunk: int = 32_768


ATTN_RATIOS = {"5:1": 6, "4:1": 5, "2:1": 3, "full": 1}

_C = ModelConfig()
_P2C = ModelConfig(d_model=1536, n_heads=12, n_kv_heads=4, dense_ffn=6144, expert_ffn=384, shared_ffn=3072)

PRESETS: dict[str, ModelConfig] = {"C": _C, "C64": replace(_C, n_experts=64, expert_ffn=1280, top_k=4, router_bias_rate=1e-2), "A": replace(_C, n_experts=136, top_k=16, shared_ffn=0), "D": replace(_C, n_experts=0), "C7": replace(_C, n_experts=32),
                                   "P1": ModelConfig(d_model=1024, n_layers=12, n_heads=8, n_kv_heads=2, dense_ffn=4096, expert_ffn=256, shared_ffn=2048), "P2-C": _P2C, "P2-C7": replace(_P2C, n_experts=32), "P2-A": replace(_P2C, n_experts=136, top_k=16, shared_ffn=0), "P2-D1": replace(_P2C, n_experts=0), "P2-D2": replace(_P2C, n_experts=0, dense_ffn=12_288)}
ALIASES = {"C21": "C", "P2-C21": "P2-C"}


def get_config(name: str, attn: str | None = None, **overrides) -> ModelConfig:
    cfg = PRESETS[ALIASES.get(name, name)]
    if attn is not None:
        cfg = replace(cfg, global_every=ATTN_RATIOS[attn])
    return replace(cfg, **overrides) if overrides else cfg


def resolve_config(state: dict, name: str, attn: str | None, overrides: dict) -> tuple[ModelConfig, bool]:
    saved = state.get("model_config")
    if saved:
        names = {f.name for f in fields(ModelConfig)}
        return replace(ModelConfig(**{k: v for k, v in saved.items() if k in names}), **overrides), True
    return get_config(name, attn=attn, **overrides), False


PHASE_FIELDS = {"rope_scale_global", "router_bias_rate", "router_bias_update", "router_bias_rate_late", "router_bias_late_tokens", "seq_aux_alpha", "z_loss", "init_std"}


def check_against_checkpoint(cfg: ModelConfig, saved: dict | None) -> list[str]:
    if not saved:
        return []
    diff = [f.name for f in fields(ModelConfig) if f.name in saved and saved[f.name] != getattr(cfg, f.name)]
    arch = [k for k in diff if k not in PHASE_FIELDS]
    if arch:
        raise ValueError("model config differs from the checkpoint: " + ", ".join(f"{k} {saved[k]} -> {getattr(cfg, k)}" for k in arch))
    return [f"{k} {saved[k]} -> {getattr(cfg, k)}" for k in diff]


def parse_overrides(items: list[str]) -> dict:
    types = {f.name: f.type for f in fields(ModelConfig)}
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        if key not in types:
            raise KeyError(f"unknown ModelConfig field: {key}")
        kind = types[key]
        if kind in ("bool", bool):
            out[key] = value.lower() in ("1", "true", "yes")
        elif kind in ("int", int):
            out[key] = int(float(value))
        elif kind in ("float", float):
            out[key] = float(value)
        else:
            out[key] = value
    return out
