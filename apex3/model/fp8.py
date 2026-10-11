from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

FAST_ACCUM = False
RECIPE = "rowwise"
RECIPES = ("rowwise", "tensorwise", "rowwise_with_gw_hp")
_CONFIGS: dict = {}


def available() -> bool:
    try:
        import torchao.float8
        from torchao.prototype.moe_training import fp8_grouped_mm
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def float8_linear_config(recipe: str, fast_accum: bool):
    from torchao.float8 import Float8GemmConfig, Float8LinearConfig

    if recipe not in RECIPES:
        raise ValueError(f"fp8 recipe: one of {RECIPES}, not {recipe!r}")
    cfg = _CONFIGS.get((recipe, fast_accum))
    if cfg is None:
        cfg = Float8LinearConfig.from_recipe_name(recipe)
        if not fast_accum:
            cfg = dataclasses.replace(cfg, gemm_config_output=Float8GemmConfig(use_fast_accum=False))
        _CONFIGS[(recipe, fast_accum)] = cfg
    return cfg


def to_float8_linear(lin: nn.Linear) -> nn.Module:
    from torchao.float8.float8_linear import Float8Linear

    if lin.in_features % 16 or lin.out_features % 16:
        raise ValueError(f"FP8 linear needs in/out features divisible by 16: {lin.in_features}x{lin.out_features}")
    return Float8Linear.from_float(lin, float8_linear_config(RECIPE, FAST_ACCUM))


def grouped_mm(a: torch.Tensor, w_t: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    from torchao.prototype.moe_training.fp8_grouped_mm import _to_fp8_rowwise_then_scaled_grouped_mm

    return _to_fp8_rowwise_then_scaled_grouped_mm(a, w_t, offs, out_dtype=a.dtype, pad_token_groups_for_grouped_mm=False)
