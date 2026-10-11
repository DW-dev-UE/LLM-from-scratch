from __future__ import annotations

import contextlib

TE_PRECISIONS = ("fp8_block", "fp8_current", "mxfp8")
PRECISIONS = ("bf16", "fp8_rowwise") + TE_PRECISIONS


def is_te_precision(precision: str) -> bool:
    return precision in TE_PRECISIONS


def linear_for(precision: str) -> str:
    if precision == "fp8_rowwise":
        return "fp8_rowwise"
    return "te" if is_te_precision(precision) else "torch"


def fp8_recipe(precision: str):
    from transformer_engine.common import recipe

    if precision == "fp8_block":
        return recipe.Float8BlockScaling()
    if precision == "fp8_current":
        return recipe.Float8CurrentScaling()
    if precision == "mxfp8":
        return recipe.MXFP8BlockScaling()
    raise ValueError(f"unknown precision: {precision}")


def precision_context(precision: str):
    if not is_te_precision(precision):
        return contextlib.nullcontext()
    import transformer_engine.pytorch as te

    if hasattr(te, "autocast"):
        return te.autocast(enabled=True, recipe=fp8_recipe(precision))
    return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe(precision))


def fp8_weights_context(precision: str):
    if not is_te_precision(precision):
        raise ValueError("FP8 weights need a Transformer Engine FP8 precision")
    import transformer_engine.pytorch as te

    model_init = getattr(te, "quantized_model_init", None) or te.fp8_model_init
    return model_init(enabled=True, recipe=fp8_recipe(precision))


def moe_align_for(precision: str) -> int:
    return 32 if precision == "mxfp8" else 16
