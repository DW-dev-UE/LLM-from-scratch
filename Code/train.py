"""Apex-2 3.02B pretrain entry point — 2x H100 80GB, FSDP2.

Stage 1 of 2. This trains the 4,096-token base model; apex2/extend_context.py
then takes its final checkpoint to 16,384 tokens with YaRN.

    torchrun --standalone --nproc_per_node=2 apex2/train.py
    torchrun --standalone --nproc_per_node=2 apex2/train.py --resume latest

Smoke test (single GPU, tiny model, no real data needed):

    python apex2/train.py --smoke

Runtime, from FLOPs rather than from a peak-TFLOPS ratio: 6ND undercounts
attention by 12 * n_layers * seq_len * d_model per token, which at 4,096 context
is +23%. 450B tokens is therefore 1.0e22 FLOPs, not 8.15e21, and at the 40-48%
MFU a 2-GPU full-shard job actually sustains that is 122-147 days, not 106.
Budget 600B only if the instance reservation covers ~6 months.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Running as a script (``python apex2/train.py`` or ``torchrun ... train.py``)
# puts apex2/ on sys.path but not the repo root, so ``import apex2`` fails.
# Both that form and ``python -m apex2.train`` work with this.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

from composelm import (ModelConfig, Trainer, TrainingConfig, build_model,
    load_model_weights)
from composelm.train.callbacks import TensorBoardCallback

from apex2.data import PackedCorpus, corpus_fingerprint

# --- Architecture -------------------------------------------------------
# 28 x (attn 25.17M + SwiGLU FFN 75.50M) + tied embedding 201.33M = 3.02B
ARCH = dict(
    d_model=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    head_dim=128,
    intermediate_size=8192,
    vocab_size=65_536,
    max_seq_len=4096,
    attention_type="gqa",
    ffn_type="swiglu",
    norm="rmsnorm",
    norm_placement="pre",
    pos_emb="rope",
    rope_theta=10_000.0,
    original_max_seq_len=4096,
    tie_word_embeddings=True,
    use_bias=False,
    qkv_bias=False,
    attention_output_bias=False,
    ffn_bias=False,
    # Recompute stays off only because MICRO_BATCH is 2. Per token per layer the
    # backward pass holds ~51K bf16 elements (qkv 5120, gate/up 16384, silu 8192,
    # residual/norm ~21.5K) = ~100 KB, so activations run
    # 100 KB x 8192 tok x 28 layers = 22.9 GB on top of 24.2 GB of sharded
    # state -> ~47 GB of 80. At the old bs=4 the same arithmetic gives 45.8 GB
    # of activations for ~70 GB total, which does not survive FSDP all-gather
    # buffers plus allocator fragmentation. The 16K extension stage re-enables
    # recompute (see extend_context.py) because 16,384 tokens per micro-batch
    # puts it back in that range.
    activation_checkpointing=False,
    loss_chunk_size=2048,
    precision="bf16_mixed",
    # Measured on 1xH100 (bench.sh, 2026-08-08), 3 steady steps each:
    #   eager  mb1  0.2786 s/sample  33.2% MFU  73.8 GB reserved
    #   compile mb1 0.2209 s/sample  41.9% MFU  64.1 GB
    #   eager  mb2  OOM at 79.1 GB
    #   compile mb2 0.1895 s/sample  48.8% MFU  76.0 GB   <- production
    # Inductor's fusion is what buys the headroom that makes mb2 fit at all, so
    # compile and MICRO_BATCH=2 are one decision, not two. composelm falls back
    # to eager if the backend breaks (kernels/compile.py:52) — which would then
    # OOM at mb2, so a fallback must be treated as a hard failure, not a warning.
    use_compile=True,
    compile_mode="default",
)

# 2 ranks x 2 x 4096 x 128 = 2,097,152 tokens per optimizer step — unchanged
# from bs=4/accum=64, so the LR and step budget below carry over exactly.
MICRO_BATCH = 2
GRAD_ACCUM = 128
SEQ_LEN = 4096
TOKENS_PER_STEP = 2 * MICRO_BATCH * SEQ_LEN * GRAD_ACCUM

# max_steps is part of TrainingConfig.fingerprint and therefore FROZEN for the
# life of the run — set it to the largest budget you might use (600B), never to
# the one you expect to stop at. Where you actually stop is decided by
# --decay-start, which lives outside the config and can change between resumes.
#
# Sizing rationale: at 1.0e22 FLOPs the compute-optimal point is ~8.2B params on
# ~165B tokens, so 3.02B is NOT loss-optimal for this budget — it is a
# deliberate over-train (450B/3.02B = 149 tokens/param, 199 at the 600B
# extension) that buys ~2.7x cheaper inference for a few points of loss. Do not
# describe 3.02B as "the largest model the budget allows"; it is the smallest
# one that still hits the quality targets.
MAX_STEPS = 286_000            # 600B tokens = 199 tokens/param
DEFAULT_DECAY_START = 193_000  # decay ends at ~214.5K steps = 450B = 149 tok/param


def batch_plan(world_size: int, micro_override: int | None = None) -> tuple[int, int]:
    """(micro_batch, grad_accum) preserving TOKENS_PER_STEP at any world size.

    Every world size runs micro_batch=2. The estimate this once carried — that
    one rank could only afford one sample beside a 48.3 GB unsharded optimizer
    state — was measured and found wrong in the direction that matters: eager
    mb2 does OOM (79.1 GB), but with torch.compile the same config sits at
    76.0 GB and runs 32% faster per sample than mb1. See ARCH for the table.

    Because tokens/step is unchanged, the LR schedule and step budget carry
    over, and a later move to more GPUs is a weights-only restart
    (--init-weights), not a schedule change.

    ``micro_override`` exists because that 1-rank reasoning is a memory estimate,
    not a measurement: --micro-batch lets the bench settle it empirically. Any
    value that divides TOKENS_PER_STEP keeps the schedule intact.
    """
    micro = micro_override or MICRO_BATCH
    accum, remainder = divmod(TOKENS_PER_STEP, world_size * micro * SEQ_LEN)
    if remainder:
        raise ValueError(
            f"world_size {world_size} cannot hold tokens/step at {TOKENS_PER_STEP:,}")
    return micro, accum


def wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    decay_start: int,
    decay_end: int,
    min_ratio: float = 0.1,
) -> LambdaLR:
    """Warmup - Stable - Decay.

    composelm hardwires cosine-to-zero over ``max_steps``
    (composelm/train/optim.py via trainer.py:777), which forces you to commit to
    an endpoint before the run starts. WSD keeps LR flat and decays only over
    the final stretch, so the stopping point can be chosen months in — whenever
    the eval curve flattens or the budget runs out.

    Safe to change across resumes: LambdaLR.state_dict() carries ``last_epoch``
    but not the lambda, and ``decay_start`` is not a TrainingConfig field, so it
    never enters the resume fingerprint.
    """
    if not 0 <= warmup_steps <= decay_start < decay_end:
        raise ValueError(
            f"need 0 <= warmup({warmup_steps}) <= decay_start({decay_start}) "
            f"< decay_end({decay_end})"
        )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / (decay_end - decay_start))
        # 1-sqrt decay (MiniCPM's WSD result): since sqrt(p) >= p on [0,1], this
        # is at or below linear everywhere — it drops fast at the start of the
        # ramp and then flattens, spending most of the decay near the floor.
        # That front-loaded drop is the point; a schedule that instead held the
        # LR high and dropped late would need a longer ramp to converge.
        return min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class ApexTrainer(Trainer):
    """Trainer with the WSD schedule swapped in for composelm's fixed cosine."""

    def __init__(self, *args, decay_start: int, decay_end: int,
                 min_lr_ratio: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduler = wsd_scheduler(self.optimizer, warmup_steps=self.warmup_steps,
            decay_start=decay_start, decay_end=decay_end, min_ratio=min_lr_ratio)


def build_config(smoke: bool, compile_mode: str | None = None) -> ModelConfig:
    arch = dict(ARCH)
    if smoke:
        arch.update(d_model=512, n_layers=4, n_heads=8, n_kv_heads=2, head_dim=64,
            intermediate_size=1376, vocab_size=65_536, max_seq_len=1024)
    if compile_mode:
        # composelm probes the Inductor backend and falls back to eager on
        # failure (kernels/compile.py:48), so this cannot harden into a crash.
        arch.update(use_compile=True, compile_mode=compile_mode)
    return ModelConfig.from_arch("custom", **arch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, default=Path("apex2/mixture.json"))
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens"))
    parser.add_argument("--output-dir", type=str, default="runs/apex2-3b")
    parser.add_argument("--resume", type=str, default=None,
        help="'latest' or a checkpoint directory")
    parser.add_argument("--init-weights", type=Path, default=None,
        help="checkpoint dir to load model weights from when restarting on a "
             "different GPU count — optimizer/data state start fresh, so do "
             "this early in the run or accept a brief loss bump")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--decay-start", type=int, default=DEFAULT_DECAY_START,
        help="step at which LR starts decaying; changeable between resumes")
    parser.add_argument("--decay-steps", type=int, default=21_500,
        help="length of the decay ramp (~10%% of the run)")
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--smoke", action="store_true",
        help="tiny model + synthetic data, single process")
    parser.add_argument("--micro-batch", type=int, default=None,
        help="override batch_plan's micro batch; grad_accum is re-derived so "
             "tokens/step stays at TOKENS_PER_STEP")
    parser.add_argument("--compile", dest="compile_mode", nargs="?",
        const="default", default=None,
        help="enable torch.compile (optionally a mode: default, max-autotune)")
    parser.add_argument("--bench-steps", type=int, default=0,
        help="throughput probe: run N steps with --bench-accum micro-steps "
             "each, log every step, never checkpoint. Real model and real data, "
             "so step_time/accum is the production per-micro-step cost")
    parser.add_argument("--bench-accum", type=int, default=8,
        help="grad accumulation used while benching (see --bench-steps)")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    try:
        model = build_model(build_config(args.smoke, args.compile_mode))
        params = sum(p.numel() for p in model.parameters())
        if local_rank == 0:
            print(f"parameters: {params / 1e9:.3f}B", flush=True)

        if args.smoke:
            dataset, fingerprint = None, None  # composelm generates synthetic data
            # save_steps here is not about safety — it produces the intermediate
            # checkpoints the resume test needs.
            max_steps, save_steps, logging_steps = 500, 100, 20
            grad_accum, micro_batch, seq_len = 1, 4, 1024
            decay_start, decay_end = 400, 500
        else:
            dataset = PackedCorpus(args.mixture, args.tokens, seq_len=SEQ_LEN, seed=42)
            fingerprint = corpus_fingerprint(args.mixture, args.tokens)
            max_steps, save_steps, logging_steps = MAX_STEPS, args.save_steps, 20
            micro_batch, grad_accum = batch_plan(max(1, world_size), args.micro_batch)
            seq_len = SEQ_LEN
            decay_start = args.decay_start
            decay_end = args.decay_start + args.decay_steps

        warmup = 50 if args.smoke else args.warmup_steps
        if args.bench_steps:
            # Throughput probe. A short accumulation makes each logged step cheap,
            # while the thing being measured — optimizer state plus one
            # micro-batch of activations — is bit-for-bit the production
            # footprint, so step_time_s / grad_accum carries over directly.
            max_steps, grad_accum = args.bench_steps, args.bench_accum
            logging_steps, save_steps = 1, 10 ** 9
            warmup, decay_start, decay_end = 0, max_steps, max_steps + 1

        if args.init_weights is not None:
            if args.resume is not None:
                raise SystemExit("--init-weights and --resume are mutually "
                    "exclusive: resume restores weights itself")
            load_model_weights(args.init_weights, model, strict=True)
            if local_rank == 0:
                print(f"initialized weights from {args.init_weights}", flush=True)

        training = TrainingConfig(
            batch_size=micro_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=args.lr,
            weight_decay=0.1,
            max_steps=max_steps,
            warmup_steps=warmup,
            max_grad_norm=1.0,
            output_dir=args.output_dir,
            logging_steps=logging_steps,
            save_steps=save_steps,
            precision="bf16_mixed",
            device=device,
            seq_len=seq_len,
            # num_workers=0 keeps exact resume on the dataset's own state_dict
            # (trainer.py:1214) — torchdata StatefulDataLoader is only required
            # above zero, and memmap reads do not need a worker pool.
            num_workers=0,
            seed=42,
            data_fingerprint=fingerprint,
            strategy="fsdp2" if distributed else "single",
            fsdp_auto_wrap=True,
            # Must equal the precision compute dtype or Trainer raises
            # (trainer.py:706-719).
            fsdp_mixed_precision="bf16" if distributed else None,
        )

        callbacks = []
        if local_rank == 0:
            callbacks.append(
                TensorBoardCallback(str(Path(args.output_dir) / "tensorboard")))

        trainer = ApexTrainer(model, dataset, config=training,
            decay_start=decay_start, decay_end=decay_end, callbacks=callbacks)

        if local_rank == 0:
            print(f"tokens/step: {micro_batch * seq_len * grad_accum * max(1, world_size):,}"
                  f"   max_steps: {max_steps:,}"
                  f"   decay: {decay_start:,} -> {decay_end:,}", flush=True)

        result = trainer.fit(resume_from=args.resume)
        if local_rank == 0:
            print(f"done: {result.steps} steps  {result.elapsed_sec / 3600:.1f} h")
            print(f"last: {result.history[-1] if result.history else '-'}")
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
