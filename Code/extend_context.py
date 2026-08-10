"""Stage 2 — extend Apex-2 from 4,096 to 16,384 tokens with YaRN.

The base run (apex2/train.py) never sees a position past 4,096, so the 16K
context claimed in the spec does not exist until this stage runs. YaRN's
NTK-by-parts interpolation degrades much less than raw extrapolation at a 4x
jump, but it is not free: without a short fine-tune at the target length the
model loses several points on short-context benchmarks as well.

    torchrun --standalone --nproc_per_node=2 apex2/extend_context.py \
        --init-from runs/apex2-3b/latest

Budget is 8B tokens, ~1.8% of pretrain. Runtime is not proportional to that
share: at 16,384 context the attention term (12 * n_layers * seq_len * d_model)
grows 4x to 16.9 GFLOPs/token, roughly equal to the 18.1 GFLOPs of 6N, and
recompute adds ~33% on top. 8B tokens is therefore ~3.5e20 FLOPs = 5-6 days on
2x H100 at 40-45% MFU.

All positional buffers in composelm are registered with persistent=False
(layers/pos_emb.py:184,380,609), so swapping pos_emb rope -> yarn changes no
parameter names and the base weights load with strict=True.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist

from composelm import ModelConfig, TrainingConfig, build_model, load_model_weights
from composelm.train.callbacks import TensorBoardCallback

from apex2.data import PackedCorpus, corpus_fingerprint
from apex2.train import ARCH, ApexTrainer

EXTENDED_SEQ_LEN = 16_384
BASE_SEQ_LEN = 4_096
YARN_FACTOR = EXTENDED_SEQ_LEN / BASE_SEQ_LEN  # 4.0

# 2 ranks x 1 x 16384 x 32 = 1,048,576 tokens per optimizer step
MICRO_BATCH = 1
GRAD_ACCUM = 32
TOKENS_PER_STEP = 2 * MICRO_BATCH * EXTENDED_SEQ_LEN * GRAD_ACCUM
DEFAULT_TOKENS = 8_000_000_000
DEFAULT_STEPS = DEFAULT_TOKENS // TOKENS_PER_STEP  # 7,629


def grad_accum_for(world_size: int) -> int:
    """Accum that keeps TOKENS_PER_STEP at any world size (1 GPU -> 64).

    Without this, a single-GPU run would train half the intended tokens/step
    while DEFAULT_STEPS stayed sized for two ranks — an 8B-token stage would
    silently become 4B.
    """
    accum, remainder = divmod(TOKENS_PER_STEP,
        world_size * MICRO_BATCH * EXTENDED_SEQ_LEN)
    if remainder:
        raise ValueError(f"world_size {world_size} cannot hold "
            f"tokens/step at {TOKENS_PER_STEP:,}")
    return accum


def build_extended_config() -> ModelConfig:
    """Base architecture at 16K with YaRN. Weight-compatible with train.ARCH."""
    arch = dict(ARCH)
    arch.update(
        pos_emb="yarn",
        max_seq_len=EXTENDED_SEQ_LEN,
        original_max_seq_len=BASE_SEQ_LEN,
        # beta_fast/beta_slow left at composelm's 32/1 — the values from the YaRN
        # paper, tuned for exactly this head_dim=128 / theta=10000 regime.
        rope_scaling={
            "type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": BASE_SEQ_LEN,
        },
        # 16,384 tokens per micro-batch is the activation footprint that forced
        # bs=2 at 4K (~46 GB). Recompute is mandatory here, and its ~33% FLOPs
        # cost is affordable across 8B tokens.
        activation_checkpointing=True,
    )
    return ModelConfig.from_arch("custom", **arch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-from", type=Path, required=True,
        help="base 4K checkpoint dir or 'latest' pointer from runs/apex2-3b")
    parser.add_argument("--mixture", type=Path, default=Path("apex2/mixture.json"),
        help="reuse the pretrain mixture, or point at a long-document variant; "
             "packed 16K blocks are concatenated documents, so a mixture skewed "
             "toward long files (stack-v3, pes2o) teaches long-range attention "
             "better than the pretrain weights do")
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens"))
    parser.add_argument("--output-dir", type=str, default="runs/apex2-3b-16k")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-5,
        help="0.1x the pretrain peak — the WSD floor the base run decayed to")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--save-steps", type=int, default=250)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    try:
        model = build_model(build_extended_config())
        if args.resume is None:
            # Only on a fresh start: resuming restores these weights from the
            # stage-2 checkpoint, and loading the base over them would silently
            # discard the extension progress.
            load_model_weights(args.init_from, model, strict=True)
            if local_rank == 0:
                print(f"loaded base weights from {args.init_from}", flush=True)

        dataset = PackedCorpus(args.mixture, args.tokens,
            seq_len=EXTENDED_SEQ_LEN, seed=43)  # seed != 42: fresh data order

        grad_accum = grad_accum_for(max(1, world_size))
        training = TrainingConfig(
            batch_size=MICRO_BATCH,
            gradient_accumulation_steps=grad_accum,
            learning_rate=args.lr,
            weight_decay=0.1,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            max_grad_norm=1.0,
            output_dir=args.output_dir,
            logging_steps=20,
            save_steps=args.save_steps,
            precision="bf16_mixed",
            device=device,
            seq_len=EXTENDED_SEQ_LEN,
            num_workers=0,
            seed=43,
            data_fingerprint=corpus_fingerprint(args.mixture, args.tokens),
            strategy="fsdp2" if distributed else "single",
            fsdp_auto_wrap=True,
            fsdp_mixed_precision="bf16" if distributed else None,
        )

        callbacks = []
        if local_rank == 0:
            callbacks.append(
                TensorBoardCallback(str(Path(args.output_dir) / "tensorboard")))

        # Decay over the last 30% — short stage, so the flat stretch is brief.
        decay_start = int(args.max_steps * 0.7)
        trainer = ApexTrainer(model, dataset, config=training,
            decay_start=decay_start, decay_end=args.max_steps, callbacks=callbacks)

        if local_rank == 0:
            print(f"16K extension: {args.max_steps:,} steps x "
                  f"{MICRO_BATCH * EXTENDED_SEQ_LEN * grad_accum * max(1, world_size):,}"
                  f" tokens = {args.max_steps * TOKENS_PER_STEP / 1e9:.1f}B"
                  f"   yarn factor {YARN_FACTOR:g}", flush=True)

        result = trainer.fit(resume_from=args.resume)
        if local_rank == 0:
            print(f"done: {result.steps} steps  {result.elapsed_sec / 3600:.1f} h")
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
