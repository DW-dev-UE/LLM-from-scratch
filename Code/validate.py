"""Held-out loss for an Apex-2 checkpoint.

lm-eval scores are noisy at 3B and only move on multi-thousand-step timescales,
which makes them a bad instrument for the one decision this run actually has to
make: when to set --decay-start. Held-out loss is the instrument for that.

Runs against saved checkpoints rather than inside the training loop. composelm
dispatches callbacks on rank 0 only (train/trainer.py:1399), so a callback that
ran a forward pass would deadlock under FSDP2 the moment it hit an all-gather.
Decoupling also means validation cannot take the run down.

    python apex2/validate.py runs/apex2-3b/latest
    */30 * * * * cd /home/ubuntu/llm && python apex2/validate.py runs/apex2-3b/latest \
        --append evals/validation.jsonl

Each source's val.bin is the first 2M tokens of its stream, carved off by
prepare_data.py before any training shard was written. Costs ~10 GB of GPU
memory alongside training (bf16 weights 6 GB + one batch), which fits the ~33 GB
left free by the bs=2 pretrain config.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from composelm import build_model, load_model_weights

from apex2.train import SEQ_LEN, build_config


def load_blocks(val_path: Path, seq_len: int, max_blocks: int) -> torch.Tensor:
    """Non-overlapping blocks from a val.bin, deterministic and front-to-back."""
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    count = min(max_blocks, data.size // seq_len)
    if count == 0:
        raise ValueError(
            f"{val_path} holds {data.size:,} tokens, under one {seq_len}-token block"
        )
    usable = np.asarray(data[: count * seq_len]).reshape(count, seq_len)
    return torch.from_numpy(usable.astype(np.int64))


@torch.no_grad()
def source_loss(model: torch.nn.Module, blocks: torch.Tensor, device: str,
                batch_size: int) -> tuple[float, int]:
    """Sum of token NLL and token count — summed, not averaged, so the weighted
    mean across sources is not biased by differing block counts."""
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, blocks.size(0), batch_size):
        batch = blocks[start : start + batch_size].to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits = model(input_ids=batch)
        logits = logits.logits if hasattr(logits, "logits") else logits
        # Standard causal shift: position t predicts t+1.
        predicted = logits[:, :-1, :].float().flatten(0, 1)
        actual = batch[:, 1:].flatten()
        nll = torch.nn.functional.cross_entropy(predicted, actual, reduction="sum")
        total_nll += float(nll)
        total_tokens += actual.numel()
    return total_nll, total_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path,
        help="checkpoint dir or 'latest' pointer")
    parser.add_argument("--mixture", type=Path, default=Path("apex2/mixture.json"))
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens"))
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--max-blocks", type=int, default=64,
        help="per source; 64 x 4096 = 262K tokens is well past the point where "
             "the estimate stops moving")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--extended", action="store_true",
        help="checkpoint is from extend_context.py (16K YaRN)")
    parser.add_argument("--append", type=Path, default=Path("evals/validation.jsonl"))
    args = parser.parse_args()

    mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
    weights = {s["name"]: s["weight"] for s in mixture["sources"]}

    missing = [name for name in weights
        if not (args.tokens / name / "val.bin").is_file()]
    if missing:
        print(f"no val.bin for: {', '.join(missing)} — re-run prepare_data.py "
              f"(holdout was added after the first tokenization pass)",
            file=sys.stderr)
        return 1

    if args.extended:
        from apex2.extend_context import build_extended_config

        config = build_extended_config()
    else:
        config = build_config(smoke=False)
    model = build_model(config)
    load_model_weights(args.checkpoint, model, strict=True)
    model.to(args.device).eval()

    per_source: dict[str, float] = {}
    weighted_nll = 0.0
    for name, weight in sorted(weights.items()):
        blocks = load_blocks(args.tokens / name / "val.bin", args.seq_len,
            args.max_blocks)
        nll, tokens = source_loss(model, blocks, args.device, args.batch_size)
        loss = nll / tokens
        per_source[name] = round(loss, 4)
        # Weighted by mixture probability: this is the quantity the training
        # objective is actually minimizing, so it is the one whose flattening
        # justifies setting --decay-start.
        weighted_nll += weight * loss
        print(f"  {name:22s} w={weight:5.3f}  loss {loss:6.4f}  "
              f"ppl {np.exp(loss):8.2f}", flush=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "seq_len": args.seq_len,
        "blocks_per_source": args.max_blocks,
        "weighted_loss": round(weighted_nll, 4),
        "weighted_ppl": round(float(np.exp(weighted_nll)), 2),
        "per_source_loss": per_source,
    }
    print(f"\nweighted loss {record['weighted_loss']}  "
          f"ppl {record['weighted_ppl']}")

    if args.append:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"appended to {args.append}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
