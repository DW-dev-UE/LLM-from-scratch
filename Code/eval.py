"""Locked evaluation protocol for Apex-2.

Benchmark numbers are only meaningful against a frozen protocol — lm-eval task
definitions, shot counts and metrics all drift between releases, which is why
the same model can post different ARC-Challenge scores under different harness
configs. Everything that affects a number is pinned here, and the harness
version is recorded next to every result.

    pip install "lm-eval[api]==0.4.9" transformers
    python apex2/eval.py --ckpt runs/apex2-3b/latest --tag step-50k

Comparison baseline is Apex-1 (lm-eval 0-shot average 49.62) — keep TASKS_0SHOT
unchanged so the two remain comparable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from composelm import build_model, export_vllm_checkpoint, load_model_weights

from apex2.train import ARCH, build_config

LM_EVAL_VERSION = "0.4.9"

# Primary target set — Path A goals. 0-shot, matching the Apex-1 baseline.
TASKS_0SHOT = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "openbookqa",
    "boolq",
    "lambada_openai",
]

# Knowledge, conventionally few-shot. Reported separately — never folded into
# the 0-shot average, or the comparison with Apex-1 breaks.
TASKS_5SHOT = ["mmlu"]

# Code. Requires --confirm_run_unsafe_code (the harness executes generations).
TASKS_CODE = ["humaneval", "mbpp"]

# Release targets, (floor, ceiling). Calibrated against models of this size at a
# comparable token budget rather than against models of this size generally:
# Pythia-2.8B (300B tokens) and OpenLLaMA-3B (1T) are the honest reference
# points, not Llama-3.2-3B (9T) or Qwen2.5-3B (18T). Data quality moves these —
# FineWeb-Edu's own ablation reports ARC 46 -> 57 at 1.8B/350B — but it moves
# them from the Pythia baseline, not from the 9T-model baseline.
TARGETS: dict[str, tuple[float, float]] = {
    # Pythia-2.8B 64.4 / OpenLLaMA-3B 69.4; edu-heavy mixture is the upside.
    "0shot/arc_easy": (63.0, 70.0),
    # Pythia-2.8B 32.9 / OpenLLaMA-3B 34.1. 45 was a 9T-model number.
    "0shot/arc_challenge": (36.0, 42.0),
    # Scales with general web tokens, which edu+code+math filtering costs us;
    # 68 needs ~1.5T tokens. Raising dclm-baseline to 0.20 is what buys the 63.
    "0shot/hellaswag": (58.0, 63.0),
    "0shot/piqa": (73.0, 77.0),
    "0shot/winogrande": (58.0, 63.0),
    # 5-shot. 25 is chance; anything above ~30 at 450B is a real signal, and 38
    # is Llama-3.2-3B territory.
    "5shot/mmlu": (29.0, 33.0),
    # 90B of unfiltered permissive code. StarCoder2-3B needed 3T code tokens for
    # 31.7, and dropping Stack-Edu gave up the quality-per-token that would have
    # narrowed that gap.
    "code/humaneval": (15.0, 22.0),
    "code/mbpp": (18.0, 26.0),
}

# Ship gate: must beat Apex-1 on the frozen 0-shot protocol.
BASELINE_0SHOT_AVERAGE = 49.62
TARGET_0SHOT_AVERAGE = (53.0, 57.0)


def export_hf(checkpoint: Path, tokenizer_dir: Path, out_dir: Path,
              *, extended: bool = False) -> Path:
    """composelm checkpoint -> HF directory that lm-eval can load."""
    if out_dir.exists() and (out_dir / "config.json").is_file():
        print(f"reusing existing export at {out_dir}")
        return out_dir

    if extended:
        from apex2.extend_context import build_extended_config

        config = build_extended_config()
    else:
        config = build_config(smoke=False)
    model = build_model(config)
    load_model_weights(checkpoint, model, strict=True)
    model.eval()

    from transformers import PreTrainedTokenizerFast

    specials = json.loads(
        (tokenizer_dir / "special_tokens.json").read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
        eos_token="<|endoftext|>", bos_token="<|endoftext|>",
        pad_token="<|fim_pad|>", unk_token=None,
        additional_special_tokens=[t for t in specials if t.startswith("<|reserved")],
    )

    # Llama-compatible layout (GQA + SwiGLU + RMSNorm + RoPE, bias-free), which
    # is what the exporter accepts.
    export_vllm_checkpoint(model, out_dir, tokenizer=tokenizer)
    print(f"exported HF checkpoint -> {out_dir}")
    return out_dir


def run_lm_eval(
    model_dir: Path,
    tasks: list[str],
    num_fewshot: int,
    out_path: Path,
    *,
    batch_size: str = "auto",
    unsafe_code: bool = False,
) -> dict:
    command = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args",
        f"pretrained={model_dir},dtype=bfloat16,trust_remote_code=False",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", batch_size,
        "--device", "cuda:0" if torch.cuda.is_available() else "cpu",
        "--output_path", str(out_path),
        "--seed", "1234",
    ]
    if unsafe_code:
        command.append("--confirm_run_unsafe_code")
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)

    results = sorted(out_path.rglob("results_*.json"))
    if not results:
        raise RuntimeError(f"lm-eval wrote no results under {out_path}")
    return json.loads(results[-1].read_text(encoding="utf-8"))["results"]


def pick_metric(task_result: dict) -> float | None:
    """acc_norm where the task defines it, else acc — fixed choice, no cherry-picking."""
    for key in ("acc_norm,none", "acc,none", "pass@1,create_test", "exact_match,none"):
        if key in task_result:
            return float(task_result[key]) * 100.0
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True,
        help="composelm checkpoint dir, 'latest' pointer, or safetensors file")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/apex2"))
    parser.add_argument("--tag", type=str, required=True, help="e.g. step-50k")
    parser.add_argument("--out", type=Path, default=Path("evals"))
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--skip-mmlu", action="store_true")
    parser.add_argument("--extended", action="store_true",
        help="checkpoint is from extend_context.py (16K YaRN), not the 4K base")
    args = parser.parse_args()

    run_dir = args.out / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = export_hf(args.ckpt, args.tokenizer, run_dir / "hf",
        extended=args.extended)

    scores: dict[str, float] = {}
    raw: dict[str, dict] = {}

    raw["0shot"] = run_lm_eval(model_dir, TASKS_0SHOT, 0, run_dir / "0shot")
    if not args.skip_mmlu:
        raw["5shot"] = run_lm_eval(model_dir, TASKS_5SHOT, 5, run_dir / "5shot")
    if not args.skip_code:
        raw["code"] = run_lm_eval(model_dir, TASKS_CODE, 0, run_dir / "code",
            unsafe_code=True)

    for group, results in raw.items():
        for task, values in results.items():
            score = pick_metric(values)
            if score is not None:
                scores[f"{group}/{task}"] = round(score, 2)

    zero_shot = [v for k, v in scores.items()
        if k.startswith("0shot/") and k.split("/", 1)[1] in TASKS_0SHOT]
    average = round(sum(zero_shot) / len(zero_shot), 2) if zero_shot else None

    summary = {
        "tag": args.tag,
        "checkpoint": str(args.ckpt),
        "context": 16_384 if args.extended else ARCH["max_seq_len"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lm_eval_pinned": LM_EVAL_VERSION,
        "arch": {k: ARCH[k] for k in
            ("d_model", "n_layers", "n_heads", "n_kv_heads", "intermediate_size",
             "vocab_size", "max_seq_len")},
        "zero_shot_average": average,
        "baseline_zero_shot_average": BASELINE_0SHOT_AVERAGE,
        "targets": {k: list(v) for k, v in TARGETS.items()},
        "scores": scores,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n=== {args.tag} ===")
    for name, value in sorted(scores.items()):
        target = TARGETS.get(name)
        if target is None:
            print(f"  {name:32s} {value:6.2f}")
            continue
        low, high = target
        mark = "  " if value >= low else "  BELOW"
        print(f"  {name:32s} {value:6.2f}   target {low:.0f}-{high:.0f}{mark}")
    if average is not None:
        low, high = TARGET_0SHOT_AVERAGE
        gate = "PASS" if average > BASELINE_0SHOT_AVERAGE else "FAIL"
        print(f"  {'0-shot average':32s} {average:6.2f}   target {low:.0f}-{high:.0f}"
              f"   vs Apex-1 {BASELINE_0SHOT_AVERAGE}: {gate}")
    print(f"\nwrote {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
