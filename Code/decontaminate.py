"""Benchmark contamination audit for the Apex-2 corpus.

Every number eval.py produces is only worth what the corpus's separation from
the test sets is worth, and none of the sources in mixture.json ship a
decontamination guarantee against this particular task list.

    python apex2/decontaminate.py --docs-per-source 200000
    python apex2/decontaminate.py --verify        # fail if a rate regressed

Method is the GPT-3 / Llama one: normalize to lowercase alphanumerics, hash
every 13-gram of every test item, and count how many sampled corpus documents
contain at least one of those hashes. 13 words is long enough that natural
collisions are rare and short enough to catch a paraphrase-free copy.

This AUDITS; it does not filter. Filtering would mean running the n-gram check
inside prepare_data.py over all 450B tokens, which is a Python-speed scan
bolted onto the one job in this pipeline that must not fail halfway, and the
published corpora that this mixture is built from (FineWeb-Edu, DCLM, Pythia's
Pile) all report rather than filter for the same reason. A rate above
--threshold is a signal to drop the task from the reported set or footnote it
in the model card, not to retokenize.

Sampling is not proof of absence: at 200K docs per source the detection floor
is ~5e-6, so a rate of 0.0 means "below the floor", never "clean".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ``python apex2/decontaminate.py`` puts apex2/ on sys.path but not the repo
# root, so the ``apex2.prepare_data`` import inside audit_source() would fail
# only after every test set had been hashed — minutes in. Same guard as
# train.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NGRAM = 13
NORMALIZE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE = re.compile(r"\s+")

# (lm-eval task, HF dataset id, config, split, text fields). Mirrors eval.py's
# task list — a task evaluated but not audited is the gap this file exists to
# close, so test_apex2_pipeline.py asserts the two stay in sync.
TEST_SETS: dict[str, tuple[str, str | None, str, tuple[str, ...]]] = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", ("question",)),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", ("question",)),
    "hellaswag": ("Rowan/hellaswag", None, "validation", ("ctx",)),
    "piqa": ("ybisk/piqa", None, "validation", ("goal", "sol1", "sol2")),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation",
                   ("sentence",)),
    "openbookqa": ("allenai/openbookqa", "main", "test", ("question_stem",)),
    "boolq": ("google/boolq", None, "validation", ("question", "passage")),
    "lambada_openai": ("EleutherAI/lambada_openai", "en", "test", ("text",)),
    "mmlu": ("cais/mmlu", "all", "test", ("question",)),
    "humaneval": ("openai/openai_humaneval", None, "test", ("prompt",)),
    "mbpp": ("google-research-datasets/mbpp", "full", "test", ("text", "code")),
}


def ngram_hashes(text: str, n: int = NGRAM) -> set[bytes]:
    """8-byte blake2b digests of every n-gram. Digests, not the n-grams
    themselves, so the test-set set stays under a few hundred MB in memory."""
    words = WHITESPACE.sub(" ", NORMALIZE.sub(" ", text.lower())).split()
    if len(words) < n:
        return set()
    return {
        hashlib.blake2b(" ".join(words[i : i + n]).encode("utf-8"),
                        digest_size=8).digest()
        for i in range(len(words) - n + 1)
    }


def load_test_split(repo: str, config: str | None, split: str):
    """``load_dataset`` with a fallback to the Hub's parquet conversion.

    datasets 4 dropped loading-script support, which breaks any test set still
    published as a .py loader — ybisk/piqa is one, and an audit that silently
    skips PIQA is exactly the blind spot this file exists to prevent. The Hub
    auto-converts every dataset to parquet under refs/convert/parquet, and that
    copy is generated from the script itself, so the rows are the same rows.
    """
    from datasets import load_dataset

    try:
        return load_dataset(repo, config, split=split)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" not in str(e):
            raise
        print(f"  {repo}: script loader removed, using refs/convert/parquet",
            flush=True)
        return load_dataset(repo, config, split=split,
            revision="refs/convert/parquet")


def build_test_ngrams(tasks: list[str]) -> dict[str, set[bytes]]:
    per_task: dict[str, set[bytes]] = {}
    for task in tasks:
        repo, config, split, fields = TEST_SETS[task]
        dataset = load_test_split(repo, config, split)
        hashes: set[bytes] = set()
        for record in dataset:
            for field in fields:
                value = record.get(field)
                if isinstance(value, str):
                    hashes |= ngram_hashes(value)
        per_task[task] = hashes
        print(f"  {task:16s} {len(dataset):>7,} items  {len(hashes):>9,} 13-grams",
            flush=True)
    return per_task


def audit_source(source: dict, per_task: dict[str, set[bytes]],
                 docs: int) -> dict[str, float]:
    from datasets import load_dataset

    from apex2.prepare_data import record_text

    dataset = load_dataset(source["hf_id"], source["hf_config"], split="train",
        streaming=True)
    # One flat set for the cheap "is this document dirty at all" test; per-task
    # attribution only runs on the documents that already hit.
    combined: set[bytes] = set()
    for hashes in per_task.values():
        combined |= hashes

    hits = {task: 0 for task in per_task}
    seen = 0
    for record in dataset:
        # record_text, not a raw text_field read: stack_repo rows keep their
        # text in files[].content, and a None here would also stop `seen` from
        # advancing — the doc cap would never trigger on such a source.
        text = record_text(record, source)
        if not text:
            continue
        seen += 1
        found = ngram_hashes(text) & combined
        if found:
            for task, hashes in per_task.items():
                if found & hashes:
                    hits[task] += 1
        if seen >= docs:
            break
        if seen % 50_000 == 0:
            print(f"    {source['name']}: {seen:,} docs", flush=True)
    return {task: count / seen if seen else 0.0 for task, count in hits.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, default=Path("apex2/mixture.json"))
    parser.add_argument("--out", type=Path, default=Path("DECONTAMINATION.json"))
    parser.add_argument("--docs-per-source", type=int, default=200_000)
    parser.add_argument("--tasks", type=str, default=None,
        help="comma-separated subset of the audited tasks")
    parser.add_argument("--threshold", type=float, default=0.001,
        help="per-task contamination rate that fails --verify (default 0.1%%)")
    parser.add_argument("--verify", action="store_true",
        help="read the existing report and fail on any rate over --threshold")
    args = parser.parse_args()

    if args.verify:
        if not args.out.is_file():
            print(f"no report at {args.out} — run without --verify first",
                file=sys.stderr)
            return 1
        report = json.loads(args.out.read_text(encoding="utf-8"))
        over = {task: rate for task, rate in report["overall"].items()
            if rate > args.threshold}
        for task, rate in sorted(report["overall"].items()):
            flag = "  OVER" if task in over else ""
            print(f"  {task:16s} {rate * 100:7.4f}%{flag}")
        if over:
            print(f"\n{len(over)} task(s) over {args.threshold * 100:.2f}% — "
                  f"footnote them in the model card or drop them from the "
                  f"reported set.", file=sys.stderr)
            return 1
        print(f"\nall tasks under {args.threshold * 100:.2f}%")
        return 0

    tasks = args.tasks.split(",") if args.tasks else list(TEST_SETS)
    unknown = [t for t in tasks if t not in TEST_SETS]
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")

    print("building test-set 13-grams")
    per_task = build_test_ngrams(tasks)

    mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
    weights = {s["name"]: s["weight"] for s in mixture["sources"]}
    per_source: dict[str, dict[str, float]] = {}
    for source in mixture["sources"]:
        print(f"auditing {source['name']} ({args.docs_per_source:,} docs)")
        per_source[source["name"]] = audit_source(source, per_task,
            args.docs_per_source)

    # Weight by mixture probability: a hit in a source the model sees 34% of the
    # time matters more than the same hit in a 0.2% source.
    overall = {
        task: sum(weights[name] * rates[task] for name, rates in per_source.items())
        for task in per_task
    }

    args.out.write_text(json.dumps({
        "schema_version": 1,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "method": f"{NGRAM}-gram blake2b-64, lowercase alphanumeric",
        "docs_per_source": args.docs_per_source,
        "detection_floor": 1.0 / args.docs_per_source,
        "threshold": args.threshold,
        "overall": {k: round(v, 8) for k, v in overall.items()},
        "per_source": {name: {k: round(v, 8) for k, v in rates.items()}
            for name, rates in per_source.items()},
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nweighted contamination rate (floor {1 / args.docs_per_source:.2e}):")
    for task, rate in sorted(overall.items()):
        flag = "  OVER" if rate > args.threshold else ""
        print(f"  {task:16s} {rate * 100:7.4f}%{flag}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
