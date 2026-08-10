"""Tokenize the mixture into flat uint16 shards, plus a held-out split.

Pre-tokenizing to local .bin files (rather than streaming from the Hub during
training) removes the network from a 3-month run's failure modes and makes the
data cursor a plain integer offset, which is what exact resume needs.

    python apex2/prepare_data.py --source fineweb-edu
    python apex2/prepare_data.py --all          # every source at its mixture weight

Per-source target is ``--budget * weight``; the mixture's ``target_tokens``
fields are documentation and are not read here.

Output layout:
    data/tokens/<source>/shard-00000.bin      uint16, little-endian
    data/tokens/<source>/val.bin              held out, never in index.json
    data/tokens/<source>/index.json

The holdout is the FIRST ``--holdout-tokens`` tokens of each source's stream,
carved off before any training shard is written. Taking it from the front makes
disjointness structural rather than something to verify: PackedCorpus only ever
reads the files listed in index["shards"], and val.bin is not one of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SHARD_TOKENS = 256_000_000  # 512 MB per shard as uint16
DEFAULT_HOLDOUT_TOKENS = 2_000_000


# Size caps exist because one pathological repo row (generated code, data
# dumps checked in as source) joined into a single doc and fed to
# encode_batch drove RSS to 227 GB and an OOM kill 43.8B tokens into
# stack-v3. Files past 1M chars are noise for training anyway (StarCoder
# caps near this size); 10M chars bounds the whole repo document.
MAX_FILE_CHARS = 1_000_000
MAX_DOC_CHARS = 10_000_000


def record_text(record: dict, source: dict) -> str | None:
    """Extract one training document from a stream record.

    ``"format": "stack_repo"`` sources (stack-v3) are repo-level rows whose
    text lives in ``files[].content`` — there is no top-level text field, so a
    plain ``text_field`` read silently skips every record. Emit one document
    per repo in the StarCoder repo-context layout the tokenizer's specials
    were reserved for; vendored files carry near-zero training signal.
    """
    if source.get("format") == "stack_repo":
        parts = [f"<|repo_name|>{record.get('repo_path', '')}"]
        total = 0
        for file in record.get("files") or []:
            content = file.get("content")
            if not content or file.get("is_vendor") or len(content) > MAX_FILE_CHARS:
                continue
            parts.append(f"<|file_sep|>{file.get('file_path', '')}\n{content}")
            total += len(content)
            if total >= MAX_DOC_CHARS:
                break
        return "\n".join(parts) if len(parts) > 1 else None
    return record.get(source["text_field"])

# A source that ends more than this fraction short of target silently changes
# the mixture the model actually sees. 2% of a 90B source is 1.8B tokens.
SHORTFALL_TOLERANCE = 0.02


def tokenize_source(
    source: dict,
    out_root: Path,
    target_tokens: int,
    tokenizer_path: Path,
    eos_id: int,
    holdout_tokens: int = DEFAULT_HOLDOUT_TOKENS,
) -> dict:
    """Returns the written index. ``complete`` is False if the stream ran dry."""
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    name = source["name"]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(source["hf_id"], source["hf_config"], split="train",
        streaming=True)

    buffer = np.empty(SHARD_TOKENS, dtype=np.uint16)
    holdout = np.empty(holdout_tokens, dtype=np.uint16)
    holdout_filled = 0
    filled = 0
    shard_index = 0
    written_total = 0
    shards: list[dict] = []
    batch: list[str] = []
    batch_chars = 0

    def flush_shard() -> None:
        nonlocal filled, shard_index
        if filled == 0:
            return
        path = out_dir / f"shard-{shard_index:05d}.bin"
        buffer[:filled].tofile(path)
        shards.append({"path": path.name, "tokens": int(filled)})
        print(f"  wrote {path.name}  {filled:,} tokens  (total {written_total:,})",
            flush=True)
        filled = 0
        shard_index += 1

    def absorb(ids: list[int]) -> None:
        nonlocal filled, written_total, holdout_filled
        array = np.asarray(ids, dtype=np.uint16)
        position = 0
        if holdout_filled < holdout_tokens:
            take = min(holdout_tokens - holdout_filled, array.size)
            holdout[holdout_filled : holdout_filled + take] = array[:take]
            holdout_filled += take
            position = take
        while position < array.size:
            space = SHARD_TOKENS - filled
            take = min(space, array.size - position)
            buffer[filled : filled + take] = array[position : position + take]
            filled += take
            position += take
            written_total += take
            if filled == SHARD_TOKENS:
                flush_shard()

    exhausted = True
    for record in dataset:
        text = record_text(record, source)
        if not text:
            continue
        batch.append(text)
        batch_chars += len(text)
        # Char-bounded flush: 1000 capped repo docs can still be 10 GB of
        # strings, and encode_batch roughly quadruples peak memory.
        if len(batch) >= 1000 or batch_chars >= 100_000_000:
            for encoding in tokenizer.encode_batch(batch):
                absorb(encoding.ids + [eos_id])
            batch.clear()
            batch_chars = 0
            if written_total >= target_tokens:
                exhausted = False
                break
    if batch and written_total < target_tokens:
        for encoding in tokenizer.encode_batch(batch):
            absorb(encoding.ids + [eos_id])
    flush_shard()

    if holdout_filled:
        holdout[:holdout_filled].tofile(out_dir / "val.bin")

    ratio = written_total / target_tokens if target_tokens else 1.0
    complete = ratio >= 1.0 - SHORTFALL_TOLERANCE
    index = {
        "name": name,
        "hf_id": source["hf_id"],
        "hf_config": source["hf_config"],
        "target_tokens": int(target_tokens),
        "total_tokens": written_total,
        "holdout_tokens": int(holdout_filled),
        "holdout_path": "val.bin" if holdout_filled else None,
        "stream_exhausted": exhausted,
        "complete": complete,
        "shards": shards,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n",
        encoding="utf-8")
    print(f"{name}: {written_total:,} tokens in {len(shards)} shards "
          f"({ratio * 100:.1f}% of target) + {holdout_filled:,} held out")
    if not complete:
        # The mixture is a set of sampling probabilities; a source that comes up
        # short does not slow down — PackedCorpus just wraps around and epochs it
        # harder, so the failure is invisible in the loss curve.
        print(f"  SHORT by {target_tokens - written_total:,} tokens "
              f"({'stream exhausted' if exhausted else 'stopped early'}) — "
              f"effective epochs {1 / max(ratio, 1e-9):.2f}x", file=sys.stderr)
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, default=Path("apex2/mixture.json"))
    parser.add_argument("--tokenizer", type=Path,
        default=Path("tokenizer/apex2/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("data/tokens"))
    parser.add_argument("--budget", type=float, default=450e9,
        help="total pretrain token budget; per-source target = budget * weight")
    parser.add_argument("--holdout-tokens", type=int, default=DEFAULT_HOLDOUT_TOKENS,
        help="tokens carved off the front of each source for validate.py")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--allow-short", action="store_true",
        help="exit 0 even if a source ended short of its target")
    args = parser.parse_args()

    mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
    specials = json.loads(
        (args.tokenizer.parent / "special_tokens.json").read_text(encoding="utf-8"))
    eos_id = specials["<|endoftext|>"]

    selected = [s for s in mixture["sources"]
        if args.all or s["name"] == args.source]
    if not selected:
        raise SystemExit("pass --all or --source <name>")

    short: list[str] = []
    for source in selected:
        target = int(args.budget * source["weight"])
        print(f"=== {source['name']}  target {target:,} tokens ===")
        index = tokenize_source(source, args.out, target, args.tokenizer, eos_id,
            holdout_tokens=args.holdout_tokens)
        if not index["complete"]:
            short.append(source["name"])

    if short:
        print(f"\n{len(short)} source(s) short of target: {', '.join(short)}",
            file=sys.stderr)
        print("Either lower the weight in mixture.json to match what the source "
              "actually holds, or re-run with --allow-short to accept the "
              "increased epoch count.", file=sys.stderr)
        if not args.allow_short:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
