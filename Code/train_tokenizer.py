"""Apex-2 tokenizer: 65,536 byte-level BPE with FIM + repo-context specials.

Trained from scratch on a sample of the pretrain mixture — no external
pretrained tokenizer is used.

    python apex2/train_tokenizer.py --out tokenizer/apex2 --sample-docs 4000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE

VOCAB_SIZE = 65_536

# Reserved ids 0..15. StarCoder-style FIM + repo context, plus spare slots so
# later additions never shift existing ids (a shift would invalidate every
# tokenized shard and every checkpoint).
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<|im_start|>",
    "<|im_end|>",
    *[f"<|reserved_{i}|>" for i in range(7)],
]


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=False))
    # ByteLevel with add_prefix_space=False keeps code indentation intact;
    # use_regex=True applies the GPT-2 split pattern so digits and punctuation
    # do not merge into unbounded tokens.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # Split runs of digits into groups of at most 3 — improves arithmetic
            # far more than its token cost, and GSM8K is on the eval list.
            pre_tokenizers.Digits(individual_digits=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def iter_training_text(mixture_path: Path, sample_docs: int):
    """Stream a weighted sample of the mixture for BPE training.

    Uses prepare_data.record_text so stack_repo sources contribute their
    files[].content. NOTE: the canonical Apex-2 tokenizer (trained 2026-08-05,
    md5 998f542a...) predates this fix and saw no stack-v3 docs — its code
    compression is accordingly a bit worse. Do NOT retrain it for the current
    run: every shard and checkpoint depends on it. This fix is for the next
    tokenizer generation.
    """
    import random

    from datasets import load_dataset

    from apex2.prepare_data import record_text

    mixture = json.loads(mixture_path.read_text(encoding="utf-8"))
    sources = mixture["sources"]
    rng = random.Random(1234)
    streams = []
    weights = []
    for source in sources:
        dataset = load_dataset(source["hf_id"], source["hf_config"], split="train",
            streaming=True)
        streams.append((iter(dataset), source, source["name"]))
        weights.append(source["weight"])

    emitted = 0
    exhausted: set[int] = set()
    while emitted < sample_docs and len(exhausted) < len(streams):
        index = rng.choices(range(len(streams)), weights=weights, k=1)[0]
        if index in exhausted:
            continue
        iterator, source, _ = streams[index]
        try:
            record = next(iterator)
        except StopIteration:
            exhausted.add(index)
            continue
        text = record_text(record, source)
        if not text:
            continue
        yield text
        emitted += 1
        if emitted % 100_000 == 0:
            print(f"  sampled {emitted:,} docs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, default=Path("apex2/mixture.json"))
    parser.add_argument("--out", type=Path, default=Path("tokenizer/apex2"))
    parser.add_argument("--sample-docs", type=int, default=4_000_000)
    args = parser.parse_args()

    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2,
        show_progress=True)

    print(f"Training BPE to {VOCAB_SIZE:,} on ~{args.sample_docs:,} docs")
    tokenizer.train_from_iterator(iter_training_text(args.mixture, args.sample_docs),
        trainer=trainer, length=args.sample_docs)

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out / "tokenizer.json"))
    (args.out / "special_tokens.json").write_text(
        json.dumps({name: tokenizer.token_to_id(name) for name in SPECIAL_TOKENS},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    size = tokenizer.get_vocab_size()
    print(f"Saved {args.out}/tokenizer.json  vocab_size={size:,}")
    if size != VOCAB_SIZE:
        # uint16 shards require every id < 65536; a larger vocab silently wraps.
        raise SystemExit(f"vocab_size {size} != {VOCAB_SIZE}; adjust before tokenizing")


if __name__ == "__main__":
    main()
