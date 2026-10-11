from __future__ import annotations

import json
import os
import random

import numpy as np

from ..tokens import load_tokenizer, special_ids
from .io import read_docs


def doc_bytes(doc: dict) -> int:
    if doc.get("files"):
        return sum(len(f["text"].encode()) for f in doc["files"])
    return len(doc["text"].encode())


def segments(doc: dict, ids: dict[str, int], rng: random.Random, fim_rate: float) -> list:
    eos = ids["<|endoftext|>"]
    if doc.get("files"):
        segs: list = [ids["<|repo|>"], f"{(doc.get('meta') or {}).get('repo_name', '')}\n"]
        for f in doc["files"]:
            segs += [ids["<|file|>"], f"{f['path']}\n", f["text"]]
        return segs + [eos]
    text = doc["text"]
    if doc.get("domain") == "code" and fim_rate > 0 and len(text) >= 2 and rng.random() < fim_rate:
        i, j = sorted(rng.sample(range(len(text) + 1), 2))
        return [ids["<|fim_prefix|>"], text[:i], ids["<|fim_suffix|>"], text[j:], ids["<|fim_middle|>"], text[i:j], eos]
    return [text, eos]


def encode_docs(tok, docs: list[dict], ids: dict[str, int], rng: random.Random, fim_rate: float) -> np.ndarray:
    all_segs = [segments(d, ids, rng, fim_rate) for d in docs]
    texts = [s for segs in all_segs for s in segs if isinstance(s, str)]
    encode = getattr(tok, "encode_batch_fast", tok.encode_batch)
    encoded = iter(encode(texts, add_special_tokens=False))
    pieces = [np.asarray(next(encoded).ids if isinstance(s, str) else [s], dtype=np.uint32) for segs in all_segs for s in segs]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint32)


def long_docs_file(path: str, out_path: str, eos_id: int, min_tokens: int) -> dict:
    arr = np.memmap(path, dtype=np.uint32, mode="r")
    ends = np.flatnonzero(arr == eos_id)
    starts = np.concatenate([[0], ends[:-1] + 1]) if len(ends) else np.empty(0, dtype=np.int64)
    keep = np.flatnonzero(ends - starts + 1 >= min_tokens)
    tokens = 0
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        for k in keep:
            arr[starts[k] : ends[k] + 1].tofile(f)
            tokens += int(ends[k] - starts[k] + 1)
    os.replace(tmp, out_path)
    stats = {"docs": len(keep), "tokens": tokens, "src_docs": len(ends), "src_tokens": int(len(arr))}
    src = path + ".json"
    if os.path.exists(src):
        with open(src) as f:
            stats["bytes"] = int(json.load(f).get("bytes", 0) * tokens / max(1, len(arr)))
    with open(out_path + ".json", "w") as f:
        json.dump(stats, f)
    return stats


def compress(path: str) -> str:
    import zstandard

    tmp = path + ".zst.tmp"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        zstandard.ZstdCompressor(level=3).copy_stream(src, dst)
    os.replace(tmp, path + ".zst")
    os.remove(path)
    return path + ".zst"


def tokenize_file(path: str, out_path: str, tokenizer_path: str, fim_rate: float = 0.0, seed: int = 0, batch_docs: int = 1000, batch_bytes: int = 16 << 20) -> dict:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok = load_tokenizer(tokenizer_path)
    ids = special_ids(tok)
    rng = random.Random(f"{seed}:{os.path.basename(path)}")
    stats = {"docs": 0, "tokens": 0, "bytes": 0}
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        batch: list[dict] = []
        size = 0

        def flush():
            arr = encode_docs(tok, batch, ids, rng, fim_rate)
            arr.tofile(f)
            stats["docs"] += len(batch)
            stats["tokens"] += int(arr.size)
            stats["bytes"] += size
            batch.clear()

        for doc in read_docs(path):
            batch.append(doc)
            size += doc_bytes(doc)
            if len(batch) >= batch_docs or size >= batch_bytes:
                flush()
                size = 0
        if batch:
            flush()
    os.replace(tmp, out_path)
    with open(out_path + ".json", "w") as f:
        json.dump(stats, f)
    return stats
