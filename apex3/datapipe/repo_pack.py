from __future__ import annotations

import hashlib
import os
from collections import defaultdict

from .io import ShardWriter, list_files, read_docs


def _bucket(repo: str, n: int) -> int:
    return int.from_bytes(hashlib.blake2b(repo.encode(), digest_size=4).digest(), "little") % n


def partition(paths: list[str], out_dir: str, n_buckets: int, task: int) -> None:
    writers: dict[int, ShardWriter] = {}
    try:
        for path in paths:
            for doc in read_docs(path):
                repo = (doc.get("meta") or {}).get("repo_name")
                if not repo:
                    continue
                b = _bucket(repo, n_buckets)
                if b not in writers:
                    writers[b] = ShardWriter(os.path.join(out_dir, f"bucket_{b:05d}"), f"t{task:05d}", 1 << 40)
                writers[b].write(doc)
    finally:
        for w in writers.values():
            w.close()


def _file_order(path: str) -> tuple[int, str]:
    name = path.rsplit("/", 1)[-1].lower()
    return (0 if name.startswith("readme") else 1, path)


def pack(bucket_dir: str, out_dir: str, max_chars: int = 1_000_000, min_files: int = 2) -> int:
    repos: dict[str, list[dict]] = defaultdict(list)
    for path in list_files(os.path.join(bucket_dir, "*.jsonl.zst")):
        for doc in read_docs(path):
            repos[doc["meta"]["repo_name"]].append(doc)
    n_out = 0
    with ShardWriter(out_dir, os.path.basename(bucket_dir.rstrip("/"))) as w:
        for repo, docs in repos.items():
            if len(docs) < min_files:
                continue
            docs.sort(key=lambda d: _file_order(d["meta"].get("path") or ""))
            chunk, size = [], 0
            for d in docs + [None]:
                if d is None or (chunk and size + len(d["text"]) > max_chars):
                    if chunk:
                        w.write({"id": f"repo/{repo}/{n_out}", "text": "", "source": "repo_pack", "domain": "code", "profile": "code", "meta": {"repo_name": repo}, "files": chunk})
                        n_out += 1
                    chunk, size = [], 0
                if d is not None:
                    chunk.append({"path": d["meta"].get("path") or "", "text": d["text"]})
                    size += len(d["text"])
    return n_out
