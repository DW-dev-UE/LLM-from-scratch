from __future__ import annotations

import glob
import gzip
import io
import json
import os
from typing import Iterator


def list_files(patterns: list[str] | str) -> list[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    return sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})


def task_args(task_id: int | None, num_tasks: int | None) -> tuple[int, int]:
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if num_tasks is None:
        num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    return task_id, num_tasks


def open_text(path: str, mode: str = "r"):
    if path.endswith(".zst"):
        import zstandard

        if mode == "r":
            raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"), read_across_frames=True, closefd=True)
        else:
            raw = zstandard.ZstdCompressor(level=3).stream_writer(open(path, "wb"), closefd=True)
        return io.TextIOWrapper(raw, encoding="utf-8")
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_docs(path: str) -> Iterator[dict]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
        return
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, max_bytes: int = 1 << 30):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.max_bytes = out_dir, prefix, max_bytes
        self.n, self.size, self.f = 0, 0, None
        self.paths: list[str] = []

    def write(self, doc: dict) -> None:
        if self.f is None or self.size >= self.max_bytes:
            self._roll()
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        self.f.write(line)
        self.size += len(line)

    def _roll(self) -> None:
        if self.f is not None:
            self.f.close()
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.n:05d}.jsonl.zst")
        self.f, self.size = open_text(path, "w"), 0
        self.paths.append(path)
        self.n += 1

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
