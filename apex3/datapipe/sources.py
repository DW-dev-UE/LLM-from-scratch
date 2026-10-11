from __future__ import annotations

import gzip
import json
import os
import random
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from .io import list_files, read_docs


def load_source_configs(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {s["name"]: s for s in json.load(f)["sources"]}


def _doc(cfg: dict, text: str, key: str, meta: dict | None = None) -> dict:
    return {"id": f"{cfg['name']}/{key}", "text": text, "source": cfg["name"], "domain": cfg["domain"], "profile": cfg.get("profile", "web"), "meta": meta or {}}


def mix_streams(streams: list, weights: list[float], seed: int = 0):
    rng = random.Random(seed)
    its, w = [iter(s) for s in streams], [float(x) for x in weights]
    while its:
        i = rng.choices(range(len(its)), weights=w)[0]
        try:
            yield next(its[i])
        except StopIteration:
            del its[i], w[i]


def _hf_rows(cfg: dict, task_id: int, num_tasks: int):
    from datasets import load_dataset, load_dataset_builder
    from datasets.distributed import split_dataset_by_node

    configs = cfg.get("config")
    configs = configs if isinstance(configs, list) else [configs]
    split = cfg.get("split", "train")
    parts = [split_dataset_by_node(load_dataset(cfg["path"], c, data_dir=cfg.get("data_dir"), split=split, streaming=True), rank=task_id, world_size=num_tasks) for c in configs]
    if len(parts) == 1:
        return parts[0]
    weights = []
    for c in configs:
        splits = load_dataset_builder(cfg["path"], c).info.splits
        weights.append(splits[split].num_examples if splits and split in splits else 1)
    return mix_streams(parts, weights, seed=task_id)


def _local_rows(cfg: dict, task_id: int, num_tasks: int):
    files = list_files(cfg["glob"])
    random.Random(0).shuffle(files)
    for path in files[task_id::num_tasks]:
        yield from read_docs(path)


DATA_SUFFIXES = (".parquet", ".jsonl", ".jsonl.gz", ".json.gz", ".jsonl.zst")


def _downloaded_rows(local_dir: str, task_id: int, num_tasks: int):
    files = [f for f in list_files(os.path.join(local_dir, "**", "*")) if f.endswith(DATA_SUFFIXES)]
    if not files:
        raise FileNotFoundError(f"no data files under {local_dir}")
    random.Random(0).shuffle(files)
    for path in files[task_id::num_tasks]:
        yield from read_docs(path)


def _rows(cfg: dict, task_id: int, num_tasks: int, local_dir: str | None):
    if local_dir:
        return _downloaded_rows(local_dir, task_id, num_tasks)
    if cfg["type"] == "local" or "path" not in cfg:
        return _local_rows(cfg, task_id, num_tasks)
    return _hf_rows(cfg, task_id, num_tasks)


def row_filter(spec: dict | None):
    if not spec:
        return lambda row: True

    def ok(row: dict) -> bool:
        for key, cond in spec.items():
            v = row.get(key)
            if isinstance(cond, dict):
                if v is None or ("min" in cond and v < cond["min"]) or ("max" in cond and v > cond["max"]):
                    return False
            elif isinstance(cond, list):
                if v not in cond:
                    return False
            elif v != cond:
                return False
        return True

    return ok


def source_files(cfg: dict, task_id: int, num_tasks: int, local_dir: str | None = None) -> list[str] | None:
    if local_dir:
        files = [f for f in list_files(os.path.join(local_dir, "**", "*")) if f.endswith(DATA_SUFFIXES)]
    elif cfg["type"] == "local" or "path" not in cfg:
        files = list_files(cfg["glob"])
    else:
        return None
    random.Random(0).shuffle(files)
    return files[task_id::num_tasks]


def iter_file(cfg: dict, path: str, key: str) -> Iterator[dict]:
    rows = read_docs(path)
    yield from (_stack_v2(cfg, rows) if cfg["type"] == "stack_v2" else _docs(cfg, rows, key))


def iter_source(cfg: dict, task_id: int = 0, num_tasks: int = 1, local_dir: str | None = None) -> Iterator[dict]:
    rows = _rows(cfg, task_id, num_tasks, local_dir)
    yield from (_stack_v2(cfg, rows) if cfg["type"] == "stack_v2" else _docs(cfg, rows, str(task_id)))


_SC2_TAG = re.compile(r"<(issue_start|issue_comment|issue_closed|jupyter_start|jupyter_text|jupyter_code|jupyter_output|jupyter_script|empty_output|commit_before|commit_msg|commit_after|reponame|filename|gh_stars)>")
_SC_META = ("reponame", "filename", "gh_stars")


def starcoder2_text(text: str) -> str:
    parts = _SC2_TAG.split(text)
    head = parts[0].strip("\n").rstrip()
    out = [head] if head.strip() else []
    for tag, body in zip(parts[1::2], parts[2::2]):
        if tag in _SC_META:
            body = body.split("\n", 1)[1] if "\n" in body else ""
            if body.strip():
                out.append(body.strip("\n").rstrip())
            continue
        body = body.strip()
        if not body or tag in ("jupyter_output", "empty_output"):
            continue
        if tag == "jupyter_code":
            body = f"```python\n{body}\n```"
        elif tag == "commit_before":
            body = f"Code before the change:\n```\n{body}\n```"
        elif tag == "commit_msg":
            body = f"Commit message: {body}"
        elif tag == "commit_after":
            body = f"Code after the change:\n```\n{body}\n```"
        out.append(body)
    return "\n\n".join(out)


def chat_text(row: dict, cfg: dict) -> str | None:
    if cfg.get("format") == "qa":
        prompt, response = row.get(cfg["prompt_field"]), row.get(cfg["response_field"])
        if not prompt or not response or prompt.strip() in ("", "-") or not response.strip():
            return None
        return f"Question:\n{prompt.strip()}\n\nAnswer:\n{response.strip()}"
    turns = [f"{(m.get('role') or 'user').capitalize()}:\n{m['content'].strip()}" for m in row.get(cfg.get("messages_field", "messages")) or [] if (m.get("content") or "").strip()]
    return "\n\n".join(turns) if len(turns) > 1 else None


def _docs(cfg: dict, rows, key: str) -> Iterator[dict]:
    field, title = cfg.get("text_field", "text"), cfg.get("title_field")
    meta_fields = cfg.get("meta_fields", [])
    keep = row_filter(cfg.get("filter"))
    fmt, sample = cfg.get("format"), cfg.get("sample")
    for i, row in enumerate(rows):
        if not keep(row):
            continue
        if sample is not None and zlib.crc32(f"{key}-{i}".encode()) % 1_000_000 >= sample * 1_000_000:
            continue
        text = chat_text(row, cfg) if fmt in ("qa", "messages") else row.get(field)
        if text and fmt == "starcoder2":
            text = starcoder2_text(text)
        if not text:
            continue
        if title and row.get(title):
            text = f"{row[title]}\n\n{text}"
        yield _doc(cfg, text, f"{key}-{i}", {k: row.get(k) for k in meta_fields})


class _AnonymousBlobs:
    URL = "https://softwareheritage.s3.amazonaws.com/content/{}"

    def __init__(self, concurrency: int):
        import asyncio

        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self._open(concurrency))

    async def _open(self, concurrency: int) -> None:
        import asyncio

        import aiohttp

        self.sem = asyncio.Semaphore(concurrency)
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency), timeout=aiohttp.ClientTimeout(total=60))

    async def _get(self, blob_id: str) -> bytes | None:
        import asyncio

        async with self.sem:
            for attempt in range(5):
                try:
                    async with self.session.get(self.URL.format(blob_id)) as r:
                        if r.status == 200:
                            return await r.read()
                        if r.status in (403, 404):
                            return None
                except Exception:
                    pass
                await asyncio.sleep(0.5 * 2**attempt)
            return None

    async def _all(self, blob_ids: list[str]) -> list[bytes | None]:
        import asyncio

        return await asyncio.gather(*(self._get(b) for b in blob_ids))

    def fetch(self, blob_ids: list[str]) -> list[bytes | None]:
        return self.loop.run_until_complete(self._all(blob_ids))

    def close(self) -> None:
        self.loop.run_until_complete(self.session.close())
        self.loop.close()


def _stack_v2(cfg: dict, rows) -> Iterator[dict]:
    import boto3
    from botocore.config import Config

    session = boto3.Session()
    blobs = s3 = None
    if session.get_credentials() is None:
        blobs = _AnonymousBlobs(cfg.get("download_concurrency", 128))
    else:
        s3 = session.client("s3", region_name="us-east-1", config=Config(max_pool_connections=cfg.get("download_threads", 64)))
    allowed = set(cfg.get("license_types", ["permissive"]))
    languages = set(cfg.get("languages", []))
    max_bytes = cfg.get("max_file_bytes", 1_000_000)

    def keep(r: dict) -> bool:
        return (r.get("license_type") in allowed and not r.get("is_generated") and not r.get("is_vendor") and (r.get("length_bytes") or 0) <= max_bytes and (not languages or r.get("language") in languages))

    def get(r: dict) -> bytes | None:
        try:
            return s3.get_object(Bucket="softwareheritage", Key=f"content/{r['blob_id']}")["Body"].read()
        except Exception:
            return None

    def decode(r: dict, raw: bytes | None) -> str | None:
        try:
            return gzip.decompress(raw).decode(r.get("src_encoding") or "utf-8", errors="replace") if raw else None
        except Exception:
            return None

    def emit(pool, batch):
        raws = blobs.fetch([r["blob_id"] for r in batch]) if blobs else pool.map(get, batch)
        for r, raw in zip(batch, raws):
            text = decode(r, raw)
            if text:
                meta = {"repo_name": r.get("repo_name"), "path": r.get("path"), "language": r.get("language"), "licenses": r.get("detected_licenses"), "revision_date": str(r.get("revision_date"))}
                yield _doc(cfg, text, r["blob_id"], meta)

    batch: list[dict] = []
    try:
        with ThreadPoolExecutor(cfg.get("download_threads", 64)) as pool:
            for r in rows:
                if keep(r):
                    batch.append(r)
                if len(batch) >= 1024:
                    yield from emit(pool, batch)
                    batch = []
            if batch:
                yield from emit(pool, batch)
    finally:
        if blobs:
            blobs.close()
