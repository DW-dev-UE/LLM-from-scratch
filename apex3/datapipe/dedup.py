from __future__ import annotations

import glob
import hashlib
import os
import re
import zlib
from typing import Callable

import numpy as np

from .io import ShardWriter, read_docs

REC = np.dtype([("h", "<u8"), ("ref", "<u8")])
_TOKEN = re.compile(r"\w+|[^\w\s]")
_MERSENNE = np.uint64((1 << 61) - 1)
_MUL = np.uint64(1099511628211)


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


class MinHasher:
    def __init__(self, n_bands: int = 14, rows: int = 8, ngram: int = 5, seed: int = 1):
        self.n_bands, self.rows, self.ngram = n_bands, rows, ngram
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 1 << 32, size=n_bands * rows, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 32, size=n_bands * rows, dtype=np.uint64)

    def shingles(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower())
        if not toks:
            return np.zeros(1, dtype=np.uint64)
        wh = np.fromiter((zlib.crc32(t.encode()) for t in toks), dtype=np.uint64, count=len(toks))
        n = min(self.ngram, len(toks))
        m = len(toks) - n + 1
        h = np.zeros(m, dtype=np.uint64)
        for j in range(n):
            h = h * _MUL + wh[j : j + m]
        return np.unique(h & np.uint64(0xFFFFFFFF))

    def bands(self, text: str) -> list[int]:
        x = self.shingles(text)
        sig = np.full(self.a.size, _MERSENNE, dtype=np.uint64)
        for s in range(0, x.size, 4096):
            chunk = x[s : s + 4096]
            sig = np.minimum(sig, ((self.a[:, None] * chunk[None, :] + self.b[:, None]) % _MERSENNE).min(axis=1))
        return [_h64(row.tobytes()) for row in sig.reshape(self.n_bands, self.rows)]


def stage_signatures(files: list[str], shard_ids: list[int], work: str, task: int, hasher: MinHasher, n_parts: int = 16) -> None:
    os.makedirs(os.path.join(work, "sig"), exist_ok=True)
    n_cols = hasher.n_bands + 1
    outs = {}
    try:
        for shard in shard_ids:
            hashes, refs = [], []
            for line, doc in enumerate(read_docs(files[shard])):
                hashes.append(hasher.bands(doc["text"]) + [_h64(doc["text"].encode())])
                refs.append((shard << 32) | line)
            if not refs:
                continue
            H, R = np.array(hashes, dtype=np.uint64), np.array(refs, dtype=np.uint64)
            for b in range(n_cols):
                col = H[:, b]
                parts = col % np.uint64(n_parts)
                for p in range(n_parts):
                    sel = parts == p
                    if sel.any():
                        rec = np.empty(int(sel.sum()), dtype=REC)
                        rec["h"], rec["ref"] = col[sel], R[sel]
                        key = (b, p)
                        if key not in outs:
                            outs[key] = open(os.path.join(work, "sig", f"b{b:02d}_p{p:03d}_t{task:05d}.bin"), "wb")
                        rec.tofile(outs[key])
    finally:
        for f in outs.values():
            f.close()


def stage_edges(work: str, band: int, part: int) -> None:
    paths = sorted(glob.glob(os.path.join(work, "sig", f"b{band:02d}_p{part:03d}_t*.bin")))
    os.makedirs(os.path.join(work, "edges"), exist_ok=True)
    out = os.path.join(work, "edges", f"b{band:02d}_p{part:03d}.bin")
    recs = np.concatenate([np.fromfile(p, dtype=REC) for p in paths]) if paths else np.empty(0, dtype=REC)
    if recs.size < 2:
        np.empty(0, dtype=np.uint64).tofile(out)
        return
    order = np.lexsort((recs["ref"], recs["h"]))
    h, ref = recs["h"][order], recs["ref"][order]
    is_start = np.concatenate(([True], h[1:] != h[:-1]))
    first = ref[np.flatnonzero(is_start)][np.cumsum(is_start) - 1]
    dup = ref != first
    np.stack([first[dup], ref[dup]], axis=1).astype(np.uint64).tofile(out)


def unique_pairs(edges: np.ndarray) -> np.ndarray:
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    keep = np.ones(len(edges), dtype=bool)
    keep[1:] = (edges[1:] != edges[:-1]).any(axis=1)
    return edges[keep]


def stage_cluster(work: str, compact_rows: int = 1 << 27) -> int:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    paths = sorted(glob.glob(os.path.join(work, "edges", "*.bin")))
    os.makedirs(os.path.join(work, "remove"), exist_ok=True)
    edges, pending, rows = np.empty((0, 2), dtype=np.uint64), [], 0
    for p in paths:
        pending.append(np.fromfile(p, dtype=np.uint64).reshape(-1, 2))
        rows += len(pending[-1])
        if rows >= compact_rows:
            edges, pending, rows = unique_pairs(np.concatenate([edges, *pending])), [], 0
    edges = unique_pairs(np.concatenate([edges, *pending]))
    if edges.size == 0:
        return 0
    nodes, inv = np.unique(edges.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(inv), dtype=np.int8), (inv[:, 0], inv[:, 1])), shape=(nodes.size, nodes.size))
    _, labels = connected_components(graph, directed=False)
    keep = np.zeros(nodes.size, dtype=bool)
    keep[np.unique(labels, return_index=True)[1]] = True
    removed = nodes[~keep]
    shards = removed >> np.uint64(32)
    for shard in np.unique(shards):
        lines = np.sort((removed[shards == shard] & np.uint64(0xFFFFFFFF)).astype(np.int64))
        np.save(os.path.join(work, "remove", f"{int(shard):06d}.npy"), lines)
    return int(removed.size)


def stage_filter(files: list[str], shard: int, work: str, out_dir: str, reject: Callable[[dict], str | None] | None = None) -> tuple[int, int, int]:
    path = os.path.join(work, "remove", f"{shard:06d}.npy")
    removed = set(np.load(path).tolist()) if os.path.exists(path) else set()
    stem = f"{shard:06d}_" + os.path.basename(files[shard]).split(".")[0]
    kept = rejected = 0
    with ShardWriter(out_dir, stem, max_bytes=1 << 40) as w:
        for line, doc in enumerate(read_docs(files[shard])):
            if line in removed:
                continue
            if reject is not None and reject(doc):
                rejected += 1
                continue
            w.write(doc)
            kept += 1
    return kept, len(removed), rejected
