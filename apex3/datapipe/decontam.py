from __future__ import annotations

import re
import zlib

import numpy as np

_WORD = re.compile(r"\w+")
_MUL = np.uint64(1099511628211)


def ngram_hashes(text: str, n: int = 13) -> np.ndarray:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    wh = np.fromiter((zlib.crc32(w.encode()) for w in words), dtype=np.uint64, count=len(words))
    m = len(words) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    for j in range(n):
        h = h * _MUL + wh[j : j + m]
    return h


def build_ngram_table(texts, n: int = 13) -> np.ndarray:
    parts = [ngram_hashes(t, n) for t in texts]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)


def token_ngram_hashes(ids: np.ndarray, n: int = 13) -> np.ndarray:
    m = len(ids) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    t = ids.astype(np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    for j in range(n):
        np.multiply(h, _MUL, out=h)
        np.add(h, t[j : j + m], out=h)
    return h


def token_ngram_table(token_lists, n: int = 13, max_texts: int = 3, alpha: np.ndarray | None = None, min_alpha: int = 0) -> np.ndarray:
    def windows(ids):
        ids = np.asarray(ids, dtype=np.uint32)
        h = token_ngram_hashes(ids, n)
        if alpha is not None and min_alpha and h.size:
            cs = np.concatenate([[0], np.cumsum(alpha[ids], dtype=np.int64)])
            h = h[cs[n:] - cs[:-n] >= min_alpha]
        return np.unique(h)

    per = [windows(ids) for ids in token_lists]
    per = [p for p in per if p.size]
    if not per:
        return np.empty(0, dtype=np.uint64)
    uniq, counts = np.unique(np.concatenate(per), return_counts=True)
    return uniq[counts <= max_texts]


def contaminated_docs(arr: np.ndarray, eos_id: int, table: np.ndarray, n: int = 13, chunk: int = 1 << 26) -> tuple[np.ndarray, np.ndarray]:
    ends = np.flatnonzero(arr == eos_id)
    if not table.size or not ends.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)
    pos, hashes = [], []
    for s in range(0, len(arr), chunk):
        h = token_ngram_hashes(arr[s : min(len(arr), s + chunk + n - 1)], n)
        idx = np.minimum(np.searchsorted(table, h), table.size - 1)
        hit = np.flatnonzero(table[idx] == h)
        pos.append(hit + s)
        hashes.append(h[hit])
    pos, hashes = np.concatenate(pos), np.concatenate(hashes)
    first, last = np.searchsorted(ends, pos), np.searchsorted(ends, pos + n - 1)
    inside = (first == last) & (first < len(ends))
    return np.unique(first[inside]), hashes[inside]


def drop_docs(arr: np.ndarray, eos_id: int, drop: np.ndarray) -> np.ndarray:
    ends = np.flatnonzero(arr == eos_id)
    lengths = np.diff(np.concatenate([[-1], ends]))
    keep = np.ones(len(ends), dtype=bool)
    keep[drop] = False
    if not len(ends):
        return arr
    body = arr[: int(ends[-1]) + 1][np.repeat(keep, lengths)]
    return np.concatenate([body, arr[int(ends[-1]) + 1 :]])


class Decontaminator:
    def __init__(self, ngram_file: str | None = None, repo_blocklist: str | None = None, n: int = 13):
        self.n = n
        self.table = np.unique(np.load(ngram_file)) if ngram_file else np.empty(0, dtype=np.uint64)
        self.repos: set[str] = set()
        if repo_blocklist:
            with open(repo_blocklist, encoding="utf-8") as f:
                self.repos = {ln.strip().lower() for ln in f if ln.strip() and not ln.startswith("#")}

    def reason(self, doc: dict) -> str | None:
        meta = doc.get("meta") or {}
        repo = meta.get("repo_name") or meta.get("max_stars_repo_name")
        if repo and repo.lower() in self.repos:
            return "blocked_repo"
        if self.table.size:
            h = ngram_hashes(doc["text"], self.n)
            if h.size:
                idx = np.minimum(np.searchsorted(self.table, h), self.table.size - 1)
                if (self.table[idx] == h).any():
                    return "contaminated"
        return None
