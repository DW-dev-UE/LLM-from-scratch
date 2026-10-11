from __future__ import annotations

import glob
import math
import zlib

import numpy as np
import torch


def positions_from_tokens(tokens: np.ndarray, eos_id: int) -> np.ndarray:
    S = tokens.shape[0]
    is_start = np.zeros(S, dtype=bool)
    is_start[0] = True
    is_start[1:] = tokens[:-1] == eos_id
    idx = np.arange(S)
    return idx - np.maximum.accumulate(np.where(is_start, idx, 0))


class TokenStream:
    def __init__(self, pattern: str):
        self.glob = pattern
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(pattern)
        self.shards = [np.memmap(p, dtype=np.uint32, mode="r") for p in paths]
        self.offsets = np.cumsum([0] + [len(s) for s in self.shards])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def read(self, start: int, length: int) -> np.ndarray:
        out = np.empty(length, dtype=np.int64)
        i = int(np.searchsorted(self.offsets, start, side="right") - 1)
        pos, filled = start - int(self.offsets[i]), 0
        while filled < length:
            piece = self.shards[i][pos : pos + length - filled]
            out[filled : filled + len(piece)] = piece
            filled += len(piece)
            i, pos = i + 1, 0
        return out


def _coprime(n: int, rng) -> int:
    if n <= 2:
        return 1
    while True:
        a = int(rng.integers(n // 3, n)) | 1
        if math.gcd(a, n) == 1:
            return a


class PackedBatches:
    def __init__(self, stream: TokenStream, seq_len: int, micro_bs: int, eos_id: int, rank: int, world: int, seed: int = 0, stride: int = 1):
        self.stream, self.seq_len, self.micro_bs, self.eos_id = stream, seq_len, micro_bs, eos_id
        self.rank, self.world, self.seed, self.stride = rank, world, seed, stride
        self.n_windows = len(stream) // (seq_len + 1) // stride
        if self.n_windows < 1:
            raise ValueError("not enough tokens for one window")
        self._perm: tuple[int, int, int] | None = None
        self.micro_step = 0
        self.start_row, self.offset = 0, 0
        self.last_domains = ["all"] * micro_bs

    @property
    def spec(self) -> str:
        return self.stream.glob if self.stride == 1 else f"{self.stream.glob}#stride={self.stride}"

    def __iter__(self):
        return self

    def window(self, g: int) -> np.ndarray:
        epoch, i = divmod(g, self.n_windows)
        if self._perm is None or self._perm[0] != epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm = (epoch, _coprime(self.n_windows, rng), int(rng.integers(0, self.n_windows)))
        _, a, b = self._perm
        w = (a * i + b) % self.n_windows * self.stride
        return self.stream.read(w * (self.seq_len + 1), self.seq_len + 1)

    def rows_seen(self) -> int:
        return self.micro_step * self.world * self.micro_bs

    def epochs(self) -> dict[str, float]:
        return {"all": (self.offset + self.rows_seen() - self.start_row) / self.n_windows}

    def state(self) -> dict:
        row_tokens = self.seq_len + 1
        return {"specs": {"all": self.spec}, "seq_len": self.seq_len, "start_row": self.start_row, "offset_tokens": {"all": self.offset * row_tokens}, "tokens": {"all": (self.offset + self.rows_seen() - self.start_row) * row_tokens}}

    def resume(self, rows_seen: int, saved: dict | None) -> None:
        self.micro_step = rows_seen // (self.world * self.micro_bs)
        if not saved or "offset_tokens" not in saved:
            return
        row_tokens, specs = self.seq_len + 1, self.state()["specs"]
        if saved["specs"] == specs and saved.get("seq_len") == self.seq_len:
            self.start_row, self.offset = saved["start_row"], saved["offset_tokens"]["all"] // row_tokens
        else:
            self.start_row = rows_seen
            self.offset = saved["tokens"]["all"] // row_tokens if saved["specs"] == specs else 0

    def __next__(self) -> dict[str, torch.Tensor]:
        base = (self.micro_step * self.world + self.rank) * self.micro_bs
        self.micro_step += 1
        return pack_rows([self.window(self.offset + base + j - self.start_row) for j in range(self.micro_bs)], self.eos_id)


def pack_rows(rows: list[np.ndarray], eos_id: int) -> dict[str, torch.Tensor]:
    rows = np.stack(rows)
    inp, lab = rows[:, :-1], rows[:, 1:]
    pos = np.stack([positions_from_tokens(r, eos_id) for r in inp])
    return {"input_ids": torch.from_numpy(np.ascontiguousarray(inp)), "labels": torch.from_numpy(np.ascontiguousarray(lab)), "position_ids": torch.from_numpy(pos)}


def domain_schedule(weights, seed: int = 0, min_len: int = 1000) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if (w <= 0).any():
        raise ValueError("domain weights must be positive")
    w = w / w.sum()
    M = max(min_len, math.ceil(1 / w.min()))
    raw = w * M
    counts = np.floor(raw).astype(np.int64)
    counts[np.argsort(-(raw - counts), kind="stable")[: M - int(counts.sum())]] += 1
    pattern = np.repeat(np.arange(len(w)), counts)
    np.random.default_rng(seed).shuffle(pattern)
    return pattern


class MixtureBatches:
    def __init__(self, domains: dict[str, tuple[float, TokenStream]], seq_len: int, micro_bs: int, eos_id: int, rank: int, world: int, seed: int = 0, strides: dict[str, int] | None = None):
        names = sorted(domains)
        self.names, self.seq_len = names, seq_len
        strides = strides or {}
        self.parts = [PackedBatches(domains[n][1], seq_len, 1, eos_id, 0, 1, (seed << 32) + zlib.crc32(n.encode()), strides.get(n, 1)) for n in names]
        self.last_domains: list[str] = []
        self.pattern = domain_schedule([domains[n][0] for n in names], seed)
        self.counts = np.bincount(self.pattern, minlength=len(names))
        self.slot_rank = np.zeros(len(self.pattern), dtype=np.int64)
        for d in range(len(names)):
            slots = np.flatnonzero(self.pattern == d)
            self.slot_rank[slots] = np.arange(len(slots))
        self.micro_bs, self.eos_id, self.rank, self.world, self.seed = micro_bs, eos_id, rank, world, seed
        self.micro_step = 0
        self.start_row = 0
        self.offsets = np.zeros(len(names), dtype=np.int64)

    def locate(self, g: int) -> tuple[int, int]:
        period, slot = divmod(g - self.start_row, len(self.pattern))
        d = int(self.pattern[slot])
        return d, int(self.offsets[d]) + period * int(self.counts[d]) + int(self.slot_rank[slot])

    def rows_seen(self) -> int:
        return self.micro_step * self.world * self.micro_bs

    def domain_rows(self) -> np.ndarray:
        period, rest = divmod(self.rows_seen() - self.start_row, len(self.pattern))
        return self.offsets + period * self.counts + np.bincount(self.pattern[:rest], minlength=len(self.names))

    def epochs(self) -> dict[str, float]:
        return {n: float(r) / p.n_windows for n, r, p in zip(self.names, self.domain_rows(), self.parts)}

    def state(self) -> dict:
        row_tokens = self.seq_len + 1
        return {"specs": {n: p.spec for n, p in zip(self.names, self.parts)}, "seq_len": self.seq_len, "start_row": self.start_row,
                "offset_tokens": {n: int(o) * row_tokens for n, o in zip(self.names, self.offsets)}, "tokens": {n: int(r) * row_tokens for n, r in zip(self.names, self.domain_rows())}}

    def resume(self, rows_seen: int, saved: dict | None) -> None:
        self.micro_step = rows_seen // (self.world * self.micro_bs)
        if not saved or "offset_tokens" not in saved:
            return
        row_tokens, specs = self.seq_len + 1, self.state()["specs"]
        if saved["specs"] == specs and saved.get("seq_len") == self.seq_len:
            self.start_row = saved["start_row"]
            self.offsets = np.array([saved["offset_tokens"].get(n, 0) // row_tokens for n in self.names])
        else:
            self.start_row = rows_seen
            self.offsets = np.array([saved["tokens"].get(n, 0) // row_tokens if saved["specs"].get(n) == specs[n] else 0 for n in self.names])

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        base = (self.micro_step * self.world + self.rank) * self.micro_bs
        self.micro_step += 1
        rows = [self.locate(base + j) for j in range(self.micro_bs)]
        self.last_domains = [self.names[d] for d, _ in rows]
        return pack_rows([self.parts[d].window(i) for d, i in rows], self.eos_id)


def pool_strides(items: list[str]) -> dict[str, int]:
    out = {}
    for item in items:
        name, frac = item.split("=", 1)
        if not 0 < float(frac) <= 1:
            raise ValueError(f"pool fraction must be in (0, 1]: {item}")
        out[name] = max(1, round(1 / float(frac)))
    return out


def build_batches(specs: list[str], seq_len: int, micro_bs: int, eos_id: int, rank: int, world: int, seed: int = 0, strides: dict[str, int] | None = None):
    strides = strides or {}
    if len(specs) == 1 and "=" not in specs[0]:
        return PackedBatches(TokenStream(specs[0]), seq_len, micro_bs, eos_id, rank, world, seed, strides.get("all", 1))
    domains = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        weight, pattern = rest.split(":", 1)
        domains[name] = (float(weight), TokenStream(pattern))
    unknown = set(strides) - set(domains)
    if unknown:
        raise KeyError(f"pool for unknown domains: {sorted(unknown)}")
    return MixtureBatches(domains, seq_len, micro_bs, eos_id, rank, world, seed, strides)


def build_val(specs: list[str], seq_len: int, micro_bs: int, eos_id: int, rank: int, world: int, seed: int = 1) -> dict[str, PackedBatches]:
    if len(specs) == 1 and "=" not in specs[0]:
        return {"val": PackedBatches(TokenStream(specs[0]), seq_len, micro_bs, eos_id, rank, world, seed)}
    out = {}
    for spec in specs:
        name, pattern = spec.split("=", 1)
        out[name] = PackedBatches(TokenStream(pattern), seq_len, micro_bs, eos_id, rank, world, seed)
    return out


def synthetic_batch(micro_bs: int, seq_len: int, vocab: int, doc_len: int | None = None, generator: torch.Generator | None = None, device: str = "cuda") -> dict[str, torch.Tensor]:
    ids = torch.randint(vocab, (micro_bs, seq_len + 1), device=device, generator=generator)
    pos = torch.arange(seq_len, device=device).repeat(micro_bs, 1)
    if doc_len:
        pos = pos % doc_len
    return {"input_ids": ids[:, :-1].contiguous(), "labels": ids[:, 1:].contiguous(), "position_ids": pos}


def to_device(batch: dict[str, torch.Tensor], device: str = "cuda") -> dict[str, torch.Tensor]:
    return {k: v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
