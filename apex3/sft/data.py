from __future__ import annotations

import bisect
import json
import math
import os

import numpy as np

from ..agent.template import VERSION

TARGET, START = 1, 2
IGNORE_INDEX = -100


def conversation_messages(conv: dict, template) -> list[dict]:
    msgs = list(conv["messages"])
    base = conv.get("system")
    if msgs and msgs[0].get("role") == "system":
        if base is not None:
            raise ValueError("conversation has both 'system' and a system message")
        base = msgs.pop(0).get("content") or ""
    system = template.system_text(base or "", conv.get("tools") or [], conv.get("skills") or "")
    return [{"role": "system", "content": system}] + msgs


def check(msgs: list[dict]) -> str | None:
    body = msgs[1:]
    if not body or body[0].get("role") != "user":
        return "first_turn_not_user"
    pending = 0
    for m in body:
        role = m.get("role")
        if role not in ("user", "assistant", "tool"):
            return "bad_role"
        content_ok = isinstance(m.get("content"), str) if role != "assistant" else isinstance(m.get("content") or "", str)
        if not content_ok or not isinstance(m.get("think") or "", str):
            return "bad_content"
        if role == "tool":
            if pending == 0:
                return "tool_result_without_call"
            pending -= 1
            continue
        if pending:
            return "missing_tool_result"
        if role == "assistant":
            calls = m.get("tool_calls") or []
            if any(not isinstance(c, dict) or not isinstance(c.get("name"), str) or not isinstance(c.get("arguments"), dict) for c in calls):
                return "bad_tool_call"
            pending = len(calls)
    if pending:
        return "missing_tool_result"
    if body[-1]["role"] != "assistant":
        return "does_not_end_with_assistant"
    if not any(m["role"] == "assistant" and m.get("train", True) for m in body):
        return "nothing_to_train"
    return None


def conversation_text(conv: dict) -> str:
    parts = [conv.get("system") or ""]
    for m in conv["messages"]:
        parts += [m.get("content") or "", m.get("think") or ""]
        parts += [json.dumps(c.get("arguments"), ensure_ascii=False) for c in m.get("tool_calls") or []]
    return "\n".join(parts)


def render(conv: dict, template, max_tokens: int, decontam=None) -> tuple[np.ndarray, np.ndarray] | str:
    try:
        msgs = conversation_messages(conv, template)
    except (KeyError, TypeError, ValueError):
        return "bad_format"
    reason = check(msgs)
    if reason:
        return reason
    if decontam is not None and decontam.reason({"text": conversation_text(conv)}):
        return "contaminated"
    ids, mask = template.render(msgs)
    if len(ids) > max_tokens:
        return "too_long"
    return np.asarray(ids, dtype=np.uint32), np.asarray(mask, dtype=np.uint8)


def think_targets(ids: np.ndarray, mask: np.ndarray, open_id: int, close_id: int) -> int:
    inside = np.cumsum(ids == open_id) - np.cumsum(ids == close_id) > 0
    return int((inside & (mask > 0)).sum())


def loss_weight(n: int, n_global: float, world: int) -> float:
    return n * world / max(n_global, 1.0)


def pack(lengths: list[int], capacity: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    min_len = min(lengths) if lengths else 0
    bins: list[list[int]] = []
    free: list[tuple[int, int]] = []
    for i in order:
        n = lengths[i]
        k = bisect.bisect_left(free, (n, -1))
        if k < len(free):
            rem, b = free.pop(k)
            bins[b].append(i)
        else:
            rem, b = capacity, len(bins)
            bins.append([i])
        if rem - n >= min_len:
            bisect.insort(free, (rem - n, b))
    return bins


def write_rows(out_dir: str, convs: list[tuple[np.ndarray, np.ndarray]], seq_len: int, pad_id: int, seed: int = 0, stats: dict | None = None, think_ids: tuple[int, int] | None = None) -> dict:
    R = seq_len + 1
    bins = pack([len(c[0]) for c in convs], R)
    np.random.default_rng(seed).shuffle(bins)
    os.makedirs(out_dir, exist_ok=True)
    tok = np.lib.format.open_memmap(os.path.join(out_dir, "rows.tok.npy"), "w+", np.uint32, (len(bins), R))
    flag = np.lib.format.open_memmap(os.path.join(out_dir, "rows.flag.npy"), "w+", np.uint8, (len(bins), R))
    for r, b in enumerate(bins):
        off = 0
        for i in b:
            ids, mask = convs[i]
            tok[r, off : off + len(ids)] = ids
            flag[r, off : off + len(ids)] = mask
            flag[r, off] |= START
            off += len(ids)
        if off < R:
            tok[r, off:] = pad_id
            flag[r, off:] = 0
            flag[r, off] = START
    tok.flush()
    flag.flush()
    used = sum(len(c[0]) for c in convs)
    targets = int(sum(int(c[1].sum()) for c in convs))
    meta = {"template": VERSION, "seq_len": seq_len, "rows": len(bins), "conversations": len(convs), "tokens": used, "target_tokens": targets, "fill": used / max(1, len(bins) * R), **(stats or {})}
    if think_ids:
        meta["think_target_frac"] = sum(think_targets(c[0], c[1], *think_ids) for c in convs) / max(1, targets)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def make_batch(tok: np.ndarray, flag: np.ndarray) -> dict:
    import torch

    tok = tok.astype(np.int64)
    inp = tok[:, :-1]
    labels = np.where(flag[:, 1:] & TARGET, tok[:, 1:], IGNORE_INDEX)
    idx = np.arange(inp.shape[1])
    first = np.maximum.accumulate(np.where(flag[:, :-1] & START, idx, 0), axis=1)
    return {"input_ids": torch.from_numpy(np.ascontiguousarray(inp)), "labels": torch.from_numpy(np.ascontiguousarray(labels)), "position_ids": torch.from_numpy(idx - first)}


def _coprime(n: int, rng) -> int:
    if n <= 2:
        return 1
    while True:
        a = int(rng.integers(n // 3, n)) | 1
        if math.gcd(a, n) == 1:
            return a


class SFTRows:
    def __init__(self, path: str, micro_bs: int, rank: int, world: int, seed: int = 0, one_pass: bool = False):
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        if self.meta["template"] != VERSION:
            raise ValueError(f"{path} was built with template {self.meta['template']}, the code uses {VERSION}")
        self.seq_len = self.meta["seq_len"]
        self.tok = np.load(os.path.join(path, "rows.tok.npy"), mmap_mode="r")
        self.flag = np.load(os.path.join(path, "rows.flag.npy"), mmap_mode="r")
        self.n_rows = len(self.tok)
        self.micro_bs, self.rank, self.world, self.seed, self.one_pass = micro_bs, rank, world, seed, one_pass
        self.micro_step = 0
        self._perm: dict[int, tuple[int, int]] = {}

    def rows_seen(self) -> int:
        return self.micro_step * self.world * self.micro_bs

    def row(self, g: int) -> int:
        epoch, i = divmod(g, self.n_rows)
        if epoch not in self._perm:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm = {epoch: (_coprime(self.n_rows, rng), int(rng.integers(0, self.n_rows)))}
        a, b = self._perm[epoch]
        return (a * i + b) % self.n_rows

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        base = (self.micro_step * self.world + self.rank) * self.micro_bs
        self.micro_step += 1
        rows = [self.row(base + j) for j in range(self.micro_bs)]
        batch = make_batch(self.tok[rows], self.flag[rows])
        if self.one_pass:
            for j in range(self.micro_bs):
                if base + j >= self.n_rows:
                    batch["labels"][j] = IGNORE_INDEX
        return batch
