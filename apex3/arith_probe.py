from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch


def _add(lo, hi):
    def gen(r):
        a, b = r.randint(lo, hi), r.randint(lo, hi)
        return f"{a} + {b}", a + b
    return gen


def _sub(r):
    a = r.randint(100, 999)
    b = r.randint(100, a)
    return f"{a} - {b}", a - b


def _mul(alo, ahi, blo, bhi):
    def gen(r):
        a, b = r.randint(alo, ahi), r.randint(blo, bhi)
        return f"{a} * {b}", a * b
    return gen


def _chain(r):
    a, b, c = r.randint(10, 99), r.randint(10, 99), r.randint(10, 99)
    return f"{a} + {b} + {c}", a + b + c


def _mixed(r):
    a, b, c = r.randint(2, 30), r.randint(2, 30), r.randint(2, 9)
    return f"({a} + {b}) * {c}", (a + b) * c


CATEGORIES: dict[str, Callable] = {"add2": _add(10, 99), "add3": _add(100, 999), "add4": _add(1000, 9999), "add5": _add(10000, 99999),
                                   "sub3": _sub, "mul1x2": _mul(10, 99, 2, 9), "mul2x2": _mul(10, 99, 10, 99), "chain3": _chain, "mixed": _mixed}


def make_problems(per_cat: int, shots: int, seed: int = 0) -> list[tuple[str, str, str]]:
    r = random.Random(seed)
    out = []
    for name, gen in CATEGORIES.items():
        for _ in range(per_cat):
            lines = []
            for _ in range(shots):
                e, v = gen(r)
                lines.append(f"{e} = {v}\n")
            e, v = gen(r)
            out.append((name, "".join(lines) + f"{e} = ", f"{v}\n"))
    return out


def build_probe(problems, encode, max_tries: int = 20, seed: int = 0) -> dict:
    r = random.Random(seed + 1)
    names = list(CATEGORIES)
    ids, lens, starts, cats = [], [], [], []
    for name, prompt, answer in problems:
        for _ in range(max_tries):
            tok, offs = encode(prompt + answer)
            b = [i for i, (s, _) in enumerate(offs) if s >= len(prompt)]
            if b and offs[b[0]][0] == len(prompt) and all(e <= len(prompt) for _, e in offs[: b[0]]):
                break
            shots = prompt.count("\n")
            name, prompt, answer = make_problems(1, shots, r.randrange(1 << 30))[names.index(name)]
        else:
            raise ValueError(f"no token boundary before the answer after {max_tries} tries: {prompt!r}")
        ids.extend(tok)
        lens.append(len(tok))
        starts.append(b[0])
        cats.append(names.index(name))
    return {"ids": np.asarray(ids, dtype=np.int32), "lens": np.asarray(lens, dtype=np.int32), "starts": np.asarray(starts, dtype=np.int32), "cats": np.asarray(cats, dtype=np.int16), "names": np.asarray(names)}


def pack(probe: dict, rows: int, seq_len: int, device="cpu", pad_id: int = 0) -> dict:
    lens, starts = probe["lens"].tolist(), probe["starts"].tolist()
    offsets = np.concatenate([[0], np.cumsum(lens)])
    if max(lens) > seq_len:
        raise ValueError(f"a probe sequence ({max(lens)} tokens) is longer than a row ({seq_len})")
    batches, place = [], []
    b = row = off = 0
    for n in lens:
        if off + n > seq_len:
            row, off = row + 1, 0
            if row == rows:
                b, row = b + 1, 0
        place.append((b, row, off))
        off += n
    for k in range(b + 1):
        batches.append({"input_ids": torch.full((rows, seq_len), pad_id, dtype=torch.long), "position_ids": torch.zeros((rows, seq_len), dtype=torch.long), "fill": [0] * rows, "r": [], "c": [], "t": [], "seq": []})
    ids = torch.from_numpy(probe["ids"].astype(np.int64))
    for i, ((k, rw, o), n, s) in enumerate(zip(place, lens, starts)):
        bt = batches[k]
        bt["input_ids"][rw, o : o + n] = ids[offsets[i] : offsets[i] + n]
        bt["position_ids"][rw, o : o + n] = torch.arange(n)
        bt["fill"][rw] = o + n
        cols = list(range(o + s, o + n))
        bt["r"] += [rw] * len(cols)
        bt["c"] += cols
        bt["t"] += ids[offsets[i] + s : offsets[i] + n].tolist()
        bt["seq"] += [i] * len(cols)
    out = []
    for bt in batches:
        for rw, f in enumerate(bt["fill"]):
            bt["position_ids"][rw, f:] = torch.arange(seq_len - f)
        out.append({"input_ids": bt["input_ids"].to(device), "position_ids": bt["position_ids"].to(device), "r": torch.tensor(bt["r"], device=device), "c": torch.tensor(bt["c"], device=device), "t": torch.tensor(bt["t"], device=device), "seq": torch.tensor(bt["seq"], device=device)})
    return {"batches": out, "cats": torch.from_numpy(probe["cats"].astype(np.int64)), "names": list(probe["names"]), "n": len(lens)}


def load(path: str, rows: int, seq_len: int, device) -> dict:
    with np.load(path) as f:
        probe = {k: f[k] for k in f.files}
    return pack(probe, rows, seq_len, device)


def _hidden(model, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    from .act_offload import mark_layer
    from .model.attention import build_attn_meta

    S = ids.size(1)
    meta = build_attn_meta(pos)
    ropes = {}
    for key in {layer.attn.rope_key for layer in model.layers}:
        cos, sin = model.rope.get(*key, S, ids.device)
        ropes[key] = (cos[meta.position_ids], sin[meta.position_ids])
    x = model.embed(ids)
    for i, layer in enumerate(model.layers):
        mark_layer(i)
        x, _ = layer(x, meta, ropes[layer.attn.rope_key])
    mark_layer(len(model.layers))
    return model.norm(x)


@torch.no_grad()
def run(model, packed: dict, precision: str = "bf16", chunk: int = 1024) -> dict:
    from .precision import precision_context

    n = packed["n"]
    dev = packed["batches"][0]["input_ids"].device
    lp_sum = torch.zeros(n, dtype=torch.float64, device=dev)
    hits = torch.zeros(n, dtype=torch.float64, device=dev)
    count = torch.zeros(n, dtype=torch.float64, device=dev)
    w = model.embed.weight if model.cfg.tie_embeddings else model.head.weight
    was_training = model.training
    model.eval()
    try:
        for bt in packed["batches"]:
            with precision_context(precision):
                h = _hidden(model, bt["input_ids"], bt["position_ids"])
            hs = h[bt["r"], bt["c"] - 1]
            for s in range(0, hs.size(0), chunk):
                logits = (hs[s : s + chunk] @ w.t()).float()
                t = bt["t"][s : s + chunk]
                lp = logits.gather(1, t[:, None]).squeeze(1) - torch.logsumexp(logits, -1)
                seq = bt["seq"][s : s + chunk]
                lp_sum.index_add_(0, seq, lp.double())
                hits.index_add_(0, seq, (logits.argmax(-1) == t).double())
                count.index_add_(0, seq, torch.ones_like(lp, dtype=torch.float64))
    finally:
        model.train(was_training)
    lp_sum, hits, count = lp_sum.cpu(), hits.cpu(), count.cpu()
    exact, p = (hits == count).double(), lp_sum.exp()
    cats = packed["cats"]
    out = {}
    for k, name in enumerate(packed["names"]):
        m = cats == k
        if m.any():
            out[name] = {"em": float(exact[m].mean()), "p": float(p[m].mean()), "acc": float(hits[m].sum() / count[m].sum()), "lp": float(lp_sum[m].sum() / count[m].sum()), "n": int(m.sum())}
    out["all"] = {"em": float(exact.mean()), "p": float(p.mean()), "acc": float(hits.sum() / count.sum()), "lp": float(lp_sum.sum() / count.sum()), "n": n}
    return out


def format_stats(stats: dict) -> str:
    return "(em/p/acc) " + " ".join(f"{k} {v['em']:.3f}/{v['p']:.3f}/{v['acc']:.3f}" for k, v in stats.items())
