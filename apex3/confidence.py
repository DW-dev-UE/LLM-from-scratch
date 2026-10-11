from __future__ import annotations

import math
import zlib

import numpy as np
import torch
import torch.nn.functional as F

DEPTH_FRACS = (0.5, 0.75, 1.0)


def default_layers(n_layers: int, fracs=DEPTH_FRACS) -> list[int]:
    return sorted({min(n_layers, max(1, math.ceil(f * n_layers))) for f in fracs})


def pool(states: torch.Tensor) -> torch.Tensor:
    s = states.float()
    return torch.cat([s.mean(0), s[-1]], dim=-1).reshape(-1)


def last_turn_span(ids: list[int], assistant_id: int, stop_ids) -> tuple[int, int]:
    start = max(i for i, t in enumerate(ids) if t == assistant_id)
    end = next((i for i in range(start + 1, len(ids)) if ids[i] in stop_ids), len(ids))
    return start, end


@torch.no_grad()
def turn_features(backend, ids: list[int], span: tuple[int, int], layers: list[int]) -> tuple[torch.Tensor, float]:
    start, end = span
    backend.cached = []
    backend.capture_layers = tuple(i - 1 for i in sorted(layers))
    states, logps = [], []
    try:
        for s in range(0, end, backend.chunk):
            e = min(s + backend.chunk, end)
            h = backend.hidden(ids[s:e], s)
            lo = max(start, s)
            if lo >= e:
                continue
            states.append(backend.layer_states[lo - s : e - s])
            targets = ids[lo + 1 : e + 1]
            if targets:
                logits = backend.logits(h[lo - s : lo - s + len(targets)])
                t = torch.tensor(targets, device=logits.device)
                logps.append(torch.log_softmax(logits, -1).gather(1, t[:, None]).squeeze(1))
    finally:
        backend.capture_layers = ()
    lp = torch.cat(logps) if logps else torch.zeros(0)
    return pool(torch.cat(states)), float(lp.mean()) if lp.numel() else 0.0


def auroc(scores, labels) -> float | None:
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inv]
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ece(probs, labels, bins: int = 10) -> float:
    p, y = np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def best_of_n(scores, labels, groups) -> dict | None:
    by: dict = {}
    for s, y, g in zip(scores, labels, groups):
        by.setdefault(g, []).append((float(s), float(y)))
    picked = rand = oracle = 0.0
    n = 0
    for cands in by.values():
        if len(cands) < 2:
            continue
        n += 1
        top = max(c[0] for c in cands)
        tops = [c[1] for c in cands if c[0] == top]
        picked += sum(tops) / len(tops)
        rand += sum(c[1] for c in cands) / len(cands)
        oracle += float(any(c[1] for c in cands))
    return {"tasks": n, "picked": picked / n, "random": rand / n, "oracle": oracle / n} if n else None


def report(probs, labels, logprob, groups) -> dict:
    labels = np.asarray(labels)
    out = {"n": int(len(labels)), "positive_rate": float(labels.mean()) if len(labels) else None, "auroc": auroc(probs, labels),
           "auroc_logprob": auroc(logprob, labels), "ece": ece(probs, labels), "brier": float(np.mean((np.asarray(probs) - labels) ** 2))}
    bon, bon_lp = best_of_n(probs, labels, groups), best_of_n(logprob, labels, groups)
    if bon:
        out["best_of_n"] = {**bon, "picked_logprob": bon_lp["picked"]}
    return out


def gate_g8(r: dict, auroc_margin: float = 0.05, max_ece: float = 0.05, bon_margin: float = 0.02) -> list[str]:
    fails = []
    if r["auroc"] is None or r["auroc_logprob"] is None:
        fails.append("both outcomes are needed to measure AUROC")
    elif r["auroc"] < r["auroc_logprob"] + auroc_margin:
        fails.append(f"AUROC {r['auroc']:.3f} < log-prob baseline {r['auroc_logprob']:.3f} + {auroc_margin}")
    if r["ece"] > max_ece:
        fails.append(f"ECE {r['ece']:.3f} > {max_ece}")
    bon = r.get("best_of_n")
    if bon is None:
        fails.append("no task with two or more candidates (best-of-n)")
    elif bon["picked"] < bon["picked_logprob"] + bon_margin:
        fails.append(f"best-of-n {bon['picked']:.3f} < log-prob pick {bon['picked_logprob']:.3f} + {bon_margin}")
    return fails


def split_by_group(groups, val_frac: float = 0.2, calib_frac: float = 0.2, seed: int = 0):
    def u(g, salt):
        return zlib.crc32(f"{seed}:{salt}:{g}".encode()) / 2**32

    groups = list(groups)
    val = [i for i, g in enumerate(groups) if u(g, "val") < val_frac]
    rest = [i for i, g in enumerate(groups) if u(g, "val") >= val_frac]
    calib = [i for i in rest if u(groups[i], "calib") < calib_frac]
    train = [i for i in rest if u(groups[i], "calib") >= calib_frac]
    return train, calib, val


def fit_logistic(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-3, max_iter: int = 200):
    w = torch.zeros(x.size(1), dtype=x.dtype, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + 0.5 * l2 * w.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
    logits, y = logits.detach().float().cpu(), y.float().cpu()

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


class ConfidenceHead:
    def __init__(self, layers, mean, std, weight, bias, temperature: float = 1.0, meta: dict | None = None):
        self.layers = sorted(int(i) for i in layers)
        self.mean, self.std, self.weight = mean.float().cpu(), std.float().cpu(), weight.float().cpu()
        self.bias, self.temperature, self.meta = float(bias), float(temperature), dict(meta or {})

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        x = (feats.float().cpu() - self.mean) / self.std
        return (x @ self.weight + self.bias) / self.temperature

    def prob(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(feats))

    def score_turn(self, backend) -> float | None:
        states = getattr(backend, "turn_states", None)
        return None if states is None else float(self.prob(pool(states)))

    def save(self, path: str) -> None:
        torch.save({"layers": self.layers, "mean": self.mean, "std": self.std, "weight": self.weight, "bias": self.bias, "temperature": self.temperature, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str) -> "ConfidenceHead":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(d["layers"], d["mean"], d["std"], d["weight"], d["bias"], d["temperature"], d["meta"])
