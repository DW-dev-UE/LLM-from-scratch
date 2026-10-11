from __future__ import annotations

import math

import numpy as np
import torch

from .data import TokenStream, positions_from_tokens
from .model.attention import build_attn_meta
from .model.moe import Router

ICL_EARLY, ICL_LATE = (40, 60), (450, 550)


class Scorer:
    def __init__(self, model, row_len: int = 4096, rows_per_batch: int = 8, pad_id: int = 0, chunk: int = 2048):
        self.model, self.row_len, self.rows_per_batch, self.pad_id, self.chunk = model, row_len, rows_per_batch, pad_id, chunk
        self.device = model.embed.weight.device
        self.capture = False
        self._routes: list[torch.Tensor] = []
        self._hooks = [m.register_forward_hook(self._hook) for m in model.modules() if isinstance(m, Router)]
        self.n_moe = len(self._hooks)
        self.n_experts = model.cfg.n_experts

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _hook(self, module, inputs, output):
        if self.capture:
            self._routes.append(output[0].detach())

    @torch.no_grad()
    def hidden(self, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        m = self.model
        S = ids.size(1)
        meta = build_attn_meta(pos)
        ropes = {}
        for key in {b.attn.rope_key for b in m.layers}:
            cos, sin = m.rope.get(*key, S, ids.device)
            ropes[key] = (cos[meta.position_ids], sin[meta.position_ids])
        x = m.embed(ids)
        for b in m.layers:
            x, _ = b(x, meta, ropes[b.attn.rope_key])
        return m.norm(x)

    @torch.no_grad()
    def logprobs(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        m = self.model
        w = m.embed.weight if m.cfg.tie_embeddings else m.head.weight
        out = []
        for s in range(0, h.size(0), self.chunk):
            logits = (h[s : s + self.chunk] @ w.t()).float()
            out.append(logits.gather(1, targets[s : s + self.chunk, None]).squeeze(1) - torch.logsumexp(logits, -1))
        return torch.cat(out) if out else torch.zeros(0, device=h.device)

    @torch.no_grad()
    def score(self, seqs: list[tuple[list[int], int]]) -> list[tuple[float, int]]:
        L = self.row_len
        rows: list[list[list[int]]] = []
        places = []
        cur, cur_len = [], 0
        for ids, start in seqs:
            if len(ids) > L:
                cut = len(ids) - L
                ids, start = ids[cut:], max(1, start - cut)
            if cur_len + len(ids) > L:
                rows.append(cur)
                cur, cur_len = [], 0
            places.append((len(rows), cur_len, len(ids), start))
            cur.append(ids)
            cur_len += len(ids)
        if cur:
            rows.append(cur)
        by_row: dict[int, list[int]] = {}
        for i, p in enumerate(places):
            by_row.setdefault(p[0], []).append(i)
        sums = torch.zeros(len(seqs), dtype=torch.float64)
        counts = [p[2] - p[3] for p in places]
        for b0 in range(0, len(rows), self.rows_per_batch):
            batch = rows[b0 : b0 + self.rows_per_batch]
            ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
            pos = torch.zeros((len(batch), L), dtype=torch.long)
            for r, segs in enumerate(batch):
                off = 0
                for s in segs:
                    ids[r, off : off + len(s)] = torch.tensor(s)
                    pos[r, off : off + len(s)] = torch.arange(len(s))
                    off += len(s)
                pos[r, off:] = torch.arange(L - off)
            ids, pos = ids.to(self.device), pos.to(self.device)
            h = self.hidden(ids, pos)
            r_idx, c_idx, owner = [], [], []
            for r in range(len(batch)):
                for i in by_row.get(b0 + r, []):
                    _, off, n, start = places[i]
                    cols = list(range(off + start - 1, off + n - 1))
                    r_idx += [r] * len(cols)
                    c_idx += cols
                    owner += [i] * len(cols)
            if not r_idx:
                continue
            r_t, c_t = torch.tensor(r_idx, device=self.device), torch.tensor(c_idx, device=self.device)
            lp = self.logprobs(h[r_t, c_t], ids[r_t, c_t + 1]).double().cpu()
            sums.index_add_(0, torch.tensor(owner), lp)
        return [(float(sums[i]), counts[i]) for i in range(len(seqs))]

    @torch.no_grad()
    def windows(self, stream: TokenStream, eos_id: int, n_windows: int, seq_len: int, batch: int = 4) -> dict:
        total = len(stream) // (seq_len + 1)
        starts = np.unique(np.linspace(0, max(total - 1, 0), min(n_windows, total)).astype(np.int64))
        ce_sum, n_tok = 0.0, 0
        buckets = {"early": [0.0, 0], "late": [0.0, 0]}
        paths = []
        for b0 in range(0, len(starts), batch):
            rows = np.stack([stream.read(int(w) * (seq_len + 1), seq_len + 1) for w in starts[b0 : b0 + batch]])
            inp = torch.from_numpy(rows[:, :-1]).to(self.device)
            lab = torch.from_numpy(rows[:, 1:]).to(self.device)
            pos = torch.from_numpy(np.stack([positions_from_tokens(r, eos_id) for r in rows[:, :-1]])).to(self.device)
            self.capture, self._routes = True, []
            h = self.hidden(inp, pos)
            self.capture = False
            ce = -self.logprobs(h.reshape(-1, h.size(-1)), lab.reshape(-1)).view_as(lab)
            ce_sum += float(ce.sum())
            n_tok += ce.numel()
            for name, (lo, hi) in (("early", ICL_EARLY), ("late", ICL_LATE)):
                m = (pos >= lo) & (pos < hi)
                buckets[name][0] += float(ce[m].sum())
                buckets[name][1] += int(m.sum())
            if self._routes:
                B = rows.shape[0]
                per_layer = []
                for idx in self._routes:
                    row = torch.arange(B, device=idx.device).repeat_interleave(seq_len)[:, None].expand_as(idx)
                    counts = torch.bincount((row * self.n_experts + idx).reshape(-1), minlength=B * self.n_experts)
                    per_layer.append((counts.view(B, self.n_experts).float() / idx[: seq_len].numel()).cpu().numpy())
                paths.append(np.stack(per_layer, axis=1))
        early = buckets["early"][0] / max(1, buckets["early"][1])
        late = buckets["late"][0] / max(1, buckets["late"][1])
        return {"ce": ce_sum / max(1, n_tok), "early_ce": early, "late_ce": late, "icl": late - early if buckets["late"][1] and buckets["early"][1] else None,
                "pathways": np.concatenate(paths) if paths else None, "windows": int(len(starts))}


def eval_mc(scorer: Scorer, encode, examples: list[dict]) -> dict:
    seqs = []
    for ex in examples:
        ctx = encode(ex["context"])
        seqs += [(ctx + encode(c), len(ctx)) for c in ex["choices"]]
    res = scorer.score(seqs)
    return mc_metrics(examples, [s for s, _ in res])


def mc_metrics(examples: list[dict], sums: list[float]) -> dict:
    acc = acc_norm = p_correct = 0.0
    k = 0
    for ex in examples:
        s = np.array(sums[k : k + len(ex["choices"])])
        k += len(ex["choices"])
        nbytes = np.array([max(1, len(c.encode())) for c in ex["choices"]])
        acc += float(np.argmax(s) == ex["answer"])
        acc_norm += float(np.argmax(s / nbytes) == ex["answer"])
        p = np.exp(s - s.max())
        p_correct += float(p[ex["answer"]] / p.sum())
    n = max(1, len(examples))
    return {"acc": acc / n, "acc_norm": acc_norm / n, "p_correct": p_correct / n, "n": len(examples)}


def eval_ref(scorer: Scorer, encode, examples: list[dict]) -> dict:
    seqs = []
    for ex in examples:
        ctx = encode(ex["context"])
        seqs.append((ctx + encode(ex["target"]), len(ctx)))
    res = scorer.score(seqs)
    lp, n_tok = sum(s for s, _ in res), sum(n for _, n in res)
    n_bytes = sum(len(ex["target"].encode()) for ex in examples)
    return {"logprob_per_token": lp / max(1, n_tok), "bpb": -lp / math.log(2) / max(1, n_bytes), "n": len(examples)}


def _js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p, q = p + 1e-12, q + 1e-12
    m = (p + q) / 2
    return 0.5 * ((p * np.log(p / m)).sum(-1) + (q * np.log(q / m)).sum(-1))


def routing_metrics(pathways: dict[str, np.ndarray], prev: dict[str, np.ndarray] | None = None):
    usage = {d: a.mean(0) for d, a in pathways.items()}
    overall = np.mean(list(usage.values()), axis=0)
    E = next(iter(usage.values())).shape[-1]
    out = {}
    for d, a in pathways.items():
        p = usage[d]
        entropy = float((-(p * np.log(p + 1e-12)).sum(-1) / math.log(E)).mean())
        v = a.reshape(len(a), -1)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        n = len(v)
        consistency = float(((v @ v.T).sum() - n) / (n * (n - 1))) if n > 1 else None
        out[d] = {"entropy": entropy, "specialization": float(_js(p, overall).mean()), "consistency": consistency, "drift": float(_js(p, prev[d]).mean()) if prev and d in prev else None}
    return out, usage
