from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from apex3.model.moe import Router


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    idx = torch.topk(scores, k, dim=-1, sorted=False).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)


def bias_step(load: torch.Tensor, gamma: float, mode: str) -> torch.Tensor:
    mean = load.mean()
    if mode == "sign":
        return gamma * torch.sign(mean - load)
    return gamma * ((mean - load) / mean.clamp(min=1)).clamp(-1, 1)


def ratio(load: torch.Tensor) -> float:
    return float(load.max() / load.mean().clamp(min=1e-9))


class RouterProbe:
    def __init__(self, model):
        self.routers = [m for m in model.modules() if isinstance(m, Router)]
        self.logits: list[list[torch.Tensor]] = [[] for _ in self.routers]
        self.tags: list[str] = []
        self.tag = ""
        n = len(self.routers)
        self.h_sum: list[torch.Tensor | None] = [None] * n
        self.h_sq = [0.0] * n
        self.h_n = [0] * n
        self._hooks = [r.register_forward_hook(self._hook(i)) for i, r in enumerate(self.routers)]

    def _hook(self, i: int):
        def hook(module, inputs, output):
            h = inputs[0].float()
            self.logits[i].append(F.linear(h, module.weight.float()))
            s = h.sum(0)
            self.h_sum[i] = s if self.h_sum[i] is None else self.h_sum[i] + s
            self.h_sq[i] += float((h * h).sum())
            self.h_n[i] += h.size(0)
            if i == 0:
                self.tags.append(self.tag)
        return hook

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def stats(self, gamma: float) -> list[dict]:
        out = []
        for i, r in enumerate(self.routers):
            z = torch.cat(self.logits[i])
            s = torch.sigmoid(z)
            b = r.expert_bias.float().to(z.device)
            K, E = r.top_k, r.n_experts
            sel = topk_mask(s + b, K)
            load = sel.sum(0).float()
            row = {"layer": i, "tokens": z.size(0), "load_ratio": ratio(load), "hot": int(load.argmax()), "load_ratio_nobias": ratio(topk_mask(s, K).sum(0).float()), "zero_load": int((load == 0).sum())}
            for mode in ("sign", "proportional"):
                sel2 = topk_mask(s + b + bias_step(load, gamma, mode), K)
                load2 = sel2.sum(0).float()
                row[f"step_{mode}"] = {"load_ratio": ratio(load2), "hot": int(load2.argmax()), "moved": float((sel2 != sel).any(-1).float().mean())}
            top = torch.topk(s + b, K + 1, dim=-1).values
            margin = (top[:, K - 1] - top[:, K]).cpu().numpy()
            row["margin_median"] = float(np.median(margin))
            row["margin_lt_gamma"] = float((margin < gamma).mean())
            row["margin_lt_gamma10"] = float((margin < gamma / 10).mean())
            prior = z.mean(0)
            rest = z - prior
            between = float(prior.var(unbiased=False))
            within = float(rest.var(dim=1, unbiased=False).mean())
            row["prior_share"] = between / max(between + within, 1e-12)
            row["logit_std_token"] = float(z.std(dim=1).mean())
            row["score_std_token"] = float(s.std(dim=1).mean())
            row["saturated"] = float(((s > 0.99) | (s < 0.01)).float().mean())
            row["saturated_high"] = float((s > 0.99).float().mean())
            row["logit_mean"] = float(z.mean())
            g = torch.gather(s, 1, torch.topk(s + b, K, dim=-1).indices)
            row["gate_max"] = float((g.max(-1).values / g.sum(-1).clamp(min=1e-20)).mean())
            hm = self.h_sum[i] / self.h_n[i]
            row["h_common"] = float(hm.norm() / (self.h_sq[i] / self.h_n[i]) ** 0.5)
            w = r.weight.float()
            row["w_row"] = float(w.norm(dim=1).mean())
            row["w_mean_row"] = float(w.mean(0).norm())
            row["bias_mean"] = float(b.mean())
            row["bias_range"] = float(b.max() - b.min())
            doms = {}
            sizes = [t.size(0) for t in self.logits[i]]
            for tag in dict.fromkeys(self.tags):
                m = torch.cat([torch.full((n,), t == tag, dtype=torch.bool, device=z.device) for n, t in zip(sizes, self.tags)])
                if m.any():
                    doms[tag] = ratio(sel[m].sum(0).float())
            row["load_ratio_domain"] = doms
            out.append(row)
        return out


def summary(rows: list[dict], gamma: float) -> list[str]:
    def col(key, sub=None):
        v = np.array([(r[key][sub] if sub else r[key]) for r in rows], dtype=float)
        return v

    def mmm(v, fmt="{:.2f}"):
        return f"{fmt.format(v.mean())} / {fmt.format(v.min())} / {fmt.format(v.max())}"

    lr, lnb = col("load_ratio"), col("load_ratio_nobias")
    lines = ["SUMMARY (mean / min / max over MoE layers)", f"  load max/mean, checkpoint bias        {mmm(lr)}", f"  load max/mean, no bias (router alone) {mmm(lnb)}", f"  after one sign step                   {mmm(col('step_sign', 'load_ratio'))}  " f"hot moved in {sum(r['step_sign']['hot'] != r['hot'] for r in rows)}/{len(rows)} layers, " f"slots moved {mmm(col('step_sign', 'moved'), '{:.1%}')}", f"  after one proportional step           {mmm(col('step_proportional', 'load_ratio'))}  " f"hot moved in {sum(r['step_proportional']['hot'] != r['hot'] for r in rows)}/{len(rows)} layers, " f"slots moved {mmm(col('step_proportional', 'moved'), '{:.1%}')}", f"  margin median                         {mmm(col('margin_median'), '{:.4f}')}  (gamma {gamma:g})", f"  tokens with margin < gamma            {mmm(col('margin_lt_gamma'), '{:.1%}')}", f"  tokens with margin < gamma/10         {mmm(col('margin_lt_gamma10'), '{:.1%}')}",
             f"  prior share (token-independent)       {mmm(col('prior_share'))}", f"  logit std across experts per token    {mmm(col('logit_std_token'))}", f"  score std across experts per token    {mmm(col('score_std_token'), '{:.3f}')}", f"  saturated scores (<0.01 or >0.99)     {mmm(col('saturated'), '{:.1%}')}  (>0.99: " f"{mmm(col('saturated_high'), '{:.1%}')})", f"  mean logit                            {mmm(col('logit_mean'))}", f"  largest gate share of the top-k       {mmm(col('gate_max'), '{:.3f}')}  (1/top_k = all equal)", f"  router input common mode |mean h|/rms {mmm(col('h_common'))}", f"  router row norm / expert-mean row     {mmm(col('w_row'))} / {mmm(col('w_mean_row'))}", f"  bias mean / range                     {mmm(col('bias_mean'))} / {mmm(col('bias_range'), '{:.3f}')}", f"  zero-load experts                     {int(col('zero_load').sum())}"]
    doms = {}
    for r in rows:
        for d, v in r["load_ratio_domain"].items():
            doms.setdefault(d, []).append(v)
    if doms:
        lines.append("  load max/mean per domain (layer mean) " + ", ".join(f"{d} {np.mean(v):.2f}" for d, v in doms.items()))
    if lnb.mean() > 1.5 * lr.mean() and col("prior_share").mean() > 0.5:
        lines.append("  READING: the router alone sends most tokens to the same experts (token-independent preference) and the bias is what spreads them: a smaller gamma would let the load collapse; a router-side push (seq_aux_alpha up) addresses the cause")
    elif col("margin_lt_gamma").mean() > 0.3 and sum(r["step_sign"]["hot"] != r["hot"] for r in rows) > len(rows) / 2:
        lines.append("  READING: many tokens sit within one bias step of switching and one step moves the hottest expert: the bias overshoots; a smaller gamma (or the late rate) addresses it")
    else:
        lines.append("  READING: neither pattern is clear; compare the rows per layer below")
    lines.append("LAYERS layer: load (no bias) [sign step: load, hot moved] margin<gamma prior h_common bias_range")
    for r in rows:
        st = r["step_sign"]
        lines.append(f"  L{r['layer'] + 1}: {r['load_ratio']:.2f}@{r['hot']} ({r['load_ratio_nobias']:.2f}) " f"[{st['load_ratio']:.2f}@{st['hot']}, {st['moved']:.1%}] {r['margin_lt_gamma']:.1%} " f"{r['prior_share']:.2f} {r['h_common']:.2f} {r['bias_range']:.3f}")
    return lines


def load_model(cfg, path: str, device: str, experts: str):
    import torch.distributed.checkpoint as dcp

    from apex3.config import ImplConfig
    from apex3.model import Apex3

    with torch.device("meta"):
        model = Apex3(cfg, ImplConfig(attn="sdpa", experts=experts))
    model = model.to_empty(device="cpu")
    for r in model.modules():
        if isinstance(r, Router):
            r.load_counts.zero_()
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=path)
    model.load_state_dict(state["model"])
    for m in model.modules():
        if not isinstance(m, Router):
            for p in m.parameters(recurse=False):
                p.data = p.data.to(torch.bfloat16)
    return model.to(device).eval()


def latest_checkpoint(ckpt_dir: str) -> str:
    from apex3.checkpoint import STATE_FILE

    done = sorted(d for d in os.listdir(ckpt_dir) if os.path.exists(os.path.join(ckpt_dir, d, STATE_FILE)))
    if not done:
        raise FileNotFoundError(f"no complete checkpoint in {ckpt_dir}")
    return os.path.join(ckpt_dir, done[-1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="checkpoint directory (default: the latest complete one in --ckpt-dir)")
    p.add_argument("--ckpt-dir", default="/ckpt/main")
    p.add_argument("--val", nargs="+", required=True, help='name=windows:"glob of held-out *.bin" per domain')
    p.add_argument("--eos-id", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=2, help="windows per forward")
    p.add_argument("--gamma", type=float, default=None, help="bias step to simulate (default: router_bias_rate)")
    p.add_argument("--config", default="C", help="preset for checkpoints without model_config")
    p.add_argument("--device", default="cuda")
    p.add_argument("--experts", default="grouped_mm", choices=["grouped_mm", "loop"])
    p.add_argument("--json", default=None, help="append one JSON line with every number")
    a = p.parse_args(argv)

    from apex3.checkpoint import read_trainer_state
    from apex3.config import resolve_config
    from apex3.data import TokenStream, positions_from_tokens
    from apex3.evaluate import Scorer

    t0 = time.time()
    path = a.ckpt or latest_checkpoint(a.ckpt_dir)
    state = read_trainer_state(path)
    cfg, _ = resolve_config(state, a.config, None, {})
    gamma = a.gamma if a.gamma is not None else cfg.router_bias_rate
    model = load_model(cfg, path, a.device, experts=a.experts)
    print(f"{path}: step {state.get('step')}, tokens {state.get('tokens')}, gamma {gamma:g} " f"({cfg.router_bias_update}), loaded in {time.time() - t0:.0f} s", flush=True)
    probe = RouterProbe(model)
    scorer = Scorer(model)
    n_tok = 0
    for spec in a.val:
        name, rest = spec.split("=", 1)
        count, pattern = rest.split(":", 1)
        stream = TokenStream(pattern)
        total = len(stream) // a.seq_len
        starts = np.unique(np.linspace(0, max(total - 1, 0), min(int(count), total)).astype(np.int64))
        probe.tag = name
        for b0 in range(0, len(starts), a.batch):
            rows = np.stack([stream.read(int(w) * a.seq_len, a.seq_len) for w in starts[b0 : b0 + a.batch]])
            ids = torch.from_numpy(rows).to(model.embed.weight.device)
            pos = torch.from_numpy(np.stack([positions_from_tokens(r, a.eos_id) for r in rows])).to(ids.device)
            scorer.hidden(ids, pos)
            n_tok += ids.numel()
    scorer.close()
    probe.close()
    rows = probe.stats(gamma)
    lines = summary(rows, gamma)
    print(f"{n_tok} tokens, {time.time() - t0:.0f} s")
    print("\n".join(lines))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"checkpoint": path, "step": state.get("step"), "tokens": state.get("tokens"), "gamma": gamma, "mode": cfg.router_bias_update, "n_tokens": n_tok, "layers": rows}) + "\n")
    return rows


if __name__ == "__main__":
    main()
