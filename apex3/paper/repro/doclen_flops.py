#!/usr/bin/env python3
"""Exact attention FLOPs under document masking for the Apex-3 phase-1 pretraining data.

Standalone (numpy + stdlib only; no torch, no apex3). Reproduces how the training loader forms rows and how FA3
varlen sees them, then counts attention context exactly.

Loader facts mirrored here (apex3/data.py):
  * *.bin = flat little-endian uint32 token ids, documents separated by EOS (203776); no header, no padding.
  * A domain stream is the concatenation of sorted(glob(<root>/<domain>/*.bin)); window w covers stream tokens
    [w*(S+1), w*(S+1)+S+1), n_windows = len(stream)//(S+1) (stride 1); windows may span shard boundaries.
    Windows are visited in a seeded coprime permutation per epoch, i.e. uniformly at random over all windows.
  * pack_rows: input = window[:-1] (S tokens), labels = window[1:]. Positions restart at 0 at the row start (even when
    the row starts in the middle of a document) and on the token right AFTER every EOS (the EOS token itself is the
    last token of the previous segment). build_attn_meta: cu_seqlens = indices where position == 0 (flattened over the
    micro-batch rows) plus the total length, so every row is cut into varlen segments independently.
  * FA3 varlen, causal=True, window_size=(1023, 0) on the 25 local layers, (-1, -1) on the 5 global layers.

Counting convention: attention FLOPs/token/layer (fwd+bwd) = 12 * (n_h*d_h) * ctx, ctx = number of keys a query
attends to = p (global) or min(p, 1024) (local), p = 1-based position inside its varlen segment.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import gc
import json
import mmap
import os
import platform
import sys
import time
import zlib

import numpy as np

# ---- model / paper constants -------------------------------------------------------------------------------------
S = 8192                      # sequence length (input tokens per row)
ROW = S + 1                   # window length read by the loader
EOS = 203776
WIN = 1024                    # keys visible on local layers incl. the query itself (window_size=(1023, 0))
N_GLOBAL, N_LOCAL = 5, 25
HD = 20 * 128                 # n_h * d_h
N_ACTIVE = 3_360_194_560
SUM_C = N_GLOBAL * S + N_LOCAL * min(2 * WIN, S)          # paper convention: 92,160
F_DENSE = 6 * N_ACTIVE                                    # 20.161 GFLOP
F_PAPER = F_DENSE + 6 * HD * SUM_C                        # 21.577 GFLOP
ATTN_PAPER = 6 * HD * SUM_C                               # 1.4156 GFLOP

MIX = {"code": 0.35, "web": 0.40, "textbook": 0.02, "pdf": 0.05, "math": 0.06, "papers": 0.06, "medical": 0.03,
       "qa": 0.02, "books": 0.01}

# reported real-training MFU (paper convention): (interval, instance, MFU %)
TRAIN_MFU = [("0-2B", "A", 36.5), ("2-3B", "A", 41.2), ("3-4B", "A", 44.8),
             ("4-6B", "A", 46.6), ("4-6B", "B", 46.8), ("6-8B", "A", 47.1), ("6-8B", "B", 47.3),
             ("8-10B", "A", 47.5), ("8-10B", "B", 47.7), ("10-10.9B", "A", 48.9), ("10-10.9B", "B", 49.1)]
BENCH_MFU = [("stage 8", 45.7), ("stage 9", 49.23), ("stage 10", 50.74)]

PAGE = mmap.PAGESIZE


# ---- exact counting ----------------------------------------------------------------------------------------------
def seg_lengths(inp: np.ndarray) -> np.ndarray:
    """Varlen segment lengths of one row. A segment starts at index 0 and at i+1 for every EOS at i < S-1."""
    ends = np.flatnonzero(inp[:-1] == EOS)
    bounds = np.empty(len(ends) + 2, dtype=np.int64)
    bounds[0] = 0
    bounds[1:-1] = ends + 1
    bounds[-1] = len(inp)
    return np.diff(bounds)


def ctx_sums(L: np.ndarray, win: int = WIN) -> tuple[int, int, int]:
    """(sum of global ctx, sum of local ctx, #tokens with 1-based p < win) over all tokens of the given segments."""
    L = L.astype(np.int64)
    tri = L * (L + 1) // 2
    glob_sum = int(tri.sum())
    loc = np.where(L <= win, tri, win * (win + 1) // 2 + (L - win) * win)
    return glob_sum, int(loc.sum()), int(np.minimum(L, win - 1).sum())


def attn_gflop_per_token(mean_gctx: float, mean_lctx: float) -> float:
    return 12 * HD * (N_GLOBAL * mean_gctx + N_LOCAL * mean_lctx) / 1e9


# ---- shard access (np.memmap, small slices only) -----------------------------------------------------------------
IO = {"madvise_ok": 0, "madvise_fail": 0, "bytes": 0}


def read_slice(path: str, pos: int, n: int) -> np.ndarray:
    """Copy tokens [pos, pos+n) of one shard. Maps the file, hints the exact byte range, copies, unmaps, then drops
    the touched page-cache range so the run leaves no cache footprint behind."""
    b0, nb = pos * 4, n * 4
    mm = np.memmap(path, dtype="<u4", mode="r")
    try:
        a = (b0 // PAGE) * PAGE
        ln = min(b0 + nb - a, mm.nbytes - a)
        mm._mmap.madvise(mmap.MADV_RANDOM)
        mm._mmap.madvise(mmap.MADV_WILLNEED, a, ln)
        IO["madvise_ok"] += 1
    except Exception:
        IO["madvise_fail"] += 1
    out = np.array(mm[pos:pos + n], dtype=np.uint32)
    del mm
    IO["bytes"] += nb
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, max(0, b0 - 131072), nb + 262144, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return out


class Domain:
    def __init__(self, name: str, root: str):
        self.name = name
        self.glob = f"{root}/{name}/*.bin"
        self.paths = sorted(glob.glob(self.glob))
        if not self.paths:
            raise FileNotFoundError(self.glob)
        sizes = np.array([os.stat(p).st_size for p in self.paths], dtype=np.int64)
        if (sizes % 4).any():
            raise ValueError(f"{name}: shard size not a multiple of 4 bytes")
        self.ntok = sizes // 4
        self.offsets = np.concatenate(([0], np.cumsum(self.ntok)))
        self.n_tokens = int(self.offsets[-1])
        self.n_windows = self.n_tokens // ROW

    def read(self, start: int, length: int) -> np.ndarray:
        """Same semantics as TokenStream.read: stream[start:start+length] across shard boundaries."""
        out = np.empty(length, dtype=np.uint32)
        i = int(np.searchsorted(self.offsets, start, side="right") - 1)
        pos, filled = start - int(self.offsets[i]), 0
        while filled < length:
            n = min(length - filled, int(self.ntok[i]) - pos)
            if n > 0:
                out[filled:filled + n] = read_slice(self.paths[i], pos, n)
                filled += n
            i, pos = i + 1, 0
        return out

    def sample_windows(self, k: int, rng: np.random.Generator) -> np.ndarray:
        k = min(k, self.n_windows)
        got = np.unique(rng.integers(0, self.n_windows, size=k))
        while len(got) < k:
            got = np.unique(np.concatenate([got, rng.integers(0, self.n_windows, size=k - len(got))]))
        return got


# ---- self test (no data access) ------------------------------------------------------------------------------------
def _positions_from_tokens(tokens: np.ndarray, eos_id: int) -> np.ndarray:
    """Verbatim logic of apex3.data.positions_from_tokens."""
    n = tokens.shape[0]
    is_start = np.zeros(n, dtype=bool)
    is_start[0] = True
    is_start[1:] = tokens[:-1] == eos_id
    idx = np.arange(n)
    return idx - np.maximum.accumulate(np.where(is_start, idx, 0))


def selftest() -> None:
    rng = np.random.default_rng(123)
    # (1) closed form == position-based brute force, S = 8192, assorted EOS densities incl. edge cases
    for trial in range(300):
        p_eos = [0.0, 1e-4, 1e-3, 5e-3, 2e-2, 0.2, 0.9][trial % 7]
        inp = rng.integers(0, 1000, size=S).astype(np.uint32)
        inp[rng.random(S) < p_eos] = EOS
        if trial % 11 == 0:
            inp[0] = EOS
        if trial % 13 == 0:
            inp[-1] = EOS
        pos = _positions_from_tokens(inp, EOS)
        p1 = pos + 1
        g, l, pl = ctx_sums(seg_lengths(inp))
        assert g == int(p1.sum()) and l == int(np.minimum(p1, WIN).sum()) and pl == int((p1 < WIN).sum()), trial
        assert len(seg_lengths(inp)) == int((pos == 0).sum()) and int(seg_lengths(inp).sum()) == S
    # (2) position-based ctx == explicit key counting of the varlen causal sliding-window mask (small config)
    Ssm, Wsm = 301, 37
    for trial in range(40):
        inp = rng.integers(0, 50, size=Ssm).astype(np.uint32)
        inp[rng.random(Ssm) < [0.0, 0.01, 0.05, 0.3][trial % 4]] = EOS
        doc = np.cumsum(_positions_from_tokens(inp, EOS) == 0)
        i = np.arange(Ssm)
        same = (doc[:, None] == doc[None, :]) & (i[:, None] >= i[None, :])
        g_mask = int(same.sum())
        l_mask = int((same & (i[:, None] - i[None, :] < Wsm)).sum())
        lens = np.diff(np.concatenate(([0], np.flatnonzero(inp[:-1] == EOS) + 1, [Ssm])))
        g, l, _ = ctx_sums(lens, win=Wsm)
        assert g == g_mask and l == l_mask, (trial, g, g_mask, l, l_mask)
    # (3) full single-document row
    g, l, pl = ctx_sums(np.array([S]))
    gc_, lc_ = g / S, l / S
    att = attn_gflop_per_token(gc_, lc_)
    assert abs(gc_ - 4096.5) < 1e-9 and abs(lc_ - 960.0625) < 1e-9
    print(f"selftest PASS: full-row mean global ctx {gc_}, local ctx {lc_}, attention {att:.6f} GFLOP/token, "
          f"F {(F_DENSE / 1e9 + att):.6f} GFLOP (paper {F_PAPER / 1e9:.6f}), p<1024 tokens {pl}")


# ---- statistics ----------------------------------------------------------------------------------------------------
def bootstrap_mixture(vals: dict[str, np.ndarray], w: dict[str, float], iters: int, rng) -> np.ndarray:
    """Bootstrap distribution of the mixture-weighted mean of per-row values (rows resampled within each domain)."""
    total = np.zeros(iters)
    for d, v in vals.items():
        n = len(v)
        acc = np.empty(iters)
        step = max(1, 4_000_000 // n)
        for s in range(0, iters, step):
            e = min(iters, s + step)
            acc[s:e] = v[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
        total += w[d] * acc
    return total


def weighted_median(vals: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(vals, kind="stable")
    cw = np.cumsum(weights[order])
    return float(vals[order][np.searchsorted(cw, 0.5 * cw[-1])])


def peak_rss_mb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # Linux: KiB
    except Exception:
        return None


def read_proc_io() -> int | None:
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


# ---- main ----------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/bin")
    ap.add_argument("--out", help="output JSON path (a .md with the same stem is written beside it)")
    ap.add_argument("--rows-total", type=int, default=20000, help="rows allocated by mixture weight")
    ap.add_argument("--min-rows", type=int, default=300, help="minimum rows per domain")
    ap.add_argument("--rows-override", type=int, default=0, help="use exactly this many rows per selected domain")
    ap.add_argument("--only", default="", help="comma separated subset of domains (partial run, for checks)")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--throttle-ms", type=float, default=0.0, help="sleep after every row")
    ap.add_argument("--max-seconds", type=float, default=1080.0, help="stop sampling after this wall time")
    ap.add_argument("--selftest", action="store_true", help="run the data-free self test and exit")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.out:
        ap.error("--out is required")

    t0 = time.time()
    io0 = read_proc_io()
    names = [d for d in MIX if not args.only or d in args.only.split(",")]
    partial = len(names) != len(MIX)
    wsum = sum(MIX[d] for d in names)
    w = {d: MIX[d] / wsum for d in names}
    rows_n = {d: (args.rows_override or max(args.min_rows, int(round(MIX[d] * args.rows_total)))) for d in names}

    doms = {}
    tasks = []
    for d in names:
        doms[d] = Domain(d, args.data_root)
        rng = np.random.default_rng([args.seed, zlib.crc32(d.encode())])
        picks = doms[d].sample_windows(rows_n[d], rng)
        rows_n[d] = len(picks)
        tasks.extend((d, int(x)) for x in picks)
        print(f"{d}: {len(doms[d].paths)} shards, {doms[d].n_tokens / 1e9:.2f}B tokens, {doms[d].n_windows} windows, "
              f"sampling {rows_n[d]} rows", flush=True)
    order = np.random.default_rng([args.seed, 777]).permutation(len(tasks))   # interleave domains, unbiased if cut

    acc = {d: {"g": [], "l": [], "p": [], "nseg": [], "lens": [], "inner": [], "maxtok": 0, "eos": 0, "windows": []}
           for d in names}
    truncated = False
    for k, ti in enumerate(order):
        d, wi = tasks[int(ti)]
        inp = doms[d].read(wi * ROW, S)             # window[:-1]; the last window token is only a label
        lens = seg_lengths(inp)
        g, l, p = ctx_sums(lens)
        a = acc[d]
        a["g"].append(g)
        a["l"].append(l)
        a["p"].append(p)
        a["nseg"].append(len(lens))
        a["lens"].append(lens.astype(np.int32))
        if len(lens) > 2:
            a["inner"].append(lens[1:-1].astype(np.int32))
        a["maxtok"] = max(a["maxtok"], int(inp.max()))
        a["eos"] += len(lens) - 1
        a["windows"].append(wi)
        if args.throttle_ms:
            time.sleep(args.throttle_ms / 1000)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(order)} rows, {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > args.max_seconds:
            truncated = True
            print(f"  time limit reached after {k + 1} rows", flush=True)
            break

    # ---- per-domain results
    res = {}
    row_attn = {}
    all_lens, all_w = [], []
    for d in names:
        a = acc[d]
        n = len(a["g"])
        if n == 0:
            continue
        g, l, p = (np.array(a[x], dtype=np.float64) for x in ("g", "l", "p"))
        lens = np.concatenate(a["lens"]).astype(np.int64)
        inner = np.concatenate(a["inner"]) if a["inner"] else np.array([], dtype=np.int64)
        gctx, lctx = g.mean() / S, l.mean() / S
        per_row = 12 * HD * (N_GLOBAL * g + N_LOCAL * l) / S / 1e9          # GFLOP/token of each row
        row_attn[d] = per_row
        se = per_row.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        attn = attn_gflop_per_token(gctx, lctx)
        F = F_DENSE / 1e9 + attn
        res[d] = {
            "weight": w[d], "rows": n, "shards": len(doms[d].paths), "domain_tokens": doms[d].n_tokens,
            "n_windows": doms[d].n_windows,
            "mean_global_ctx": gctx, "mean_local_ctx": lctx,
            "attn_gflop_per_token": attn, "F_gflop": F, "F_ratio": F / (F_PAPER / 1e9),
            "F_ratio_ci95_normal": [(F - 1.96 * se) / (F_PAPER / 1e9), (F + 1.96 * se) / (F_PAPER / 1e9)],
            "segments": int(len(lens)), "segments_per_row": float(len(lens) / n),
            "mean_seg_len": float(lens.mean()), "median_seg_len": float(np.median(lens)),
            "token_weighted_mean_seg_len": float((lens.astype(np.float64) ** 2).sum() / lens.sum()),
            "complete_doc_segments": int(len(inner)),
            "complete_doc_mean_len": float(inner.mean()) if len(inner) else None,
            "complete_doc_median_len": float(np.median(inner)) if len(inner) else None,
            "frac_rows_single_segment": float(np.mean(np.array(a["nseg"]) == 1)),
            "frac_tokens_p_lt_1024": float(p.sum() / (n * S)),
            "max_token_id_seen": a["maxtok"], "eos_per_row": float(a["eos"] / n),
        }
        all_lens.append(lens)
        all_w.append(np.full(len(lens), w[d] / n))

    # ---- mixture
    used = [d for d in names if d in res]
    wn = {d: w[d] / sum(w[x] for x in used) for d in used}
    mix_g = sum(wn[d] * res[d]["mean_global_ctx"] for d in used)
    mix_l = sum(wn[d] * res[d]["mean_local_ctx"] for d in used)
    mix_attn = attn_gflop_per_token(mix_g, mix_l)
    mix_F = F_DENSE / 1e9 + mix_attn
    ratio = mix_F / (F_PAPER / 1e9)
    boot_rng = np.random.default_rng([args.seed, 999])
    boot = bootstrap_mixture(row_attn, wn, args.boot, boot_rng)
    b_lo, b_hi = np.percentile(boot, [2.5, 97.5])
    boot_ratio = (F_DENSE / 1e9 + boot) / (F_PAPER / 1e9)
    r_lo, r_hi = float(np.percentile(boot_ratio, 2.5)), float(np.percentile(boot_ratio, 97.5))
    se_attn = float(np.sqrt(sum(wn[d] ** 2 * row_attn[d].var(ddof=1) / len(row_attn[d]) for d in used if len(row_attn[d]) > 1)))
    n_lo, n_hi = (mix_F - 1.96 * se_attn) / (F_PAPER / 1e9), (mix_F + 1.96 * se_attn) / (F_PAPER / 1e9)
    segs_per_row = sum(wn[d] * res[d]["segments_per_row"] for d in used)
    wts = np.concatenate(all_w)
    wts = wts * np.concatenate([np.full(len(x), wn[d] / w[d]) for d, x in zip(used, all_lens)])
    lens_all = np.concatenate(all_lens)
    mixture = {
        "partial_subset_run": partial,
        "mean_global_ctx": mix_g, "mean_local_ctx": mix_l, "attn_gflop_per_token": mix_attn, "F_gflop": mix_F,
        "F_ratio": ratio, "F_ratio_ci95_bootstrap": [r_lo, r_hi], "F_ratio_ci95_normal": [n_lo, n_hi],
        "attn_gflop_per_token_ci95_bootstrap": [float(b_lo), float(b_hi)],
        "bootstrap_iters": args.boot,
        "segments_per_row": segs_per_row, "mean_seg_len": S / segs_per_row,
        "median_seg_len": weighted_median(lens_all.astype(np.float64), wts),
        "frac_tokens_p_lt_1024": sum(wn[d] * res[d]["frac_tokens_p_lt_1024"] for d in used),
        "frac_rows_single_segment": sum(wn[d] * res[d]["frac_rows_single_segment"] for d in used),
    }

    # ---- references and corrected MFU
    gfull, lfull, plfull = ctx_sums(np.array([S]))
    nomask_attn = attn_gflop_per_token(gfull / S, lfull / S)
    nomask_F = F_DENSE / 1e9 + nomask_attn
    nomask = {"mean_global_ctx": gfull / S, "mean_local_ctx": lfull / S, "attn_gflop_per_token": nomask_attn,
              "F_gflop": nomask_F, "F_ratio": nomask_F / (F_PAPER / 1e9), "frac_tokens_p_lt_1024": plfull / S}
    paper = {"mean_global_ctx": S / 2, "mean_local_ctx": min(2 * WIN, S) / 2,
             "attn_gflop_per_token": ATTN_PAPER / 1e9, "F_gflop": F_PAPER / 1e9, "F_ratio": 1.0,
             "note": "6*n_h*d_h*sum_l c_l with c = S (global), 2W (local); equals 12*n_h*d_h*ctx with ctx = S/2, W"}
    corr_train = [{"interval": iv, "instance": inst, "reported_mfu": m, "corrected_mfu": m * ratio,
                   "corrected_mfu_ci95_bootstrap": [m * r_lo, m * r_hi]} for iv, inst, m in TRAIN_MFU]
    corr_bench = [{"stage": st, "reported_mfu": m, "corrected_mfu": m * nomask["F_ratio"]} for st, m in BENCH_MFU]

    io1 = read_proc_io()
    out = {
        "meta": {"script": "paper/repro/doclen_flops.py", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                 "host": platform.node(), "python": platform.python_version(), "numpy": np.__version__,
                 "seed": args.seed, "seq_len": S, "window_keys": WIN, "eos_id": EOS, "layers_global": N_GLOBAL,
                 "layers_local": N_LOCAL, "n_h_d_h": HD, "N_active": N_ACTIVE, "F_dense_gflop": F_DENSE / 1e9,
                 "F_paper_gflop": F_PAPER / 1e9, "mixture_weights": MIX, "rows_planned": rows_n,
                 "rows_used": {d: res[d]["rows"] for d in res}, "truncated_by_time_limit": truncated,
                 "elapsed_s": time.time() - t0, "logical_bytes_read": IO["bytes"],
                 "proc_read_bytes": (io1 - io0) if io0 is not None and io1 is not None else None,
                 "madvise_ok": IO["madvise_ok"], "madvise_fail": IO["madvise_fail"],
                 "peak_rss_mb": peak_rss_mb()},
        "conventions": {"ctx": "keys attended by a query = p (global) or min(p, 1024) (local), p = 1-based position "
                               "in its varlen segment; attention FLOPs/token = 12*n_h*d_h*sum_l ctx_l",
                        "segments": "a row of S input tokens is cut at index 0 and after every EOS; the EOS token "
                                    "ends the previous segment",
                        "frac_p_lt_1024": "fraction of tokens whose 1-based p < 1024 (local window clipped by the "
                                          "segment start)"},
        "paper_convention": paper, "nomask_exact_full_row": nomask, "mixture_masked": mixture, "domains": res,
        "corrected_mfu": {"factor_masked": ratio, "factor_nomask_exact": nomask["F_ratio"], "training": corr_train,
                          "benchmark": corr_bench},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(render_md(out))
    print(f"mixture: global ctx {mix_g:.2f}, local ctx {mix_l:.3f}, attention {mix_attn:.4f} GFLOP/token, "
          f"F {mix_F:.4f} GFLOP, F/F_paper {ratio:.5f} (95% bootstrap {r_lo:.5f}..{r_hi:.5f}, normal {n_lo:.5f}..{n_hi:.5f})")
    print(f"no-mask exact: attention {nomask_attn:.4f} GFLOP/token, F {nomask_F:.4f}, ratio {nomask['F_ratio']:.5f}")
    print(f"elapsed {time.time() - t0:.1f}s, logical bytes {IO['bytes'] / 1e6:.0f} MB, proc read_bytes "
          f"{out['meta']['proc_read_bytes']}, truncated={truncated}; wrote {args.out} and {md_path}")


def render_md(o: dict) -> str:
    m, nm, pp, mx = o["meta"], o["nomask_exact_full_row"], o["paper_convention"], o["mixture_masked"]
    L = []
    L.append("# Document-masking correction of attention FLOPs (exact count)\n")
    L.append(f"Generated {m['date_utc']} on `{m['host']}` by `{m['script']}` (seed {m['seed']}, "
             f"{sum(m['rows_used'].values())} rows sampled uniformly over 8193-token windows, "
             f"elapsed {m['elapsed_s']:.0f} s, {m['logical_bytes_read'] / 1e6:.0f} MB read, peak RSS {(m['peak_rss_mb'] or 0):.0f} MB"
             f"{', PARTIAL domain subset' if mx['partial_subset_run'] else ''}"
             f"{', STOPPED BY TIME LIMIT' if m['truncated_by_time_limit'] else ''}).\n")
    L.append("## Headline\n")
    L.append(f"- Mixture-weighted F(masked) = {mx['F_gflop']:.4f} GFLOP/token, **F/F_paper = {mx['F_ratio']:.5f}** "
             f"(95% bootstrap {mx['F_ratio_ci95_bootstrap'][0]:.5f} to {mx['F_ratio_ci95_bootstrap'][1]:.5f}; "
             f"normal approx {mx['F_ratio_ci95_normal'][0]:.5f} to {mx['F_ratio_ci95_normal'][1]:.5f}).")
    L.append(f"- Attention GFLOP/token: masked mixture {mx['attn_gflop_per_token']:.4f}, "
             f"no-mask exact (full 8192-token document) {nm['attn_gflop_per_token']:.4f}, "
             f"paper convention {pp['attn_gflop_per_token']:.4f}.")
    L.append(f"- Mean segment length {mx['mean_seg_len']:.1f} tokens, median {mx['median_seg_len']:.1f} "
             f"({mx['segments_per_row']:.2f} segments per row); fraction of tokens with p < 1024: "
             f"{mx['frac_tokens_p_lt_1024']:.4f}; rows with a single segment: {mx['frac_rows_single_segment']:.4f}.\n")
    L.append("## Reference conventions\n")
    L.append("| convention | mean global ctx | mean local ctx | attention GFLOP/token | F (GFLOP/token) | F / F_paper |")
    L.append("|---|---:|---:|---:|---:|---:|")
    L.append(f"| paper (apex3/flops.py) | {pp['mean_global_ctx']:.1f} | {pp['mean_local_ctx']:.1f} | "
             f"{pp['attn_gflop_per_token']:.4f} | {pp['F_gflop']:.4f} | 1.00000 |")
    L.append(f"| no-mask exact (one 8192-token document per row) | {nm['mean_global_ctx']:.1f} | "
             f"{nm['mean_local_ctx']:.4f} | {nm['attn_gflop_per_token']:.4f} | {nm['F_gflop']:.4f} | {nm['F_ratio']:.5f} |")
    L.append(f"| masked, phase-1 mixture (this run) | {mx['mean_global_ctx']:.1f} | {mx['mean_local_ctx']:.4f} | "
             f"{mx['attn_gflop_per_token']:.4f} | {mx['F_gflop']:.4f} | {mx['F_ratio']:.5f} |\n")
    L.append("## Per domain\n")
    L.append("| domain | weight | rows | mean global ctx | mean local ctx | attn GFLOP/tok | F GFLOP/tok | F/F_paper "
             "(95% normal) | segs/row | mean seg | median seg | token-wtd mean seg | frac p<1024 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d, r in o["domains"].items():
        ci = r["F_ratio_ci95_normal"]
        L.append(f"| {d} | {r['weight']:.3f} | {r['rows']} | {r['mean_global_ctx']:.1f} | {r['mean_local_ctx']:.2f} | "
                 f"{r['attn_gflop_per_token']:.4f} | {r['F_gflop']:.4f} | {r['F_ratio']:.5f} ({ci[0]:.5f} to {ci[1]:.5f}) | "
                 f"{r['segments_per_row']:.2f} | {r['mean_seg_len']:.1f} | {r['median_seg_len']:.1f} | "
                 f"{r['token_weighted_mean_seg_len']:.0f} | {r['frac_tokens_p_lt_1024']:.4f} |")
    L.append(f"| **mixture** | 1.000 | {sum(m['rows_used'].values())} | {mx['mean_global_ctx']:.1f} | "
             f"{mx['mean_local_ctx']:.2f} | {mx['attn_gflop_per_token']:.4f} | {mx['F_gflop']:.4f} | "
             f"{mx['F_ratio']:.5f} ({mx['F_ratio_ci95_normal'][0]:.5f} to {mx['F_ratio_ci95_normal'][1]:.5f}) | "
             f"{mx['segments_per_row']:.2f} | {mx['mean_seg_len']:.1f} | {mx['median_seg_len']:.1f} | | "
             f"{mx['frac_tokens_p_lt_1024']:.4f} |\n")
    L.append("Segment statistics count every varlen piece as FA3 sees it, including the row-truncated first and last "
             "pieces; 'token-wtd mean seg' is the mean segment length experienced by a token (sum L^2 / sum L). "
             "Mixture median is a row-weight-corrected median over segments.\n")
    c = o["corrected_mfu"]
    L.append("## Corrected MFU\n")
    L.append("Real training: corrected = reported x F_masked / F_paper "
             f"(factor {c['factor_masked']:.5f}; interval from the bootstrap CI of the factor).\n")
    L.append("| interval | instance | reported MFU % | corrected MFU % | 95% interval |")
    L.append("|---|---|---:|---:|---|")
    for r in c["training"]:
        lo, hi = r["corrected_mfu_ci95_bootstrap"]
        L.append(f"| {r['interval']} | {r['instance']} | {r['reported_mfu']:.2f} | {r['corrected_mfu']:.2f} | {lo:.2f} to {hi:.2f} |")
    L.append("")
    L.append("Benchmark (synthetic, one document per row, no masking): corrected = reported x F_nomask_exact / F_paper "
             f"(factor {c['factor_nomask_exact']:.5f}).\n")
    L.append("| stage | reported MFU % | corrected MFU % |")
    L.append("|---|---:|---:|")
    for r in c["benchmark"]:
        L.append(f"| {r['stage']} | {r['reported_mfu']:.2f} | {r['corrected_mfu']:.2f} |")
    L.append("")
    L.append("## Method and caveats\n")
    L.append("- Rows: `*.bin` = flat little-endian uint32 tokens; a domain stream is the sorted concatenation of its shards; "
             "window w = stream[w*8193 : w*8193+8193], input = first 8192 tokens (apex3/data.py TokenStream.read, "
             "PackedBatches.window, pack_rows). The loader walks windows in a seeded permutation, so uniform random "
             "windows reproduce its row distribution; shards are never padded and rows are not aligned to documents.")
    L.append("- Segments: positions restart at row start (even mid-document) and on the token after every EOS; "
             "cu_seqlens = indices with position 0 (positions_from_tokens, build_attn_meta). The EOS token is the last "
             "token of the segment it ends.")
    L.append("- ctx counts ideal causal keys; FA3 executes whole tiles, so executed FLOPs per token are somewhat higher "
             "than this count in segments shorter than a tile, and the correction describes algorithmic FLOPs, the same "
             "convention as the paper's formula.")
    L.append("- Mixture = phase-1 weights of server/train_gh200.sh; earlier training intervals are assumed to have used it. "
             "Intervals reflect row-sampling error only (rows within a domain resampled).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
