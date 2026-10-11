from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.transforms import blended_transform_factory

import figdata as fd
import figstyle as fs

W = fs.TEXT_WIDTH_IN
ANN = 7.0
AXIS_LABEL = 8.0

MICRO_TOKENS = 49152
FLOP_PER_TOKEN = 21.57674496e9
PEAK_FLOPS = 989e12
BATCH_TOKENS = 8e6
FLOOR_MS = 1760
OPT_PLAN_FILE = "bench_plan_10090423"
MINUS = "−"


def _axes_in(fig, left, bottom, width, height):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


def _is_dark(color: str) -> bool:
    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.30


def _signed(v: float, digits: int = 1) -> str:
    s = f"{v:+.{digits}f}"
    return s.replace("-", MINUS)


_BENCH_SRC = re.compile(r"([AB]) (\S+):(\d+)")
_BENCH_CACHE: dict[str, object] = {}


def _bench_row(src: str):
    m = _BENCH_SRC.fullmatch(str(src).strip())
    if m is None:
        raise ValueError(f"cannot parse benchmark source {src!r} (expected '<host> <file>:<row>')")
    host, file, row = m.group(1), m.group(2), int(m.group(3))
    if host not in _BENCH_CACHE:
        _BENCH_CACHE[host] = fd.bench_rows(host)
    df = _BENCH_CACHE[host]
    sel = df[(df["file"] == file) & (df["row"] == row)]
    if len(sel) != 1:
        raise ValueError(f"benchmark source {src!r}: {len(sel)} matching rows in bench_and_train_{host}.tsv")
    return sel.iloc[0]


def _bench_value(src: str, column: str) -> float:
    v = float(_bench_row(src)[column])
    if not np.isfinite(v):
        raise ValueError(f"benchmark source {src!r}: {column} is not finite")
    return v


def _bench_mfu(src: str) -> float:
    return _bench_value(src, "mfu_at_batch")


def _mfu_floor_bound() -> float:
    b = fd.bench_rows("B")
    sel = b[b["file"].astype(str).str.startswith(OPT_PLAN_FILE)]
    if sel.empty or sel["optimizer_s_median"].isna().all():
        raise ValueError(f"no optimizer_s_median rows for {OPT_PLAN_FILE} in bench_and_train_B.tsv")
    t_opt = float(sel["optimizer_s_median"].median())
    step_s = FLOOR_MS / 1000.0 + t_opt * MICRO_TOKENS / BATCH_TOKENS
    return MICRO_TOKENS * FLOP_PER_TOKEN / (step_s * PEAK_FLOPS)


def _category_axis(ax, labels, spine: bool = True) -> None:
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=4)
    if not spine:
        ax.spines["left"].set_visible(False)


def fig_time_budget():
    kb = fd.read_csv("kernel_breakdown.csv")
    mt = fd.read_csv("micro_times.csv")
    xmax = 2800

    fig_h = 2.86
    fig = plt.figure(figsize=(W, fig_h))
    left, right = 1.70, 0.40
    axw = W - left - right
    ax_b = _axes_in(fig, left, 0.43, axw, 1.24)
    ax_a = _axes_in(fig, left, 1.96, axw, 0.52)

    ticks = np.arange(0, 2501, 500)
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(fs.thousands))
        fs.value_grid(ax, "x")

    configs = list(dict.fromkeys(kb["config"]))
    for i, cfg in enumerate(configs):
        sub = kb[kb["config"] == cfg].set_index("category")["ms"]
        x0 = 0.0
        for cat, col in fs.KERNEL.items():
            v = float(sub[cat])
            ax_a.barh(i, v, left=x0, height=0.52, color=col, edgecolor="white", linewidth=0.5, zorder=2)
            if v >= 150:
                ax_a.text(x0 + v / 2, i, f"{v:,.0f}", ha="center", va="center", fontsize=ANN, color="white" if _is_dark(col) else fs.INK, zorder=3)
            x0 += v
        ax_a.text(x0 + 25, i, f"{x0:,.0f} ms", ha="left", va="center", fontsize=ANN, color=fs.INK)
    ax_a.set_ylim(len(configs) - 0.5, -0.5)
    _category_axis(ax_a, configs)
    ax_a.tick_params(axis="x", labelbottom=False)
    fs.panel_label(ax_a, "a")

    handles = [Patch(facecolor=c, label=k) for k, c in fs.KERNEL.items()]
    ax_a.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 2.675 / fig_h), bbox_transform=fig.transFigure, ncol=7, borderaxespad=0.0, fontsize=7, handlelength=0.9, handleheight=0.8, handletextpad=0.4, columnspacing=0.9)

    n = len(mt)
    for i, r in mt.iterrows():
        if np.isfinite(r["kernel_ms"]) and np.isfinite(r["idle_ms"]):
            ax_b.barh(i, r["kernel_ms"], height=0.66, color=fs.BLUE, edgecolor="white", linewidth=0.5, zorder=2)
            ax_b.barh(i, r["idle_ms"], left=r["kernel_ms"], height=0.66, color=fs.LIGHT, edgecolor="white", linewidth=0.5, zorder=2)
            end = r["kernel_ms"] + r["idle_ms"]
        else:
            ax_b.barh(i, r["total_ms"], height=0.66, color=fs.MUTED, edgecolor="white", linewidth=0.5, zorder=2)
            end = r["total_ms"]
        mfu1 = Decimal(str(r["mfu_pct"])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        ax_b.text(end + 25, i, f"MFU {mfu1}%", ha="left", va="center", fontsize=ANN, color=fs.INK)
    ax_b.set_ylim(n - 0.5, -1.35)
    _category_axis(ax_b, list(mt["config"]))
    ax_b.set_xlabel("Time per 48K-token microbatch (ms)", fontsize=AXIS_LABEL)
    fs.panel_label(ax_b, "b")

    bound = _mfu_floor_bound()
    ax_b.axvline(FLOOR_MS, color=fs.INK, linestyle=(0, (3.5, 2.0)), linewidth=0.9, zorder=4)
    ax_b.text(FLOOR_MS - 22, -0.88, f"BF16 GEMM + attention floor at 700 W (≤ {bound * 100:.1f}% MFU)", ha="right", va="center", fontsize=ANN, color=fs.INK)

    ax_b.legend(handles=[Patch(facecolor=fs.BLUE, label="GPU kernels"), Patch(facecolor=fs.LIGHT, label="GPU idle"), Patch(facecolor=fs.MUTED, label="Total (no trace)")],
                loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.15, fontsize=ANN, handlelength=1.0, handleheight=0.9, columnspacing=1.3)
    print(f"  fig_time_budget: MFU bound at the {FLOOR_MS:,} ms floor = {bound * 100:.2f}%")

    return fs.save(fig, "fig_time_budget")


def fig_fp8():
    att = fd.read_csv("fp8_attempts.csv")
    ctl = fd.read_csv("b_control.csv")

    fig = plt.figure(figsize=(W, 2.4))
    ax_a = _axes_in(fig, 1.56, 0.46, 2.02, 1.66)
    ax_b = _axes_in(fig, 4.30, 0.46, 1.95, 1.66)

    # Panel (a): throughput change = t_BF16 / t_config - 1 from the median microbatch times,
    # each against the BF16 run under the same conditions (source_bf16 in fp8_attempts.csv).
    fp8 = np.array([_bench_mfu(s) for s in att["source_fp8"]])
    bf16 = np.array([_bench_mfu(s) for s in att["source_bf16"]])
    for name, e8, e16, r8, r16 in zip(att["name"], fp8, bf16, att["mfu_fp8"], att["mfu_bf16"]):
        if abs(e8 * 100 - r8) > 0.1 or abs(e16 * 100 - r16) > 0.1:
            raise ValueError(f"{name}: exact MFU {e8 * 100:.3f}/{e16 * 100:.3f} disagrees with CSV {r8}/{r16}")
    for name, s8, s16, tok in zip(att["name"], att["source_fp8"], att["source_bf16"], att["micro_tokens"]):
        for src in (s8, s16):
            r = _bench_row(src)
            if int(r["micro_bs"]) * int(r["seq_len"]) != int(tok):
                raise ValueError(f"{name}: {src} does not run a {int(tok):,}-token microbatch")
    t_fp8 = np.array([_bench_value(s, "micro_s_median") for s in att["source_fp8"]])
    t_bf16 = np.array([_bench_value(s, "micro_s_median") for s in att["source_bf16"]])
    gain = (t_bf16 / t_fp8 - 1.0) * 100.0
    print("  fig_fp8 (a) throughput change t_BF16/t_config - 1 (%):", ", ".join(f"{nm}: {g:+.3f} ({b:.3f} s/{f:.3f} s)" for nm, g, b, f in zip(att["name"], gain, t_bf16, t_fp8)))
    n = len(att)
    colors = [fs.GREEN if g > 0 else fs.RED for g in gain]
    ax_a.barh(range(n), gain, height=0.62, color=colors, zorder=2)
    for i, g in enumerate(gain):
        if g < 0:
            ax_a.text(g - 0.9, i, _signed(g) + "%", ha="right", va="center", fontsize=ANN, color=fs.INK)
        else:
            ax_a.text(g + 0.9, i, _signed(g) + "%", ha="left", va="center", fontsize=ANN, color=fs.INK)
    ax_a.axvline(0, color=fs.INK, linewidth=0.6, zorder=3)
    ax_a.set_xlim(-44, 12)
    ax_a.set_ylim(n - 0.5, -0.5)
    ax_a.xaxis.set_major_locator(MultipleLocator(10))
    fs.value_grid(ax_a, "x")
    labels = [nm.replace(" (budget", "\n(budget") for nm in att["name"]]
    _category_axis(ax_a, labels, spine=False)
    ax_a.tick_params(axis="y", pad=5)
    ax_a.set_xlabel("Throughput change vs BF16 (%)", fontsize=AXIS_LABEL)
    fs.panel_label(ax_a, "a")

    # Panel (b): the first column of each spec row is the series key in b_control.csv, the third the label drawn.
    ctl = ctl[~((ctl["code"] == "next11") & (ctl["precision"] == "bf16"))]
    spec = [("BF16 + stash", fs.DARK, "BF16 + storage", -0.004), ("FP8 tw (no stash)", fs.PURPLE, "FP8 tensor-wise\n(no storage)", 0.0),
            ("FP8 tw + partial stash (old code)", fs.TEAL, "FP8 tensor-wise +\npartial storage\n(old code)", +0.004), ("FP8 tw + full stash (fixed code)", fs.GREEN, "FP8 tensor-wise + full storage\n(fixed code)", 0.0)]
    for name, col, lab, dx in spec:
        s = ctl[ctl["series"] == name]
        if s.empty:
            raise ValueError(f"series {name!r} not found in b_control.csv")
        mean = s.groupby("budget")["mfu_pct"].mean().sort_index()
        if name != "FP8 tw (no stash)":
            ax_b.plot(mean.index + dx, mean.values, color=col, linewidth=1.2, zorder=2)
        ax_b.plot(s["budget"] + dx, s["mfu_pct"], linestyle="none", marker="o", markersize=4.2, color=col, markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        x_end, y_end = mean.index[-1] + dx, mean.values[-1]
        if name == "FP8 tw + full stash (fixed code)":
            ax_b.text(x_end + 0.0025, y_end + 0.22, lab, ha="right", va="bottom", fontsize=ANN, color=col, linespacing=1.05)
        else:
            ax_b.text(x_end + 0.0035, y_end, lab, ha="left", va="center", fontsize=ANN, color=col, linespacing=1.05)
    ax_b.set_xlim(0.085, 0.215)
    ax_b.set_xticks([0.10, 0.15, 0.20])
    ax_b.set_xticklabels(["0.10", "0.15", "0.20"])
    ax_b.set_ylim(44.5, 52.2)
    ax_b.set_yticks(range(45, 52))
    fs.value_grid(ax_b, "y")
    ax_b.set_xlabel("Activation-memory budget", fontsize=AXIS_LABEL)
    ax_b.set_ylabel("Benchmark MFU (%)", fontsize=AXIS_LABEL, labelpad=3)
    fs.panel_label(ax_b, "b")

    return fs.save(fig, "fig_fp8")


def fig_cpu_accum():
    runs = fd.read_csv("cpu_accum_runs.csv")
    bench = fd.bench_rows("A")[["file", "row", "mfu_at_batch"]]
    joined = runs.merge(bench, on=["file", "row"], how="left", validate="many_to_one")
    assert joined["mfu_at_batch"].notna().all(), "unmatched benchmark rows"
    mb = fd.read_csv("cpu_add_microbench.csv")
    stages = fd.read_csv("bench_stages.csv")
    st4 = stages[stages["stage"].astype(str) == "4"]
    if len(st4) != 1:
        raise ValueError(f'bench_stages.csv: expected exactly one stage "4" row, found {len(st4)}')
    gpu_micro_s = float(st4["micro_s"].iloc[0])

    fig = plt.figure(figsize=(W, 2.1))
    ax_a = _axes_in(fig, 1.58, 0.46, 1.06, 1.44)
    ax_b = _axes_in(fig, 4.38, 0.46, 1.89, 1.44)

    groups = list(dict.fromkeys(joined["group"]))
    rng = np.random.default_rng(7)
    for gi, g in enumerate(groups):
        v = joined.loc[joined["group"] == g, "mfu_at_batch"].to_numpy() * 100.0
        y = gi + rng.uniform(-0.08, 0.08, size=len(v))
        col = fs.BLUE if g == "C++ OpenMP kernel" else fs.MUTED
        ax_a.plot(v, y, linestyle="none", marker="o", markersize=4.2, color=col, alpha=0.85, markeredgewidth=0, zorder=3)
    ax_a.set_xlim(26, 35.5)
    ax_a.set_ylim(len(groups) - 0.5, -1.25)
    ax_a.set_xticks([26, 28, 30, 32, 34])
    fs.value_grid(ax_a, "x")
    _category_axis(ax_a, groups)
    ax_a.set_xlabel("Benchmark MFU (%)", fontsize=AXIS_LABEL)
    fs.panel_label(ax_a, "a")
    ax_a.text(26.3, -0.85, "slow mode", ha="left", va="center", fontsize=ANN, color=fs.MUTED)
    ax_a.text(35.4, -0.85, "fast mode", ha="right", va="center", fontsize=ANN, color=fs.MUTED)

    n = len(mb)
    names, cols = [], []
    for _, r in mb.iterrows():
        cpp = r["method"].startswith("C++")
        name = r["method"].replace("->", "→")
        names.append(f"{name} ({int(r['threads'])} threads)" if cpp else name)
        cols.append(fs.BLUE if cpp else fs.GRAY)
    secs = mb["seconds_21B_elements"].to_numpy()
    ax_b.barh(range(n), secs, height=0.64, color=cols, zorder=2)
    for i, s in enumerate(secs):
        ax_b.text(s + 0.02, i, f"{s:.2f} s", ha="left", va="center", fontsize=ANN, color=fs.INK)
    for i, alt in enumerate(mb["alt_seconds"].to_numpy(dtype=float)):
        if np.isfinite(alt):
            ax_b.plot([alt], [i], linestyle="none", marker="o", markersize=3.8, markerfacecolor="white", markeredgecolor=fs.MUTED, markeredgewidth=0.8, zorder=3)
            ax_b.text(alt - 0.05, i + 0.38, "other run", ha="left", va="top", fontsize=ANN, color=fs.MUTED)
    ax_b.axvline(gpu_micro_s, color=fs.MUTED, linestyle=(0, (3.5, 2.0)), linewidth=0.9, zorder=1)
    ax_b.text(gpu_micro_s - 0.04, -1.25, f"GPU compute per 32K\nmicrobatch ({gpu_micro_s:.2f} s)", ha="right", va="center", fontsize=ANN, color=fs.MUTED, linespacing=1.05)
    ax_b.set_xlim(0, 2.4)
    ax_b.set_ylim(n - 0.5, -1.95)
    ax_b.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax_b.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.1f}"))
    fs.value_grid(ax_b, "x")
    _category_axis(ax_b, names)
    ax_b.set_xlabel("Time to add 21B elements (s)", fontsize=AXIS_LABEL)
    fs.panel_label(ax_b, "b")

    return fs.save(fig, "fig_cpu_accum")


def fig_diloco():
    sa = fd.train_log("A")["syncs"].sort_values("sync").reset_index(drop=True)
    sb = fd.train_log("B")["syncs"].sort_values("sync").reset_index(drop=True)
    assert list(sa["sync"]) == list(sb["sync"])
    ymax = 650

    fig = plt.figure(figsize=(W, 2.3))
    left, right = 0.50, 0.06
    ax = _axes_in(fig, left, 0.46, W - left - right, 1.58)

    phases = [("write_s", fs.SYNC["write"], "Write pseudo-gradient"), ("push_s", fs.SYNC["push"], "Push to peer"), ("update_s", fs.SYNC["update"], "Outer update"), ("wait_s", fs.SYNC["wait"], "Wait for peer")]
    bw, off = 0.36, 0.20
    ticks, tick_labels = [], []
    for df, dx, tag in ((sa, -off, "A"), (sb, +off, "B")):
        x = df["sync"].to_numpy() + dx
        bottom = np.zeros(len(df))
        for k, (col_name, color, _lab) in enumerate(phases):
            v = df[col_name].to_numpy(dtype=float)
            ax.bar(x, v, bottom=bottom, width=bw, color=color, edgecolor="white", linewidth=0.4, zorder=2)
            bottom += v
        ticks += list(x)
        tick_labels += [tag] * len(x)
    ax.set_xlim(0.45, 6.55)
    ax.set_ylim(0, ymax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_formatter(FuncFormatter(fs.thousands))
    fs.value_grid(ax, "y")

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for s in sa["sync"]:
        ax.text(s, -0.125, f"Sync {int(s)}", transform=trans, ha="center", va="top", fontsize=7.5, color=fs.INK)

    a6 = sa[sa["sync"] == 6].iloc[0]
    total = a6["total_s"]
    assert total > ymax
    xl, xr = 6 - off - bw / 2, 6 - off + bw / 2
    yb = 585
    ax.plot([xl - 0.06, xr + 0.06], [yb - 12, yb + 12], color="white", linewidth=2.6, solid_capstyle="butt", zorder=5, clip_on=False)
    ax.text(xl - 0.17, 612, f"{total:,.0f} s total (waited for B’s benchmark pause)", ha="right", va="center", fontsize=ANN, color=fs.DARK)
    ax.plot([xl - 0.14, xl - 0.015], [612, 590], color=fs.MUTED, linewidth=0.6, zorder=4)

    ax.set_ylabel("Time per sync (s)", fontsize=AXIS_LABEL)
    handles = [Patch(facecolor=c, label=lab) for _k, c, lab in phases]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, borderaxespad=0.15, fontsize=ANN, handlelength=1.0, handleheight=0.9, columnspacing=1.3)

    return fs.save(fig, "fig_diloco")
