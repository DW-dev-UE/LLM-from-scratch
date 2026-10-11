from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator

import figdata as fd
import figstyle as fs

W = fs.TEXT_WIDTH_IN
TOK_MAX = 11.0


def _worker_handles() -> list[Line2D]:
    return [Line2D([], [], color=fs.WORKER[w], lw=1.4, label=f"Worker {w}") for w in "AB"]


def _phases(xmax: float = TOK_MAX) -> list[tuple[float, float, str]]:
    ev = fd.read_csv("train_events.csv")
    ev = ev[ev["hosts"].isin(["A", "AB"])].sort_values("tok_B")
    starts = ev["tok_B"].tolist()
    return list(zip(starts, starts[1:] + [xmax], ev["short"].tolist()))


def _bands(axes, phases) -> None:
    for i, (lo, hi, _) in enumerate(phases):
        if i % 2:
            for ax in axes:
                ax.axvspan(lo, hi, facecolor=fs.BAND, edgecolor="none", linewidth=0, zorder=0)


def _stage_tick(stage: str, label: str) -> str:
    label = label.replace(" -> ", " \u2192 ")
    label = re.sub(r"(\d)x(\d)", "\\1\u00d7\\2", label)
    tag = "6\u2032" if stage == "6b" else stage
    return f"{tag}  {label}"


def _bf16_bound_pct() -> float:
    tokens, flop_per_token, peak, batch, floor_s = 49152, 21.57674496e9, 989e12, 8e6, 1.760
    rows = fd.bench_rows("B")
    plan = rows[rows["file"].astype(str).str.startswith("bench_plan_10090423")]
    t_opt = float(plan["optimizer_s_median"].median())
    return 100 * tokens * flop_per_token / ((floor_s + t_opt * tokens / batch) * peak)


def fig_ladder():
    legend_h = 0.30
    bound = _bf16_bound_pct()
    height = 3.02
    st = fd.read_csv("bench_stages.csv")
    st["stage"] = st["stage"].astype(str)
    mp = fd.read_csv("stage_train_map.csv")
    mp["stage"] = mp["stage"].astype(str)
    steps = {w: fd.train_log(w)["steps"] for w in "AB"}
    row_of = {s: i for i, s in enumerate(st["stage"])}

    fig = plt.figure(figsize=(W, height), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[height - legend_h, legend_h], hspace=0)
    ax = top.subplots()

    y = np.arange(len(st))
    colors = [to_rgba(fs.KIND[k], 0.45 if a == 1 else 1.0) for k, a in zip(st["kind"], st["accum"])]
    ax.barh(y, st["mfu_pct"], height=0.62, color=colors, zorder=3)
    for yi, v in zip(y, st["mfu_pct"]):
        one_decimal = Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        ax.text(v + 0.8, yi, f"{one_decimal}", ha="left", va="center", fontsize=7, color=fs.INK)

    for _, r in mp.iterrows():
        yi = row_of[r["stage"]]
        for w, dy, face in (("A", -0.18, fs.INK), ("B", +0.18, "white")):
            m = fd.segment_medians(steps[w], r["tok_from_B"], r["tok_to_B"]) * 100
            if np.isfinite(m):
                ax.plot([m], [yi + dy], ls="none", marker="D", ms=3.0, mfc=face, mec=fs.INK, mew=0.6, zorder=4)

    ax.axvline(bound, color=fs.MUTED, lw=0.8, ls="--", zorder=2)
    ax.annotate(f"BF16 bound at 700 W ({bound:.1f}%)", xy=(bound, 1), xycoords=("data", "data"), xytext=(-3, 0), textcoords="offset points", ha="right", va="center", fontsize=7, color=fs.MUTED, zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    ax.set_xlim(0, 63)
    ax.set_xticks(range(0, 51, 10))
    ax.set_ylim(len(st) - 0.4, -0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([_stage_tick(s, l) for s, l in zip(st["stage"], st["label"])])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Benchmark MFU (%)")
    fs.value_grid(ax, "x")

    kinds = [Patch(facecolor=fs.KIND["system"], label="System optimization"), Patch(facecolor=fs.KIND["model"], label="Model change"), Patch(facecolor=fs.KIND["precision"], label="Precision")]
    notes = [Patch(facecolor=to_rgba(fs.BLUE, 0.45), label="Measured with accum = 1 (accumulation cost hidden)"),
             (Line2D([], [], ls="none", marker="D", ms=3.0, mfc=fs.INK, mec=fs.INK, mew=0.6), Line2D([], [], ls="none", marker="D", ms=3.0, mfc="white", mec=fs.INK, mew=0.6))]
    kw = dict(frameon=False, columnspacing=2.0, borderaxespad=0.0, borderpad=0.1)
    bottom.legend(kinds, [h.get_label() for h in kinds], loc="upper center", ncol=3, handlelength=1.4, **kw)
    bottom.legend(notes, ["Measured with accum = 1 (accumulation cost hidden)", "Real training median (A, B)"], loc="lower center", ncol=2, handlelength=2.0, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, **kw)
    return fs.save(fig, "fig_ladder")


def fig_train_mfu():
    phases = _phases()
    fig, (ax, bx) = fs.figure(3.0, nrows=2, sharex=True, height_ratios=[3, 1], layout="constrained")
    _bands((ax, bx), phases)

    for w in "AB":
        steps = fd.train_log(w)["steps"]
        for _, g in steps[steps["clean"]].groupby("seg"):
            ax.plot(g["tok_B"], g["mfu"] * 100, color=fs.WORKER[w], lw=1.0)

    for lo, hi, lab in phases:
        ax.text((lo + hi) / 2, 0.955, lab, transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=7, color=fs.DARK)
    ax.set_ylim(34, 51)
    ax.set_yticks([35, 40, 45, 50])
    ax.set_ylabel("MFU (%)")
    fs.value_grid(ax)
    ax.legend(handles=_worker_handles(), loc="lower right")

    a = fd.train_log("A")["steps"]
    a = a[~a["aborted"]].sort_values("tok_B", kind="stable")
    bx.step(a["tok_B"], a["bs_M"], where="post", color=fs.DARK, lw=0.9)
    bx.set_ylim(0, 3)
    bx.set_yticks([0, 1, 2, 3])
    bx.set_ylabel("Batch / worker\n(M tokens)", linespacing=1.1)
    bx.set_xlim(0, TOK_MAX)
    bx.set_xticks(range(0, 11))
    bx.set_xlabel("Tokens (billions)")
    fs.value_grid(bx)

    fs.panel_label(ax, "a")
    fs.panel_label(bx, "b")
    fig.align_ylabels((ax, bx))
    return fs.save(fig, "fig_train_mfu")


def fig_loss():
    fig, (ax, bx) = fs.figure(2.4, ncols=2, width_ratios=[1.7, 1.0], layout="constrained")

    for lo, _, _ in _phases()[1:]:
        ax.axvline(lo, color=fs.GRID, lw=0.9, ls=":", zorder=1)
    for w in "AB":
        s = fd.train_log(w)["steps"]
        s = s[~s["aborted"]].sort_values("tok_B", kind="stable")
        ax.scatter(s["tok_B"], s["ce"], s=2, color=fs.WORKER[w], alpha=0.25, linewidths=0, zorder=2)
        ax.plot(s["tok_B"], s["ce"].rolling(5, center=True, min_periods=1).median(), color=fs.WORKER[w], lw=1.1, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(2.3, 13.5)
    ax.set_yticks([2.5, 3, 4, 6, 8, 12])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, TOK_MAX)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("Tokens (billions)")
    ax.set_ylabel("Training loss (CE)")
    fs.value_grid(ax)
    ax.legend(handles=_worker_handles(), loc="upper right")

    domains = ["code", "web", "textbook", "pdf", "math", "papers", "medical", "qa", "books"]
    names = ["Code", "Web", "Textbook", "PDF", "Math", "Papers", "Medical", "Q&A", "Books"]
    val = fd.train_log("A")["val"]
    v5 = val[(val["tok_B"] - 5.0).abs() < 0.1].set_index("domain")["ce"].loc[domains]
    v10 = val[(val["tok_B"] - 10.0).abs() < 0.1].set_index("domain")["ce"].loc[domains]
    y = np.arange(len(domains))
    bx.barh(y - 0.19, v5.values, height=0.36, color=fs.GRAY, label="5.0B tokens", zorder=3)
    bx.barh(y + 0.19, v10.values, height=0.36, color=fs.BLUE, label="10.0B tokens", zorder=3)
    bx.set_ylim(len(domains) - 0.4, -0.6)
    bx.set_yticks(y)
    bx.set_yticklabels(names)
    bx.tick_params(axis="y", length=0)
    bx.set_xlim(0, 4.5)
    bx.set_xticks(range(0, 5))
    bx.set_xlabel("Validation loss (CE)")
    fs.value_grid(bx, "x")
    bx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.0, columnspacing=1.0, borderaxespad=0.1, borderpad=0.1)

    fs.panel_label(ax, "a")
    fs.panel_label(bx, "b")
    return fs.save(fig, "fig_loss")


def fig_power():
    fig, (ax, bx) = fs.figure(2.5, nrows=2, sharex=True, layout="constrained")
    cap, limit, clock_max = {}, 0.0, 0.0
    for w in "AB":
        p = fd.power(w)
        cap[w] = 100 * p["sw_cap"].mean()
        limit = max(limit, p["limit_w"].max())
        clock_max = max(clock_max, p["sm_mhz"].max())
        ax.plot(p["t_s"], p["power_w"], color=fs.WORKER[w], lw=0.8)
        bx.plot(p["t_s"], p["sm_mhz"], color=fs.WORKER[w], lw=0.8)

    ax.axhline(limit, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(119.5, limit + 14, f"Enforced limit {limit:,.0f} W", ha="right", va="bottom", fontsize=7, color=fs.MUTED)
    ax.text(0.5, limit + 14, f"SW power cap active: A {cap['A']:.1f}%, B {cap['B']:.1f}% of samples", ha="left", va="bottom", fontsize=7, color=fs.MUTED)
    ax.set_ylim(150, 790)
    ax.set_yticks(range(200, 701, 100))
    ax.set_ylabel("Power (W)")
    fs.value_grid(ax)
    ax.legend(handles=_worker_handles(), loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1, borderpad=0.1)

    bx.axhline(clock_max, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
    bx.text(119.5, clock_max + 14, f"Max SM clock {clock_max:,.0f} MHz", ha="right", va="bottom", fontsize=7, color=fs.MUTED)
    bx.set_ylim(1350, 2100)
    bx.set_yticks(range(1400, 2001, 200))
    bx.yaxis.set_major_formatter(FuncFormatter(fs.thousands))
    bx.set_ylabel("SM clock (MHz)")
    bx.set_xlim(0, 120)
    bx.set_xticks(range(0, 121, 20))
    bx.set_xlabel("Time (s)")
    fs.value_grid(bx)

    fs.panel_label(ax, "a")
    fs.panel_label(bx, "b")
    fig.align_ylabels((ax, bx))
    return fs.save(fig, "fig_power")
