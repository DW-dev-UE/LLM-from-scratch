from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"

TEXT_WIDTH_IN = 6.3

BLUE = "#4E79A7"
ORANGE = "#F28E2B"
RED = "#E15759"
TEAL = "#76B7B2"
GREEN = "#59A14F"
YELLOW = "#EDC948"
PURPLE = "#B07AA1"
PINK = "#FF9DA7"
BROWN = "#9C755F"
GRAY = "#BAB0AC"

INK = "#222222"
DARK = "#4D4D4D"
MUTED = "#7F7F7F"
LIGHT = "#D0D0D0"
GRID = "#E8E8E8"
BAND = "#F4F4F4"

WORKER = {"A": BLUE, "B": ORANGE}
KIND = {"system": BLUE, "model": ORANGE, "precision": GREEN}
KERNEL = {"Dense GEMM": BLUE, "Grouped GEMM (experts)": TEAL, "Inductor fused kernels": ORANGE, "Attention (FA3)": GREEN, "CuTe copy": PURPLE, "Other": GRAY, "GPU idle": LIGHT}
SYNC = {"write": TEAL, "push": PURPLE, "update": BROWN, "wait": LIGHT}


def setup() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": DARK, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5, "xtick.color": DARK, "ytick.color": DARK, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False, "legend.handlelength": 1.5, "legend.handletextpad": 0.5, "legend.columnspacing": 1.2, "legend.borderaxespad": 0.3, "lines.linewidth": 1.2, "lines.markersize": 3.5, "patch.linewidth": 0.0, "hatch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "axes.unicode_minus": True})


def figure(height_in: float, width_in: float = TEXT_WIDTH_IN, **kwargs):
    return plt.subplots(figsize=(width_in, height_in), **kwargs)


def value_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, letter: str, dx_pt: float = -4, dy_pt: float = 4) -> None:
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx_pt, dy_pt), textcoords="offset points", ha="right", va="bottom", fontsize=9, fontweight="bold", color=INK)


def thousands(x, _pos=None) -> str:
    return f"{x:,.0f}"


_HANGUL = re.compile("[ᄀ-ᇿ㄰-㆏가-힣]")


def _check_english(fig) -> None:
    bad = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if s and _HANGUL.search(s):
            bad.append(s)
    if bad:
        raise ValueError(f"figure contains Hangul text (figures must be English): {bad[:5]}")


def save(fig, name: str) -> list[Path]:
    _check_english(fig)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = [FIGS / f"{name}.pdf", FIGS / f"{name}.png"]
    fig.savefig(out[0])
    fig.savefig(out[1], dpi=300)
    plt.close(fig)
    return out
