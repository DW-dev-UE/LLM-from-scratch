from __future__ import annotations

from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

import figdata
import figstyle
from figstyle import BAND, BLUE, DARK, GRAY, INK, LIGHT

W = figstyle.TEXT_WIDTH_IN - 0.06

M = 0.0
GUT = 1.50
CW = (W - 2 * M - GUT) / 2
GX0, GX1 = M, M + CW
CX0, CX1 = GX1 + GUT, GX1 + GUT + CW
GC = (GX1 + CX0) / 2
PAD = 0.09
INS = 0.08

T_TITLE, T_SUB = 0.17, 0.30
T_LINK = 0.385
T0 = 0.54
BLK = 0.30
ROW = 0.15
GROUP_H = BLK + 4 * ROW + 0.04
GAP2 = 0.30
NOTE_GAP, NOTE_H = 0.06, 0.27
GAP3 = 0.10
BOTTOM_PAD = 0.09

T_WEIGHTS = T0
T_GROUP = T0
T_STAGE = T_GROUP + GROUP_H + GAP2
T_NOTE = T_STAGE + BLK + NOTE_GAP
T_LAST = T_NOTE + NOTE_H + GAP3
CONT_H = T_LAST + BLK + BOTTOM_PAD
T_COMPUTE = T_WEIGHTS + BLK + 0.14
COMPUTE_H = (T_NOTE + NOTE_H) - T_COMPUTE

ARROW5_LEN, PEER_H, PEER_W = 0.32, 0.27, 1.80
H = M + CONT_H + ARROW5_LEN + PEER_H + M

TITLE_PT, BLOCK_PT, COMPUTE_PT, ARROW_PT, BADGE_PT = 8.0, 7.5, 7.0, 6.8, 6.8
BADGE_R = 0.0625


def _tint(color: str, alpha: float):
    return to_rgba(color, alpha)


def fig_system() -> None:
    d = figdata.system_diagram()
    gpu, cpu = d["gpu"], d["cpu"]
    flows = {a["id"]: a for a in d["arrows"]}

    fig, ax = figstyle.figure(H, W)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        renderer = fig._get_renderer()

    top = H - M

    def Y(t: float) -> float:
        return top - t

    def width(s: str, size: float, weight: str = "normal") -> float:
        t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
        w = t.get_window_extent(renderer).width / fig.dpi
        t.remove()
        return w

    def text(x, y, s, size, *, ha="left", weight="normal", color=INK, va="center_baseline", maxw=None, z=4):
        if maxw is not None and width(s, size, weight) > maxw + 1e-6:
            raise ValueError(f"text overflows its box ({width(s, size, weight):.3f} > {maxw:.3f} in): {s!r}")
        return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z)

    def wrap2(s: str, size: float, maxw: float) -> list[str]:
        if width(s, size) <= maxw:
            return [s]
        words = s.split(" ")
        best = min(((max(width(" ".join(words[:i]), size), width(" ".join(words[i:]), size)), [" ".join(words[:i]), " ".join(words[i:])]) for i in range(1, len(words))))
        if best[0] > maxw:
            raise ValueError(f"cannot wrap into {maxw:.3f} in: {s!r}")
        return best[1]

    def box(x0, y0, x1, y1, *, fc, ec, lw, r=0.035, ls="-", z=2):
        p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
        ax.add_patch(p)
        return p

    def arrow(p0, p1, both):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=11, lw=0.9, color=INK, shrinkA=0, shrinkB=0, zorder=5))

    def badge(x, y, n):
        ax.add_patch(Circle((x, y), BADGE_R, fc=INK, ec="white", lw=0.6, zorder=6))
        ax.text(x, y, n, fontsize=BADGE_PT, fontweight="bold", color="white", ha="center", va="center_baseline", zorder=7)

    for x0, x1, spec in ((GX0, GX1, gpu["title"]), (CX0, CX1, cpu["title"])):
        box(x0, Y(CONT_H), x1, Y(0), fc=BAND, ec=DARK, lw=0.6, r=0.06, z=1)
        parts = spec.split(" · ")
        text(x0 + PAD, Y(T_TITLE), " · ".join(parts[:2]), TITLE_PT, weight="bold", maxw=CW - 2 * PAD, va="baseline")
        text(x0 + PAD, Y(T_SUB), " · ".join(parts[2:]), COMPUTE_PT, color=DARK, maxw=CW - 2 * PAD, va="baseline")

    bx0g, bx1g = GX0 + PAD, GX1 - PAD
    bx0c, bx1c = CX0 + PAD, CX1 - PAD
    bw = bx1g - bx0g

    state_fc, state_ec = _tint(BLUE, 0.16), BLUE
    buf_fc, buf_ec = _tint(GRAY, 0.22), GRAY

    def block(x0, x1, t0, label, size, *, fc, ec):
        box(x0, Y(t0 + BLK), x1, Y(t0), fc=fc, ec=ec, lw=0.7)
        yc = Y(t0 + BLK / 2)
        nm = text(x0 + INS, yc, label[0], BLOCK_PT)
        text(x1 - INS, yc, label[1], BLOCK_PT, ha="right", color=DARK)
        if width(label[0], BLOCK_PT) + width(label[1], BLOCK_PT) + 3 * INS > x1 - x0:
            raise ValueError(f"block too narrow: {label}")
        return nm

    block(bx0g, bx1g, T_WEIGHTS, gpu["weights"], BLOCK_PT, fc=state_fc, ec=state_ec)
    block(bx0g, bx1g, T_LAST, gpu["activations"], BLOCK_PT, fc=buf_fc, ec=buf_ec)

    box(bx0g, Y(T_COMPUTE + COMPUTE_H), bx1g, Y(T_COMPUTE), fc="white", ec=LIGHT, lw=0.8)
    text(bx0g + INS, Y(T_COMPUTE + 0.115), gpu["compute_header"], COMPUTE_PT, weight="bold")
    sep_t = T_COMPUTE + 0.205
    ax.plot([bx0g + INS, bx1g - INS], [Y(sep_t)] * 2, color=LIGHT, lw=0.5, zorder=3, solid_capstyle="butt")
    indent = 0.10
    items = [wrap2(s, COMPUTE_PT, bw - 2 * INS - indent) for s in gpu["compute"]]
    n_extra = sum(len(i) - 1 for i in items)
    cont = 0.12
    pitch = 0.158
    span = (len(items) - 1) * pitch + n_extra * cont
    t_bottom = T_COMPUTE + COMPUTE_H
    t = sep_t + (t_bottom - sep_t - span) / 2
    for lines in items:
        text(bx0g + INS, Y(t), "•", COMPUTE_PT, color=DARK)
        for k, ln in enumerate(lines):
            text(bx0g + INS + indent, Y(t), ln, COMPUTE_PT, maxw=bw - 2 * INS - indent)
            t += cont if k < len(lines) - 1 else 0
        t += pitch

    gy1, gy0 = Y(T_GROUP), Y(T_GROUP + GROUP_H)
    box(bx0c, gy0, bx1c, gy1, fc=state_fc, ec=state_ec, lw=0.7)
    name, size = cpu["state_group"].rsplit(" (", 1)
    yh = Y(T_GROUP + BLK / 2)
    text(bx0c + INS, yh, name, BLOCK_PT, weight="bold")
    text(bx1c - INS, yh, size.rstrip(")"), BLOCK_PT, ha="right", color=DARK)
    if width(name, BLOCK_PT, "bold") + width(size, BLOCK_PT) + 3 * INS > bx1c - bx0c:
        raise ValueError("state group header too wide")
    ax.plot([bx0c + INS, bx1c - INS], [Y(T_GROUP + BLK)] * 2, color=BLUE, lw=0.5, alpha=0.55, zorder=3, solid_capstyle="butt")
    for i, (nm, sz) in enumerate(cpu["state"]):
        yr = Y(T_GROUP + BLK + ROW * (i + 0.5) + 0.01)
        text(bx0c + INS, yr, nm, BLOCK_PT)
        text(bx1c - INS, yr, sz, BLOCK_PT, ha="right", color=DARK)

    block(bx0c, bx1c, T_STAGE, cpu["staging"], BLOCK_PT, fc=buf_fc, ec=buf_ec)
    box(bx0c, Y(T_NOTE + NOTE_H), bx1c, Y(T_NOTE), fc="white", ec=LIGHT, lw=0.8)
    text((bx0c + bx1c) / 2, Y(T_NOTE + NOTE_H / 2), cpu["compute"], COMPUTE_PT, ha="center", maxw=bx1c - bx0c - 2 * INS)
    block(bx0c, bx1c, T_LAST, cpu["stash"], BLOCK_PT, fc=buf_fc, ec=buf_ec)

    text(GC, Y(T_TITLE), d["link"][0], 7.5, ha="center", weight="bold", va="baseline", maxw=GUT - 0.1)
    text(GC, Y(T_SUB), d["link"][1], COMPUTE_PT, ha="center", color=DARK, va="baseline", maxw=GUT - 0.1)
    xa, xb, yl = GX1 + 0.02, CX0 - 0.02, Y(T_LINK)
    s, hh, hl = 0.0208, 0.050, 0.095
    ax.add_patch(Polygon([(xa, yl), (xa + hl, yl + hh), (xa + hl, yl + s), (xb - hl, yl + s), (xb - hl, yl + hh), (xb, yl), (xb - hl, yl - hh), (xb - hl, yl - s), (xa + hl, yl - s), (xa + hl, yl - hh)], closed=True, fc=GRAY, ec="none", zorder=3))

    def hflow(fid: str, t_row: float, xs: float, xe: float) -> None:
        f = flows[fid]
        y = Y(t_row)
        arrow((xs, y), (xe, y), f["both"])
        lines = wrap2(f["label"], ARROW_PT, GUT - 0.16)
        base = y + BADGE_R + 0.055
        for k, ln in enumerate(reversed(lines)):
            text(GC, base + k * 0.118, ln, ARROW_PT, ha="center", color=DARK, va="baseline", z=4)
        badge((xs + xe) / 2, y, fid)

    hflow("4", T_WEIGHTS + BLK / 2, bx1g, bx0c)
    hflow("1", T_STAGE + BLK / 2, bx1g, bx0c)
    hflow("3", T_LAST + BLK / 2, bx1g, bx0c)

    x2 = bx0c + 0.38
    y_from, y_to = Y(T_STAGE), Y(T_GROUP + GROUP_H)
    arrow((x2, y_from), (x2, y_to), flows["2"]["both"])
    badge(x2, (y_from + y_to) / 2, "2")
    text(x2 + BADGE_R + 0.06, (y_from + y_to) / 2, flows["2"]["label"], ARROW_PT, color=DARK)

    x5 = (CX0 + CX1) / 2
    y5a, y5b = Y(CONT_H), Y(CONT_H + ARROW5_LEN)
    arrow((x5, y5a), (x5, y5b), flows["5"]["both"])
    badge(x5, (y5a + y5b) / 2, "5")
    text(x5 - BADGE_R - 0.06, (y5a + y5b) / 2, flows["5"]["label"], ARROW_PT, ha="right", color=DARK)
    box(x5 - PEER_W / 2, M, x5 + PEER_W / 2, M + PEER_H, fc="white", ec=DARK, lw=0.7, ls=(0, (3, 2)))
    text(x5, M + PEER_H / 2, d["peer"], COMPUTE_PT, ha="center", maxw=PEER_W - 2 * INS)

    figstyle.save(fig, "fig_system")
