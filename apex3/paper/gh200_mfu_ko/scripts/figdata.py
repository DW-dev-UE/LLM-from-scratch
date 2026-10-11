from __future__ import annotations

import io
import json
import re

import pandas as pd

from figstyle import DATA

OUTLIER_PT = 0.015


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def system_diagram() -> dict:
    return json.loads((DATA / "system_diagram.json").read_text(encoding="utf-8"))


def bench_rows(host: str) -> pd.DataFrame:
    lines = []
    with open(DATA / f"bench_and_train_{host}.tsv", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#SEGMENTS"):
                break
            lines.append(line)
    return pd.read_csv(io.StringIO("".join(lines)), sep="\t", na_values=["None"])


_SYNC = re.compile(r"sync (\d+) done: write (\d+)s push (\d+)s wait (\d+)s update (\d+)s fingerprint ([0-9.e+-]+)")
_VAL_HEAD = re.compile(r"\[val\] step (\d+) tok ([0-9.]+)B (.*)")
_VAL_ITEM = re.compile(r"([a-z_]+) ([0-9.]+)")


def train_log(host: str) -> dict:
    rows, syncs, vals = [], [], []
    seg, first_after_restart = 0, False
    with open(DATA / f"train_log_{host}.tsv", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            tag = parts[0]
            if tag == "R":
                seg += 1
                first_after_restart = True
            elif tag == "S":
                step, tok, ce, gn, lr, bs, ts, mfu, mem, moe = parts[1:11]
                rows.append(dict(step=int(step), tok_B=float(tok), ce=float(ce), gnorm=float(gn), lr=float(lr), bs_M=float(bs), tok_s=float(ts), mfu=float(mfu), mem_gib=float(mem), moe_avg=float(moe) if moe else float("nan"), seg=seg, warmup=first_after_restart))
                first_after_restart = False
            elif tag == "D":
                m = _SYNC.search(parts[1])
                if m:
                    n, w, p, wa, u, fp = m.groups()
                    syncs.append(dict(sync=int(n), write_s=int(w), push_s=int(p), wait_s=int(wa), update_s=int(u), fingerprint=float(fp)))
            elif tag == "V":
                m = _VAL_HEAD.search(parts[1])
                if m:
                    step, tok, rest = m.groups()
                    rest = re.sub(r"\([^)]*\)", "", rest)
                    for dom, val in _VAL_ITEM.findall(rest):
                        vals.append(dict(step=int(step), tok_B=float(tok), domain=dom, ce=float(val)))
    steps = pd.DataFrame(rows)
    seg_size = steps.groupby("seg")["step"].transform("size")
    steps["aborted"] = seg_size < 3
    med = steps[~steps.warmup].groupby("seg")["mfu"].median()
    steps["outlier"] = steps["mfu"] < steps["seg"].map(med) - OUTLIER_PT
    steps["clean"] = ~steps["warmup"] & ~steps["outlier"] & ~steps["aborted"]
    sy = pd.DataFrame(syncs)
    if not sy.empty:
        sy["total_s"] = sy[["write_s", "push_s", "wait_s", "update_s"]].sum(axis=1)
    return {"steps": steps, "syncs": sy, "val": pd.DataFrame(vals)}


def power(host: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"power_{host}.csv", comment="#", skipinitialspace=True)
    df.columns = ["timestamp", "power_w", "limit_w", "sm_mhz", "mem_mhz", "util", "temp_c", "mem_mib", "reasons"]
    ts = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    out = pd.DataFrame({"t_s": (ts - ts.iloc[0]).dt.total_seconds()})
    for col, unit in [("power_w", "W"), ("limit_w", "W"), ("sm_mhz", "MHz"), ("mem_mhz", "MHz"), ("util", "%"), ("mem_mib", "MiB")]:
        out[col] = df[col].astype(str).str.replace(unit, "", regex=False).str.strip().astype(float)
    out["temp_c"] = df["temp_c"].astype(float)
    out["reasons"] = df["reasons"].astype(str).str.strip().apply(lambda s: int(s, 16))
    out["sw_cap"] = (out["reasons"] & 0x4) != 0
    return out


def segment_medians(steps: pd.DataFrame, tok_from: float, tok_to: float) -> float:
    sel = steps[steps.clean & (steps.tok_B >= tok_from) & (steps.tok_B <= tok_to)]
    return float(sel["mfu"].median()) if len(sel) else float("nan")
