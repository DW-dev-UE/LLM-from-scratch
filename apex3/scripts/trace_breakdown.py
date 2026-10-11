from __future__ import annotations

import argparse
import json
from collections import defaultdict

BUCKETS = [("ce_triton", ("_ce_grad_kernel",)), ("attention", ("flash", "fmha")), ("inductor_fused", ("triton_",)), ("elementwise_copy", ("copyatom",)), ("gemm", ("nvjet", "xmma")), ("grouped_gemm", ("group",)),
           ("gemm", ("gemm", "cutlass", "wgmma", "cublas")), ("sort", ("radix", "sort")), ("index_gather_scatter", ("index", "scatter", "gather")), ("reduce", ("reduce",)), ("elementwise_copy", ("elementwise", "copy", "fill"))]


def bucket(name: str) -> str:
    low = name.lower()
    for key, subs in BUCKETS:
        if any(s in low for s in subs):
            return key
    return "other"


def load(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def union_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if e <= end:
            continue
        total += e - max(s, end)
        end = e
    return total


def breakdown(events: list[dict], top: int = 5) -> dict:
    kernels, copies = [], defaultdict(float)
    for ev in events:
        if ev.get("ph") != "X" or "dur" not in ev:
            continue
        cat = str(ev.get("cat", "")).lower()
        if cat == "kernel":
            kernels.append((float(ev["ts"]), float(ev["dur"]), ev.get("name", "?")))
        elif cat in ("gpu_memcpy", "gpu_memset"):
            copies[cat] += float(ev["dur"])
    if not kernels:
        raise SystemExit("no kernel events: was the trace taken with bench_mfu.py --trace?")
    start = min(t for t, _, _ in kernels)
    window = max(t + d for t, d, _ in kernels) - start
    busy = union_us([(t, t + d) for t, d, _ in kernels])
    per, names = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: defaultdict(float))
    for _, d, n in kernels:
        b = bucket(n)
        per[b][0] += d
        per[b][1] += 1
        names[b][n[:120]] += d
    rows = sorted(((b, us, cnt) for b, (us, cnt) in per.items()), key=lambda r: -r[1])
    return {"window_ms": window / 1e3, "busy_ms": busy / 1e3, "busy_frac": busy / window, "kernel_sum_ms": sum(us for _, us, _ in rows) / 1e3, "buckets": [{"bucket": b, "ms": us / 1e3, "frac_of_window": us / window, "count": cnt, "top": sorted(((n, t / 1e3) for n, t in names[b].items()), key=lambda x: -x[1])[:top]} for b, us, cnt in rows], "memcpy_ms": copies["gpu_memcpy"] / 1e3, "memset_ms": copies["gpu_memset"] / 1e3}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--micro", type=int, default=1, help="micro-batches in the trace (bench --accum): per-micro times")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--json", default=None, help="append the breakdown as one JSON line")
    args = p.parse_args()
    r = breakdown(load(args.trace), args.top)
    r["trace"], r["micro"] = args.trace, args.micro
    m = args.micro
    print(f"window {r['window_ms'] / m:.1f} ms/micro, busy {r['busy_frac']:.1%} " f"(idle {r['window_ms'] / m - r['busy_ms'] / m:.1f} ms/micro), kernel sum {r['kernel_sum_ms'] / m:.1f}, " f"memcpy {r['memcpy_ms'] / m:.1f}, memset {r['memset_ms'] / m:.1f} ms/micro")
    print(f"{'bucket':<22}{'ms/micro':>10}{'window':>9}{'count':>9}")
    for b in r["buckets"]:
        print(f"{b['bucket']:<22}{b['ms'] / m:>10.1f}{b['frac_of_window']:>9.1%}{b['count']:>9}")
    for b in r["buckets"]:
        print(f"\n[{b['bucket']}]")
        for n, ms in b["top"]:
            print(f"  {ms / m:9.2f} ms/micro  {n}")
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
