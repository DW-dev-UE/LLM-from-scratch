from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex3.datapipe.io import list_files, read_docs
from apex3.sft.data import render, write_rows

_W: dict = {}


def _init(tokenizer: str, max_tokens: int, decontam: str | None, val_frac: float):
    from apex3.agent.template import ChatTemplate
    from apex3.datapipe.decontam import Decontaminator
    from apex3.tokens import load_tokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _W.update(template=ChatTemplate(load_tokenizer(tokenizer, raw_text=True)), max_tokens=max_tokens, decontam=Decontaminator(decontam) if decontam else None, val_frac=val_frac)


def _is_val(conv: dict, frac: float) -> bool:
    key = str(conv.get("id") or json.dumps(conv["messages"], sort_keys=True, ensure_ascii=False))
    return frac > 0 and zlib.crc32(key.encode()) / 2**32 < frac


def _render_file(path: str):
    out, drops = {"train": [], "val": []}, Counter()
    for conv in read_docs(path):
        r = render(conv, _W["template"], _W["max_tokens"], _W["decontam"])
        if isinstance(r, str):
            drops[r] += 1
        else:
            out["val" if _is_val(conv, _W["val_frac"]) else "train"].append(r)
    return path, out, drops


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inputs", nargs="+", required=True, help='globs, optionally with "@repeat"')
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--decontam", default=None, help="n-gram table from pipeline.py decontam")
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    repeat: dict[str, int] = {}
    for spec in a.inputs:
        pattern, _, times = spec.rpartition("@") if "@" in spec else (spec, "", "1")
        for path in list_files(pattern):
            repeat[path] = int(times)
    if not repeat:
        sys.exit("no input files")

    from apex3.tokens import load_tokenizer, special_ids

    ids = special_ids(load_tokenizer(a.tokenizer))
    pad_id, think_ids = ids["<|pad|>"], (ids["<|think|>"], ids["<|/think|>"])
    splits = {"train": [], "val": []}
    drops: Counter = Counter()
    per_file = {}
    with Pool(a.workers, _init, (a.tokenizer, a.seq_len + 1, a.decontam, a.val_frac)) as pool:
        for path, out, d in pool.imap_unordered(_render_file, sorted(repeat)):
            splits["train"] += out["train"] * repeat[path]
            splits["val"] += out["val"]
            drops += d
            per_file[path] = {"train": len(out["train"]), "val": len(out["val"]), "repeat": repeat[path], **d}
            print(f"{path}: {len(out['train'])} train, {len(out['val'])} val, dropped {dict(d)}", flush=True)
    for name, convs in splits.items():
        if not convs:
            continue
        meta = write_rows(os.path.join(a.out, name), convs, a.seq_len, pad_id, a.seed, {"dropped": dict(drops), "files": per_file, "tokenizer": os.path.abspath(a.tokenizer)}, think_ids)
        print(f"{name}: {meta['conversations']} conversations -> {meta['rows']} rows, fill {meta['fill']:.3f}, " f"target tokens {meta['target_tokens'] / 1e6:.1f}M of {meta['tokens'] / 1e6:.1f}M, " f"thinking {meta['think_target_frac']:.1%} of targets")


if __name__ == "__main__":
    main()
