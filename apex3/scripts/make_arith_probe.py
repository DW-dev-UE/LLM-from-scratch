from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from apex3.arith_probe import build_probe, make_problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-cat", type=int, default=64, help="problems per category")
    p.add_argument("--shots", type=int, default=4, help="solved examples before each question")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)

    def encode(text):
        e = tok.encode(text, add_special_tokens=False)
        return e.ids, e.offsets

    probe = build_probe(make_problems(a.per_cat, a.shots, a.seed), encode, seed=a.seed)
    np.savez(a.out, **probe)
    offsets = np.concatenate([[0], np.cumsum(probe["lens"])])
    with open(a.out + ".jsonl", "w") as f:
        for i in range(len(probe["lens"])):
            ids = probe["ids"][offsets[i] : offsets[i + 1]].tolist()
            s = int(probe["starts"][i])
            f.write(json.dumps({"cat": str(probe["names"][probe["cats"][i]]), "prompt": tok.decode(ids[:s]), "answer": tok.decode(ids[s:]), "tokens": len(ids), "answer_tokens": len(ids) - s}) + "\n")
    print(f"{a.out}: {len(probe['lens'])} problems, {int(probe['lens'].sum())} tokens " f"(max {int(probe['lens'].max())}, answer tokens {int((probe['lens'] - probe['starts']).sum())})")
    for k, name in enumerate(probe["names"]):
        m = probe["cats"] == k
        print(f"  {name}: {int(m.sum())} problems, mean {probe['lens'][m].mean():.0f} tokens, " f"answer {(probe['lens'][m] - probe['starts'][m]).mean():.1f} tokens")


if __name__ == "__main__":
    main()
