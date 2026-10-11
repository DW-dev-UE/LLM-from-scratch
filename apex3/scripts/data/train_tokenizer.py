from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apex3.datapipe.io import ShardWriter, list_files, read_docs
from apex3.tokens import NUM_SPECIAL, SPECIAL_TOKENS, load_tokenizer

STAGE1_REGEX = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


def _doc_text(doc: dict) -> str:
    return "\n".join(f["text"] for f in doc["files"]) if doc.get("files") else doc["text"]


def _parse_domains(items: list[str]) -> dict[str, tuple[float, str]]:
    out = {}
    for item in items:
        name, rest = item.split("=", 1)
        weight, pattern = rest.split(":", 1)
        out[name] = (float(weight), pattern)
    return out


def cmd_sample(a):
    rng = random.Random(a.seed)
    domains = _parse_domains(a.domain)
    total_w = sum(w for w, _ in domains.values())
    train_dir = os.path.join(a.out, "train")
    os.makedirs(train_dir, exist_ok=True)
    report = {}
    for name, (weight, pattern) in domains.items():
        files = list_files(pattern)
        rng.shuffle(files)
        budget, val_budget = a.total_gb * 1e9 * weight / total_w, a.val_gb * 1e9
        used = val_used = 0
        part, f = 0, None
        with ShardWriter(os.path.join(a.out, "val"), name) as val:
            for path in files:
                for doc in read_docs(path):
                    text = _doc_text(doc)
                    n = len(text.encode())
                    if val_used < val_budget:
                        val.write({"text": text, "domain": name})
                        val_used += n
                        continue
                    if f is None or f.tell() >= a.file_mb * 1e6:
                        if f:
                            f.close()
                        f = open(os.path.join(train_dir, f"{name}_{part:05d}.txt"), "w", encoding="utf-8")
                        part += 1
                    f.write(text + "\n\n")
                    used += n
                    if used >= budget:
                        break
                if used >= budget:
                    break
        if f:
            f.close()
        report[name] = {"train_bytes": used, "val_bytes": val_used, "files": part}
    report["total_train_bytes"] = sum(r["train_bytes"] for r in report.values())
    with open(os.path.join(a.out, "sample.json"), "w") as out:
        json.dump(report, out, indent=1)
    print(json.dumps(report))


def _finalize(tok, out_dir: str, vocab_size: int) -> None:
    from tokenizers import decoders, normalizers

    tok.normalizer = normalizers.NFC()
    tok.decoder = decoders.ByteLevel()
    learned = tok.get_vocab_size()
    tok.add_special_tokens(SPECIAL_TOKENS)
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(f"vocab {tok.get_vocab_size()} (learned {learned} + {NUM_SPECIAL}) != {vocab_size}")
    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    print(f"saved {out_dir}/tokenizer.json: learned {learned}, total {vocab_size}")


def cmd_bpe(a):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(STAGE1_REGEX), behavior="isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    trainer = trainers.BpeTrainer(vocab_size=a.vocab_size - NUM_SPECIAL, show_progress=True, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train(list_files(os.path.join(a.corpus, "*.txt")), trainer)
    _finalize(tok, os.path.join(a.out, "bpe_hf"), a.vocab_size)


def cmd_superbpe(a):
    from tokenizers import Tokenizer

    learned = a.vocab_size - NUM_SPECIAL
    s1, s2 = os.path.abspath(os.path.join(a.out, "stage1")), os.path.abspath(os.path.join(a.out, "stage2"))
    num_bytes = a.num_bytes or json.load(open(os.path.join(a.corpus, "..", "sample.json")))["total_train_bytes"]

    def run(args):
        print(" ".join(args), flush=True)
        subprocess.run([a.superbpe_python, "-m", "train_tokenizer", *args], cwd=a.superbpe_repo, check=True)

    run(["--output_dir", s1, "--corpus_dir", os.path.abspath(a.corpus), "--num_bytes", str(int(num_bytes)), "--vocab_size", str(learned), "--regex_string", STAGE1_REGEX])
    inherit = a.transition - NUM_SPECIAL - 256
    os.makedirs(s2, exist_ok=True)
    with open(os.path.join(s1, "merges.txt"), encoding="utf-8") as f:
        lines = f.readlines()
    header = [ln for ln in lines[:1] if ln.startswith("#")]
    with open(os.path.join(s2, "merges.txt"), "w", encoding="utf-8") as f:
        f.writelines(header + lines[len(header) : len(header) + inherit])
    if a.stage2_fraction < 1:
        with open(os.path.join(s1, "meta.json")) as f:
            meta = json.load(f)
        corpus2 = os.path.join(a.out, "stage2_corpus")
        os.makedirs(corpus2, exist_ok=True)
        files, total = [], 0
        for src in meta["train_files"]:
            keep = int(os.path.getsize(src) * a.stage2_fraction)
            dst = os.path.join(corpus2, os.path.basename(src))
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                data = fi.read(keep)
                cut = data.rfind(b"\n")
                fo.write(data[: cut + 1] if cut >= 0 else data)
            files.append(dst)
            total += os.path.getsize(dst)
        with open(os.path.join(s2, "meta.json"), "w") as f:
            json.dump({"total_bytes": total, "train_files": files}, f)
        print(f"stage 2 corpus: {total / 1e9:.2f} GB ({a.stage2_fraction:.0%} of stage 1)", flush=True)
    else:
        shutil.copy(os.path.join(s1, "meta.json"), os.path.join(s2, "meta.json"))
    run(["--output_dir", s2, "--vocab_size", str(learned), "--regex_string", STAGE2_REGEX])

    _finalize(Tokenizer.from_file(os.path.join(s1, "tokenizer.json")), os.path.join(a.out, "bpe"), a.vocab_size)
    _finalize(Tokenizer.from_file(os.path.join(s2, "tokenizer.json")), os.path.join(a.out, "superbpe"), a.vocab_size)


def cmd_g0(a):
    sup, bpe = load_tokenizer(a.superbpe), load_tokenizer(a.bpe)
    weights = {k: float(v) for k, v in (w.split("=") for w in a.weights)}
    report = {}
    for domain in weights:
        texts = [d["text"] for p in list_files(os.path.join(a.val_dir, f"{domain}_*.jsonl.zst")) for d in read_docs(p)]
        n_sup = sum(len(e.ids) for e in sup.encode_batch(texts, add_special_tokens=False))
        n_bpe = sum(len(e.ids) for e in bpe.encode_batch(texts, add_special_tokens=False))
        n_bytes = sum(len(t.encode()) for t in texts)
        report[domain] = {"docs": len(texts), "reduction": 1 - n_sup / max(1, n_bpe), "bytes_per_token_superbpe": n_bytes / max(1, n_sup), "bytes_per_token_bpe": n_bytes / max(1, n_bpe)}
    weighted = sum(weights[d] * report[d]["reduction"] for d in weights) / sum(weights.values())
    learned = sup.get_vocab_size() - NUM_SPECIAL
    over4 = [t for t in (sup.decode([i]) for i in range(learned)) if len(t.split()) > 4]
    report["weighted_reduction"] = weighted
    report["tokens_over_4_words"] = {"count": len(over4), "examples": over4[:20]}
    report["G0_pass"] = weighted >= 0.20 and all(report[d]["reduction"] >= 0 for d in weights)
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.set_defaults(fn=cmd_sample)
    s.add_argument("--domain", nargs="+", required=True, help='name=weight:"glob"')
    s.add_argument("--total-gb", type=float, default=20)
    s.add_argument("--val-gb", type=float, default=0.2, help="per domain")
    s.add_argument("--file-mb", type=float, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    for name, fn in (("bpe", cmd_bpe), ("superbpe", cmd_superbpe)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--corpus", required=True, help="directory of .txt files from `sample`")
        s.add_argument("--vocab-size", type=int, default=204_800)
        s.add_argument("--out", required=True)
        if name == "superbpe":
            s.add_argument("--superbpe-repo", required=True)
            s.add_argument("--superbpe-python", required=True)
            s.add_argument("--transition", type=int, default=184_320)
            s.add_argument("--num-bytes", type=float, default=None, help="default: everything `sample` wrote")
            s.add_argument("--stage2-fraction", type=float, default=1.0, help="train stage 2 on this fraction of every stage-1 file (stage 2 needs far more memory)")
    s = sub.add_parser("g0")
    s.set_defaults(fn=cmd_g0)
    s.add_argument("--superbpe", required=True)
    s.add_argument("--bpe", required=True)
    s.add_argument("--val-dir", required=True)
    s.add_argument("--weights", nargs="+", required=True, help="domain=weight (the pretraining mixture)")
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
