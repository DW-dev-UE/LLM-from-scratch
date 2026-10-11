from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

LETTERS = "ABCDEFGHIJ"


def _load(path, config=None, split="test"):
    from datasets import load_dataset

    try:
        return load_dataset(path, config, split=split)
    except RuntimeError:
        return load_dataset(path, config, split=split, revision="refs/convert/parquet")


def prepare_tasks() -> dict[str, list[dict]]:
    tasks = {}
    tasks["mmlu"] = [{"type": "mc", "context": f"Question: {r['question'].strip()}\nAnswer:", "choices": [f" {c}" for c in r["choices"]], "answer": int(r["answer"]), "subject": r["subject"]} for r in _load("cais/mmlu", "all")]
    arc = []
    for r in _load("allenai/ai2_arc", "ARC-Challenge"):
        labels = list(r["choices"]["label"])
        if r["answerKey"] in labels:
            arc.append({"type": "mc", "context": f"Question: {r['question'].strip()}\nAnswer:", "choices": [f" {t}" for t in r["choices"]["text"]], "answer": labels.index(r["answerKey"])})
    tasks["arc_challenge"] = arc
    tasks["hellaswag"] = [{"type": "mc", "context": f"{r['activity_label']}: {r['ctx']}".strip(), "choices": [f" {e}" for e in r["endings"]], "answer": int(r["label"])} for r in _load("Rowan/hellaswag", split="validation") if str(r["label"]).isdigit()]
    tasks["gsm8k"] = [{"type": "ref", "context": f"Question: {r['question'].strip()}\nAnswer:", "target": " " + r["answer"].strip()} for r in _load("openai/gsm8k", "main")]
    tasks["humaneval"] = [{"type": "ref", "context": r["prompt"], "target": r["canonical_solution"], "task_id": r["task_id"]} for r in _load("openai/openai_humaneval")]
    tasks["triviaqa"] = [{"type": "ref", "context": f"Question: {r['question'].strip()}\nAnswer:", "target": " " + r["answer"]["value"].strip()} for r in _load("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")]
    return tasks


def _hub_jsonl(repo: str, prefix: str) -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".jsonl")]
    rows = []
    for fn in files:
        with open(hf_hub_download(repo, fn, repo_type="dataset"), encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def decontam_sets() -> dict[str, list[dict]]:
    from datasets import get_dataset_config_names

    def q(context, target=""):
        return {"context": str(context or ""), "target": str(target or "")}

    def choice(options, i):
        return options[i] if isinstance(i, int) and 0 <= i < len(options) else ""

    math_subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    loaders = {"math": lambda: [q(r["problem"], r["solution"]) for s in math_subjects for r in _load("EleutherAI/hendrycks_math", s)], "mbpp": lambda: [q(r["text"], r["code"]) for split in ("train", "validation", "test", "prompt") for r in _load("google-research-datasets/mbpp", "full", split=split)], "mbpp_plus": lambda: [q(r["prompt"], r["code"]) for r in _load("evalplus/mbppplus")], "humaneval_plus": lambda: [q(r["prompt"], r["canonical_solution"]) for r in _load("evalplus/humanevalplus")], "gpqa": lambda: [q(r["Question"], r["Correct Answer"]) for r in _load("Idavidrein/gpqa", "gpqa_extended", split="train")], "mmlu_pro": lambda: [q(r["question"], choice(r["options"], r["answer_index"])) for r in _load("TIGER-Lab/MMLU-Pro")], "medqa": lambda: [q(r["question"], r["answer"]) for r in _load("GBaker/MedQA-USMLE-4-options")], "arc_easy": lambda: [q(r["question"], choice(r["choices"]["text"], list(r["choices"]["label"]).index(r["answerKey"]) if r["answerKey"] in r["choices"]["label"] else -1)) for r in _load("allenai/ai2_arc", "ARC-Easy")], "openbookqa": lambda: [q(r["question_stem"], " ".join(r["choices"]["text"])) for r in _load("allenai/openbookqa", "main")],
               "commonsense_qa": lambda: [q(r["question"], " ".join(r["choices"]["text"])) for r in _load("tau/commonsense_qa", split="validation")], "piqa": lambda: [q(r["goal"], f"{r['sol1']} {r['sol2']}") for r in _load("ybisk/piqa", split="validation")], "winogrande": lambda: [q(r["sentence"], f"{r['option1']} {r['option2']}") for r in _load("allenai/winogrande", "winogrande_xl", split="validation")], "lambada": lambda: [q(r["text"]) for r in _load("EleutherAI/lambada_openai", "default")], "ifeval": lambda: [q(r["prompt"]) for r in _load("google/IFEval", split="train")], "cruxeval": lambda: [q(r["code"], f"{r['input']} {r['output']}") for r in _load("cruxeval-org/cruxeval")], "livecodebench": lambda: [q(r["question_content"], r.get("starter_code", "")) for r in _hub_jsonl("livecodebench/code_generation_lite", "test")], "aime": lambda: ([q(r["problem"], r["solution"]) for r in _load("HuggingFaceH4/aime_2024", split="train")] + [q(r["problem"], r["answer"]) for r in _load("yentinglin/aime_2025", split="train")]), "bbh": lambda: [q(r["input"], r["target"]) for c in get_dataset_config_names("lukaemon/bbh") for r in _load("lukaemon/bbh", c)], "multipl_e_cpp": lambda: [q(r["prompt"]) for c in ("humaneval-cpp", "mbpp-cpp") for r in _load("nuprl/MultiPL-E", c)]}
    out = {}
    for name, load in loaders.items():
        try:
            out[name] = [r for r in load() if r["context"].strip() or r["target"].strip()]
        except Exception as e:
            print(f"decontam_{name}: skipped ({type(e).__name__}: {str(e)[:200]})")
    return out


def cmd_prepare(a):
    os.makedirs(a.out, exist_ok=True)
    sets = {**prepare_tasks(), **{f"decontam_{k}": v for k, v in decontam_sets().items()}}
    for name, rows in sets.items():
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")


def load_tasks(evals_dir: str, limit: int) -> dict[str, list[dict]]:
    out = {}
    for fn in sorted(os.listdir(evals_dir)):
        if fn.endswith(".jsonl") and not fn.startswith("decontam_"):
            with open(os.path.join(evals_dir, fn), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if limit and len(rows) > limit:
                rows = [rows[i] for i in np.linspace(0, len(rows) - 1, limit).astype(int)]
            out[fn[: -len(".jsonl")]] = rows
    return out


def load_vals(specs: list[str]):
    import glob

    from apex3.data import TokenStream

    vals = {}
    for spec in specs:
        name, pattern = spec.split("=", 1)
        stats = [json.load(open(p + ".json")) for p in sorted(glob.glob(pattern)) if os.path.exists(p + ".json")]
        bpt = sum(s["bytes"] for s in stats) / max(1, sum(s["tokens"] for s in stats)) if stats else None
        vals[name] = (TokenStream(pattern), bpt)
    return vals


def checkpoints(a) -> list[str]:
    from apex3.checkpoint import STATE_FILE

    if a.ckpt:
        return a.ckpt
    if not os.path.isdir(a.ckpt_dir):
        return []
    return sorted(os.path.join(a.ckpt_dir, d) for d in os.listdir(a.ckpt_dir) if os.path.exists(os.path.join(a.ckpt_dir, d, STATE_FILE)))


def evaluate_checkpoint(path: str, a, tasks, vals, encode, prev_usage):
    import torch

    from apex3.agent.backend import load_model
    from apex3.checkpoint import read_trainer_state
    from apex3.config import parse_overrides, resolve_config
    from apex3.evaluate import Scorer, eval_mc, eval_ref, routing_metrics

    state = read_trainer_state(path)
    cfg, _ = resolve_config(state, a.config, None, parse_overrides(a.set))
    model = load_model(cfg, path, a.device, experts=a.experts)
    scorer = Scorer(model, row_len=a.row_len, rows_per_batch=a.rows_per_batch)
    result = {"checkpoint": os.path.basename(path.rstrip("/")), "step": state.get("step"), "tokens": state.get("tokens"), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()
    domains, pathways = {}, {}
    for name, (stream, bpt) in vals.items():
        w = scorer.windows(stream, a.eos_id, a.windows, a.seq_len)
        domains[name] = {"ce": w["ce"], "bpb": w["ce"] / math.log(2) / bpt if bpt else None, "icl": w["icl"], "early_ce": w["early_ce"], "late_ce": w["late_ce"]}
        if w["pathways"] is not None:
            pathways[name] = w["pathways"]
    result["domains"] = domains
    usage = None
    if pathways:
        result["routing"], usage = routing_metrics(pathways, prev_usage)
    result["tasks"] = {name: (eval_mc if rows[0]["type"] == "mc" else eval_ref)(scorer, encode, rows) for name, rows in tasks.items() if rows}
    result["seconds"] = round(time.time() - t0)
    scorer.close()
    del model, scorer
    torch.cuda.empty_cache()
    return result, usage


def cmd_run(a):
    from apex3.tokens import load_tokenizer

    tok = load_tokenizer(a.tokenizer, raw_text=True)

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False).ids

    tasks = load_tasks(a.evals, a.limit) if a.evals else {}
    vals = load_vals(a.val)
    out_dir = a.out or os.path.join(a.ckpt_dir or os.path.dirname(a.ckpt[0]), "eval")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "eval.jsonl")
    while True:
        done = set()
        if os.path.exists(log_path):
            with open(log_path) as f:
                done = {json.loads(line)["checkpoint"] for line in f if line.strip()}
        todo = [p for p in checkpoints(a) if os.path.basename(p.rstrip("/")) not in done]
        for path in todo:
            prev = sorted(f for f in os.listdir(out_dir) if f.startswith("routing_") and f.endswith(".npz"))
            prev_usage = dict(np.load(os.path.join(out_dir, prev[-1]))) if prev else None
            result, usage = evaluate_checkpoint(path, a, tasks, vals, encode, prev_usage)
            if usage is not None:
                np.savez(os.path.join(out_dir, f"routing_{result['step'] or 0:08d}.npz"), **usage)
            with open(log_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            summary = {k: round(v["bpb"], 4) for k, v in result["domains"].items() if v["bpb"]}
            print(f"{result['checkpoint']}: bpb {summary} tasks " f"{ {k: round(v.get('acc', v.get('logprob_per_token', 0)), 4) for k, v in result['tasks'].items()} }", flush=True)
        if not a.watch:
            return
        time.sleep(a.poll)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.set_defaults(fn=cmd_prepare)
    s.add_argument("--out", required=True)
    s = sub.add_parser("run")
    s.set_defaults(fn=cmd_run)
    s.add_argument("--ckpt-dir", default=None)
    s.add_argument("--ckpt", nargs="*", default=None, help="specific checkpoints instead of --ckpt-dir")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--evals", default=None, help="directory from `prepare`")
    s.add_argument("--val", nargs="*", default=[], help='name="glob of held-out *.bin" per domain')
    s.add_argument("--eos-id", type=int, required=True)
    s.add_argument("--out", default=None, help="default: <ckpt-dir>/eval")
    s.add_argument("--limit", type=int, default=1000, help="examples per task")
    s.add_argument("--windows", type=int, default=128, help="held-out windows per domain")
    s.add_argument("--seq-len", type=int, default=2048)
    s.add_argument("--row-len", type=int, default=4096)
    s.add_argument("--rows-per-batch", type=int, default=8)
    s.add_argument("--config", default="C", help="preset for checkpoints without model_config")
    s.add_argument("--set", nargs="*", default=[])
    s.add_argument("--device", default="cuda")
    s.add_argument("--experts", default="grouped_mm", choices=["grouped_mm", "loop"])
    s.add_argument("--watch", action="store_true")
    s.add_argument("--poll", type=int, default=600, help="seconds between checks with --watch")
    a = p.parse_args()
    if a.cmd == "run" and not (a.ckpt_dir or a.ckpt):
        p.error("--ckpt-dir or --ckpt is required")
    a.fn(a)


if __name__ == "__main__":
    main()
