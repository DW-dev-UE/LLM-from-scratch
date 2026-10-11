from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def load_features(paths: list[str]) -> dict:
    parts = [np.load(p) for p in paths]
    layers = [tuple(int(i) for i in p["layers"]) for p in parts]
    if len(set(layers)) != 1:
        raise ValueError(f"feature files use different layers: {sorted(set(layers))}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("features", "labels", "groups", "logprob")}
    out["layers"] = list(layers[0])
    out["ckpt"] = str(parts[0]["ckpt"])
    return out


def cmd_features(a):
    from apex3.agent.backend import TorchBackend, load_model
    from apex3.agent.template import VERSION, ChatTemplate
    from apex3.checkpoint import read_trainer_state
    from apex3.config import parse_overrides, resolve_config
    from apex3.confidence import default_layers, last_turn_span, turn_features
    from apex3.datapipe.io import list_files, read_docs
    from apex3.sft.data import conversation_messages
    from apex3.tokens import load_tokenizer

    template = ChatTemplate(load_tokenizer(a.tokenizer, raw_text=True))
    cfg, _ = resolve_config(read_trainer_state(a.ckpt), a.config, None, parse_overrides(a.set))
    layers = a.layers or default_layers(cfg.n_layers)
    model = load_model(cfg, a.ckpt, a.device, experts=a.experts)
    backend = TorchBackend(model, max_len=a.max_len, temperature=0)
    feats, labels, groups, logprobs, skipped = [], [], [], [], Counter()
    for path in list_files(a.inputs):
        for rec in read_docs(path):
            if a.limit and len(labels) >= a.limit:
                break
            msgs = conversation_messages(rec, template)
            turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
            if not turns or "label" not in rec:
                skipped["no_assistant_turn_or_label"] += 1
                continue
            try:
                k = turns[rec.get("turn", -1)]
            except IndexError:
                skipped["bad_turn"] += 1
                continue
            ids, _ = template.render(msgs[: k + 1])
            if len(ids) > a.max_len:
                skipped["too_long"] += 1
                continue
            f, lp = turn_features(backend, ids, last_turn_span(ids, template.sp["<|assistant|>"], template.stop), layers)
            feats.append(f.cpu().numpy())
            labels.append(int(rec["label"]))
            groups.append(str(rec.get("group", rec.get("id", len(labels)))))
            logprobs.append(lp)
    np.savez(a.out, features=np.stack(feats).astype(np.float32), labels=np.array(labels, dtype=np.int8), groups=np.array(groups), logprob=np.array(logprobs, dtype=np.float32), layers=np.array(layers), ckpt=a.ckpt, template=VERSION)
    print(json.dumps({"turns": len(labels), "positive_rate": float(np.mean(labels)) if labels else None, "layers": layers, "skipped": dict(skipped)}))


def cmd_train(a):
    import torch

    from apex3.confidence import ConfidenceHead, fit_logistic, fit_temperature, report, split_by_group

    d = load_features(a.features)
    train, calib, val = split_by_group(d["groups"], a.val_frac, a.calib_frac, a.seed)
    if not train or not calib or not val:
        sys.exit(f"too few tasks for the splits: train {len(train)}, calib {len(calib)}, val {len(val)}")
    x = torch.from_numpy(d["features"]).float()
    y = torch.from_numpy(d["labels"].astype(np.float32))
    mean, std = x[train].mean(0), x[train].std(0).clamp(min=1e-6)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w, b = fit_logistic(((x[train] - mean) / std).to(dev), y[train].to(dev), a.l2, a.max_iter)
    head = ConfidenceHead(d["layers"], mean, std, w, b)
    head.temperature = fit_temperature(head.logits(x[calib]), y[calib]) if a.calibrate else 1.0
    r = report(head.prob(x[val]).numpy(), d["labels"][val], d["logprob"][val], d["groups"][val])
    head.meta = {"ckpt": d["ckpt"], "train_turns": len(train), "calib_turns": len(calib), "val": r}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    head.save(a.out)
    print(json.dumps({"saved": a.out, "temperature": head.temperature, "val": r}, indent=1))


def cmd_eval(a):
    import torch

    from apex3.confidence import ConfidenceHead, gate_g8, report

    head = ConfidenceHead.load(a.head)
    d = load_features(a.features)
    if d["layers"] != head.layers:
        sys.exit(f"features use layers {d['layers']}, the head {head.layers}")
    if head.meta.get("ckpt") and head.meta["ckpt"] != d["ckpt"]:
        print(f"[warning] features from {d['ckpt']}, head trained on {head.meta['ckpt']}")
    r = report(head.prob(torch.from_numpy(d["features"])).numpy(), d["labels"], d["logprob"], d["groups"])
    fails = gate_g8(r, a.auroc_margin, a.max_ece, a.bon_margin)
    print(json.dumps({"heldout": r, "G8": "pass" if not fails else fails}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("features")
    s.set_defaults(fn=cmd_features)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--in", dest="inputs", nargs="+", required=True, help="JSON lines (.jsonl, .zst, .gz) globs")
    s.add_argument("--out", required=True)
    s.add_argument("--layers", type=int, nargs="*", default=None, help="1-indexed (default: 1/2, 3/4, full depth)")
    s.add_argument("--max-len", type=int, default=32_768, help="longest context scored (KV cache size)")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--config", default="C", help="preset for checkpoints without model_config")
    s.add_argument("--set", nargs="*", default=[])
    s.add_argument("--device", default="cuda")
    s.add_argument("--experts", default="grouped_mm", choices=["grouped_mm", "loop"])
    s = sub.add_parser("train")
    s.set_defaults(fn=cmd_train)
    s.add_argument("--features", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--val-frac", type=float, default=0.2, help="tasks held out for the report")
    s.add_argument("--calib-frac", type=float, default=0.2, help="of the rest, tasks for temperature scaling")
    s.add_argument("--l2", type=float, default=1e-3)
    s.add_argument("--max-iter", type=int, default=200)
    s.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("eval")
    s.set_defaults(fn=cmd_eval)
    s.add_argument("--head", required=True)
    s.add_argument("--features", nargs="+", required=True, help="held-out tasks (not used in train)")
    s.add_argument("--auroc-margin", type=float, default=0.05, help="G8: AUROC above the log-prob baseline")
    s.add_argument("--max-ece", type=float, default=0.05, help="G8: calibration error")
    s.add_argument("--bon-margin", type=float, default=0.02, help="G8: best-of-n success above the log-prob pick")
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
