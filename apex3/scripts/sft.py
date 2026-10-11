from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.distributed.tensor import DTensor

from apex3.checkpoint import (latest_checkpoint, load_checkpoint, load_model_weights, read_trainer_state, save_checkpoint)
from apex3.config import ImplConfig, check_against_checkpoint, parse_overrides, resolve_config
from apex3.data import to_device
from apex3.model import Apex3
from apex3.model import layers as L
from apex3.model.loss import IGNORE_INDEX
from apex3.model.moe import update_router_biases
from apex3.optim import OptimizerSet, wsd_lr
from apex3.parallel import all_reduce_sum, apply_fsdp, build_mesh, init_distributed, set_grad_sync
from apex3.sft.data import SFTRows, loss_weight


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="C")
    p.add_argument("--attn-ratio", default=None)
    p.add_argument("--set", nargs="*", default=[], help="ModelConfig overrides, e.g. rope_scale_global=8")
    p.add_argument("--init", default=None, help="pretraining checkpoint (model weights only)")
    p.add_argument("--data", required=True, help="train dir from sft_prepare.py")
    p.add_argument("--val-data", default=None)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--global-rows", type=int, default=64, help="rows per optimizer step (x seq_len tokens)")
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--final-lr-ratio", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--router-bias-rate", type=float, default=0.0)
    p.add_argument("--attn", default="fa2", choices=["fa2", "fa3", "sdpa"])
    p.add_argument("--experts", default="grouped_mm", choices=["grouped_mm", "loop"])
    p.add_argument("--recompute", default="full", choices=["none", "full"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--ckpt-every", type=int, default=500, help="steps")
    p.add_argument("--resume", default=None, help="checkpoint path (default: latest in --ckpt-dir)")
    p.add_argument("--val-every", type=int, default=200, help="steps")
    p.add_argument("--val-batches", type=int, default=None, help="micro-batches per rank (default: one pass)")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def n_targets(batch: dict) -> int:
    return int((batch["labels"] != IGNORE_INDEX).sum())


@torch.no_grad()
def evaluate(model, val: SFTRows, n_batches: int) -> float:
    model.eval()
    val.micro_step = 0
    total = torch.zeros(2, device="cuda")
    for _ in range(n_batches):
        b = next(val)
        n = n_targets(b)
        total += torch.stack([model(**to_device(b))["ce"] * n, total.new_tensor(n)])
    model.train()
    total = all_reduce_sum(total)
    return (total[0] / total[1].clamp(min=1)).item()


def main():
    args = parse_args()
    rank, world, _ = init_distributed()
    log = print if rank == 0 else (lambda *a, **k: None)
    resume = args.resume or (latest_checkpoint(args.ckpt_dir) if args.ckpt_dir else None)
    if not (resume or args.init):
        raise ValueError("--init is required for a fresh run")
    source = read_trainer_state(resume or args.init)
    cfg, from_ckpt = resolve_config(source, args.config, args.attn_ratio, parse_overrides(args.set))
    if from_ckpt:
        log(f"model config from {resume or args.init} (--config/--attn-ratio ignored, --set applied)")
    impl = ImplConfig(attn=args.attn, experts=args.experts, recompute=args.recompute)
    if args.compile:
        L.enable_compile()
    torch.manual_seed(args.seed)

    with torch.device("meta"):
        model = Apex3(cfg, impl)
    apply_fsdp(model, build_mesh(world, args.shard_size))
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()
    opt = OptimizerSet(model, lr=args.lr, weight_decay=args.wd, use_muon=args.optimizer == "muon")
    data = SFTRows(args.data, args.micro_bs, rank, world, args.seed)
    val = SFTRows(args.val_data, args.micro_bs, rank, world, 1, one_pass=True) if args.val_data else None
    val_batches = args.val_batches or (math.ceil(val.n_rows / (args.micro_bs * world)) if val else 0)

    rows_per_micro = args.micro_bs * world
    if args.global_rows % rows_per_micro:
        raise ValueError(f"--global-rows {args.global_rows} is not a multiple of micro_bs x world = {rows_per_micro}")
    accum = args.global_rows // rows_per_micro
    total_steps = math.ceil(args.epochs * data.n_rows / args.global_rows)
    warmup = max(1, round(args.warmup_frac * total_steps))

    schedule = {"global_rows": args.global_rows, "epochs": args.epochs, "data": os.path.abspath(args.data)}
    step = 0
    if resume:
        state = load_checkpoint(resume, model, opt)
        for change in check_against_checkpoint(cfg, state.get("model_config")):
            log(f"config change from checkpoint: {change}")
        changed = [f"{k}: {state[k]} -> {v}" for k, v in schedule.items() if k in state and state[k] != v]
        if changed:
            raise ValueError(f"cannot resume with a different schedule ({'; '.join(changed)})")
        step = state["step"]
        rows = state.get("rows_seen", state["micro_step"] * rows_per_micro)
        if rows % rows_per_micro:
            raise ValueError(f"{rows} rows consumed is not a multiple of micro_bs x world = {rows_per_micro}")
        data.micro_step = rows // rows_per_micro
        log(f"resumed from {resume}: step {step}")
    else:
        check_against_checkpoint(cfg, source.get("model_config"))
        load_model_weights(args.init, model)
        log(f"initialized from {args.init}")
    log(f"{data.n_rows} rows x {data.seq_len} tokens, {args.epochs} epochs: {total_steps} steps of {args.global_rows} " f"rows (accum {accum}), warmup {warmup}")

    def checkpoint_now():
        path = os.path.join(args.ckpt_dir, f"step_{step:08d}")
        save_checkpoint(path, model, opt, {"step": step, "micro_step": data.micro_step, "rows_seen": data.rows_seen(), "seq_len": data.seq_len, "model_config": asdict(cfg), **schedule})
        log(f"saved {path}")

    t_last = time.perf_counter()
    while step < total_steps:
        batches = [next(data) for _ in range(accum)]
        counts = [n_targets(b) for b in batches]
        n_global = all_reduce_sum(torch.tensor(float(sum(counts)), device="cuda")).item()
        parts = torch.zeros(3, device="cuda")
        for m, (b, n) in enumerate(zip(batches, counts)):
            set_grad_sync(model, m == accum - 1)
            out = model(**to_device(b))
            (out["loss"] * loss_weight(n, n_global, world)).backward()
            parts += torch.stack([out["ce"] * n, out["z"] * n, out["aux"] / accum])
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        lr = wsd_lr(step + 1, args.lr, warmup, total_steps, 1 - warmup / total_steps, args.final_lr_ratio)
        opt.set_lr(lr)
        opt.step()
        opt.zero_grad()
        moe_stats, _ = update_router_biases(model, args.router_bias_rate)
        step += 1

        if step % args.log_every == 0:
            ce_sum, z_sum, aux = all_reduce_sum(parts).tolist()
            now = time.perf_counter()
            tok_s = args.log_every * args.global_rows * data.seq_len / (now - t_last) / world
            t_last = now
            extra = " ".join(f"{k}={v:.3g}" for k, v in moe_stats.items())
            log(f"step {step}/{total_steps} ep {data.micro_step * rows_per_micro / data.n_rows:.2f} " f"ce {ce_sum / max(n_global, 1):.4f} z {z_sum / max(n_global, 1):.3g} aux {aux / world:.3g} " f"gnorm {grad_norm.item():.3g} lr {lr:.3g} targets {n_global / 1e3:.0f}K tok/s/gpu {tok_s:.0f} " f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB {extra}")
        if val is not None and (step % args.val_every == 0 or step == total_steps):
            log(f"[val] step {step} ce {evaluate(model, val, val_batches):.4f}")
        if args.ckpt_dir and step % args.ckpt_every == 0:
            checkpoint_now()

    if args.ckpt_dir and step % args.ckpt_every:
        checkpoint_now()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
