from __future__ import annotations

import argparse
import contextlib
import math
import os
import shutil
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.distributed.tensor import DTensor

from apex3 import arith_probe
from apex3.act_offload import ActivationOffload
from apex3.checkpoint import (latest_checkpoint, load_checkpoint, load_offload_checkpoint, save_checkpoint, save_offload_checkpoint)
from apex3.config import ImplConfig, check_against_checkpoint, get_config, parse_overrides
from apex3.data import build_batches, build_val, pool_strides, to_device
from apex3.diloco import DiLoCo, LocalExchange, SshExchange
from apex3.evaluate import ICL_EARLY, ICL_LATE
from apex3.flops import flops_per_token, mfu, peak_bf16_flops
from apex3.model import Apex3
from apex3.model import layers as L
from apex3.model.loss import IGNORE_INDEX
from apex3.model.moe import Router, load_report, update_router_biases
from apex3.monitor import G2Monitor
from apex3.fsdp_offload import FSDPOffloadOptimizer, build_fsdp_offload_model
from apex3.offload import OffloadOptimizer, build_single_gpu_model
from apex3.optim import OptimizerSet, batch_tokens, router_bias_rate, wsd_lr
from apex3.parallel import (all_reduce_mean, all_reduce_sum, apply_fsdp, build_mesh, init_distributed, set_grad_sync)
from apex3.precision import PRECISIONS, is_te_precision, linear_for, moe_align_for, precision_context

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="P1")
    p.add_argument("--attn-ratio", default=None)
    p.add_argument("--set", nargs="*", default=[], help="ModelConfig overrides, e.g. rope_scale_global=8")
    p.add_argument("--data", nargs="+", required=True, help='one glob of uint32 shards, or a mixture: code=0.35:"/bin/code/*.bin" en=0.4:"..." ...')
    p.add_argument("--pool", nargs="*", default=[], help="keep a fraction of a domain's windows so it repeats, e.g. code=0.2 (G1h repetition arm)")
    p.add_argument("--eos-id", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--total-tokens", type=float, required=True, help="end of the LR schedule")
    p.add_argument("--stop-at-tokens", type=float, default=None, help="stop and checkpoint earlier, e.g. 0.935e12 = end of the stable phase at 1.1T (spec 11.3)")
    p.add_argument("--batch-start", type=float, default=2e6)
    p.add_argument("--batch-end", type=float, default=8e6)
    p.add_argument("--batch-ramp-frac", type=float, default=0.03)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup-tokens", type=float, default=4e9, help="2,000 steps x 2M tokens")
    p.add_argument("--decay-frac", type=float, default=0.15)
    p.add_argument("--bias-freeze-frac", type=float, default=0.03, help="router biases stop updating over this last fraction of --total-tokens (spec 2.2)")
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--muon-momentum", default="fp32", choices=list(DTYPES), help="bf16 saves 4.6 GiB per GPU at 21B (first H100 memory fallback, spec 7.1; check on P1)")
    p.add_argument("--reduce-dtype", default="fp32", choices=list(DTYPES), help="gradient reduce precision; bf16 halves the cross-node all-reduce (spec 7.1)")
    p.add_argument("--attn", default="fa2", choices=["fa2", "fa3", "sdpa"])
    p.add_argument("--experts", default="grouped_mm", choices=["grouped_mm", "fp8_grouped", "loop"], help="fp8_grouped: the routed experts' grouped GEMMs in FP8 (rowwise scales, torchao over torch._scaled_grouped_mm; apex3/model/fp8.py), usually with --precision fp8_rowwise")
    p.add_argument("--te-experts", action="store_true",
                   help="with FP8: routed experts in FP8 too (TE GroupedLinear). Without it FP8 covers the TE linears (attention, shared expert, dense FFN) and the experts stay BF16 grouped_mm: on GH200 the TE experts made a step 1.5x slower than BF16 (RUNLOG_GH200.md)")
    p.add_argument("--recompute", default="none", choices=["none", "full"])
    p.add_argument("--moe-chunk", type=int, default=32768, help="micro-batches with more tokens run the routed experts per chunk (S3 memory, spec 7.1); 0 = off")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--ce-impl", default="torch", choices=["torch", "triton"], help="LM head + cross-entropy: torch reference, or the Triton kernel (BF16 logits overwritten by their gradient, grad_w in an FP32-output GEMM; same math, can switch on resume)")
    p.add_argument("--ce-chunk", type=int, default=1024, help="tokens per LM head chunk (triton: 4096 uses about the memory torch needs at 1024)")
    p.add_argument("--act-budget", type=float, default=0.0,
                   help="with --compile-blocks and --recompute none: torch.compile picks what each block recomputes so its saved activations stay within this fraction of saving everything (torch._functorch.config.activation_memory_budget); cheap elementwise ops go first")
    p.add_argument("--offload-accum", default="gpu", choices=["gpu", "cpu"], help="--offload on one GPU: add gradients into the CPU accumulator with GPU kernels over the link, or copy them with the copy engine and add on the CPU (RUNLOG_GH200.md)")
    p.add_argument("--offload-gpu-share", type=float, default=0.0, help="with --offload-accum cpu: fraction of the parameter elements (first in model order) still added on the GPU (slower on GH200, keep 0 there: RUNLOG_GH200.md)")
    p.add_argument("--compile-blocks", action="store_true", help="torch.compile every block's forward (fuses the elementwise and routing work around the GEMMs)")
    p.add_argument("--act-offload", type=float, default=0.0, help="GiB of pinned CPU memory for activations: save them to CPU memory in forward and fetch them back in backward instead of recomputing (--offload on one GPU, --recompute none; GH200)")
    p.add_argument("--gemm-offload-part", default="all", choices=["all", "experts", "shared"], help="--gemm-offload: which gate/up outputs (experts: the routed ones only, half the bytes on C64; shared: the shared expert and dense FFN)")
    p.add_argument("--gemm-offload", default="none", choices=["none", "hooks", "compiler"],
                   help="with --compile-blocks: save the FFN gate/up GEMM outputs instead of recomputing them and keep them in CPU memory until backward (apex3/gemm_offload.py); hooks uses --act-offload GiB as the arena (about 1.25 GB per 1K tokens of micro-batch for C64)")
    p.add_argument("--offload", action="store_true", help="optimizer state in CPU memory: one GPU (GH200; FP32 master and gradient accumulation in CPU memory, spec v1.10, 7.4) or the GPUs of one node with FSDP2 (e.g. 2x H100, 7.6)")
    p.add_argument("--precision", default="bf16", choices=PRECISIONS,
                   help="fp8_rowwise: attention, shared expert and dense FFN linears in FP8 with rowwise scales (torchao Float8Linear over torch._scaled_mm, no TE; LM head and routers stay as they are). fp8_block (Hopper, TE blockwise) selects the TE linear and expert modules")
    p.add_argument("--fp8-recipe", default="rowwise", choices=["rowwise", "tensorwise", "rowwise_with_gw_hp"], help="--precision fp8_rowwise: torchao scaling recipe for the linears (apex3/model/fp8.py)")
    p.add_argument("--fp8-fast-accum", action="store_true", help="fp8_rowwise: Hopper fast accumulation on the forward GEMMs (torchao's default; off here keeps full FP32 accumulation)")
    p.add_argument("--fp8-weights", action="store_true", help="with --offload and FP8: keep the TE weights in FP8 on the GPU (fp8_model_init), FP32 masters in CPU memory; no per-micro-batch weight quantization (spec v1.10, 7.4)")
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--ckpt-every", type=float, default=20e9, help="tokens")
    p.add_argument("--keep-ckpts", type=int, default=0, help="keep only the last N periodic checkpoints written by this process or resumed from in --ckpt-dir (0 = keep all); the checkpoint at the end of a run is never deleted")
    p.add_argument("--resume", default=None, help="checkpoint path (default: latest in --ckpt-dir)")
    p.add_argument("--diloco-workers", type=int, default=1, help="number of DiLoCo workers; 1 = off")
    p.add_argument("--diloco-rank", type=int, default=0)
    p.add_argument("--diloco-every", type=float, default=1e9, help="global tokens between syncs (all workers); checkpoints are saved right before each sync and --ckpt-every is not used")
    p.add_argument("--diloco-dir", default=None, help="outer state (global weights and momentum, 168 GB at 21B) and exchange files, local disk")
    p.add_argument("--diloco-peers", default=None, help="the other workers as user@host,... (files pushed over ssh, received under numactl --membind=0); none: --diloco-dir is shared")
    p.add_argument("--diloco-key", default=None, help="ssh key for --diloco-peers")
    p.add_argument("--diloco-lr", type=float, default=0.7, help="outer Nesterov SGD (Douillard et al. 2023)")
    p.add_argument("--diloco-momentum", type=float, default=0.9)
    p.add_argument("--diloco-batch", default="split", choices=["split", "full"], help="split: each worker takes 1/workers of the scheduled batch, so the global batch per step is the schedule's; full: each worker takes the whole scheduled batch")
    p.add_argument("--val-data", nargs="+", default=None, help='held-out shards: one glob, or name="glob" per domain (per-domain CE and train-val gap, G2)')
    p.add_argument("--val-every", type=float, default=5e9, help="tokens")
    p.add_argument("--val-batches", type=int, default=16, help="micro-batches per rank")
    p.add_argument("--val-bytes-per-token", type=float, default=None, help="for bits-per-byte (G1, G4)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--moe-report-every", type=int, default=100, help="steps between [moe] lines: per layer max/mean load @ hottest expert (its bias); 0 = off")
    p.add_argument("--icl-every", type=int, default=50, help="steps between in-context-learning probes for the first 5,000 steps, x10 after (needs --val-data)")
    p.add_argument("--arith-probe", default=None, help="arithmetic probe file (scripts/make_arith_probe.py): [arith] lines, one GPU only (spec 11.5)")
    p.add_argument("--arith-every", type=int, default=50, help="steps between arithmetic probes")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, val, n_batches: int, precision: str = "bf16") -> float:
    model.eval()
    val.micro_step = 0
    total = torch.zeros((), device="cuda")
    for _ in range(n_batches):
        with precision_context(precision):
            total += model(**to_device(next(val)))["ce"]
    model.train()
    return all_reduce_mean(total / n_batches).item()


@torch.no_grad()
def icl_probe(model, batch, precision: str = "bf16") -> tuple[float, float] | None:
    model.eval()
    out = []
    for lo, hi in (ICL_EARLY, ICL_LATE):
        pos = batch["position_ids"]
        labels = batch["labels"].masked_fill((pos < lo) | (pos >= hi), IGNORE_INDEX)
        n = float((labels != IGNORE_INDEX).sum())
        with precision_context(precision):
            ce = model(**to_device({**batch, "labels": labels}))["ce"]
        total = all_reduce_sum(torch.stack([ce * n, ce.new_tensor(n)]))
        out.append((total[0] / total[1]).item() if total[1] > 0 else None)
    model.train()
    return None if None in out else (out[0], out[1])


def main():
    args = parse_args()
    rank, world, _ = init_distributed()
    log = print if rank == 0 else (lambda *a, **k: None)
    single_offload, fsdp_offload = args.offload and world == 1, args.offload and world > 1
    if fsdp_offload and world > args.shard_size:
        sys.exit("--offload on several GPUs runs on one node")
    te_fp8 = is_te_precision(args.precision)
    if args.precision == "fp8_rowwise" or args.experts == "fp8_grouped":
        from apex3.model import fp8

        if not fp8.available():
            sys.exit("--precision fp8_rowwise / --experts fp8_grouped need torchao and a Hopper GPU")
        fp8.FAST_ACCUM, fp8.RECIPE = args.fp8_fast_accum, args.fp8_recipe
    if args.act_offload and not (single_offload and args.recompute == "none"):
        sys.exit("--act-offload needs --offload on one GPU and --recompute none")
    if args.fp8_weights and not (te_fp8 and single_offload):
        sys.exit("--fp8-weights needs --offload on one GPU and a Transformer Engine FP8 --precision")
    if args.diloco_workers > 1 and not (single_offload and args.ckpt_dir and args.diloco_dir and 0 <= args.diloco_rank < args.diloco_workers):
        sys.exit("DiLoCo needs --offload on one GPU, --ckpt-dir, --diloco-dir and 0 <= --diloco-rank < workers")
    if te_fp8 and not single_offload and args.recompute != "full":
        log("[WARN] FP8 under FSDP2 without --recompute full keeps every layer's quantized weights until backward")
    cfg = get_config(args.config, attn=args.attn_ratio, **parse_overrides(args.set))
    experts = ("te" if single_offload else "te_stacked") if te_fp8 and args.te_experts else args.experts
    impl = ImplConfig(attn=args.attn, experts=experts, linear=linear_for(args.precision), recompute=args.recompute, moe_align=moe_align_for(args.precision),
                      moe_chunk=args.moe_chunk, precision=args.precision, compile_blocks=args.compile_blocks, ce_impl=args.ce_impl, ce_chunk=args.ce_chunk)
    if args.act_budget:
        if not (args.compile_blocks and args.recompute == "none"):
            sys.exit("--act-budget needs --compile-blocks and --recompute none")
        import torch._functorch.config as functorch_config

        functorch_config.activation_memory_budget = args.act_budget
    if args.gemm_offload != "none":
        if not (single_offload and args.compile_blocks and args.recompute == "none"):
            sys.exit("--gemm-offload needs --offload on one GPU, --compile-blocks and --recompute none")
        if args.gemm_offload == "hooks" and not args.act_offload:
            sys.exit("--gemm-offload hooks needs --act-offload GiB (its pinned CPU arena)")
        from apex3 import gemm_offload

        gemm_offload.install(cfg, args.gemm_offload, args.gemm_offload_part)
        gemm_offload.compiler_pool(args.gemm_offload).__enter__()
        log(f"gemm offload: {args.gemm_offload} ({args.gemm_offload_part}), FFN widths " f"{gemm_offload.ffn_widths(cfg, args.gemm_offload_part)}")
    if args.compile:
        L.enable_compile()
    torch.manual_seed(args.seed)

    if single_offload:
        model, init = build_single_gpu_model(cfg, impl, fp8_weights=args.fp8_weights)
        model.train()
        opt = OffloadOptimizer(model, lr=args.lr, weight_decay=args.wd, use_muon=args.optimizer == "muon", muon_momentum_dtype=DTYPES[args.muon_momentum], init=init, accumulate=args.offload_accum, gpu_share=args.offload_gpu_share)
        del init
    elif fsdp_offload:
        model = build_fsdp_offload_model(cfg, impl, world)
        model.train()
        opt = FSDPOffloadOptimizer(model, lr=args.lr, weight_decay=args.wd, use_muon=args.optimizer == "muon", muon_momentum_dtype=DTYPES[args.muon_momentum])
    else:
        with torch.device("meta"):
            model = Apex3(cfg, impl)
        apply_fsdp(model, build_mesh(world, args.shard_size), reduce_dtype=DTYPES[args.reduce_dtype])
        model.to_empty(device="cuda")
        model.init_weights()
        model.train()
        opt = OptimizerSet(model, lr=args.lr, weight_decay=args.wd, use_muon=args.optimizer == "muon", muon_momentum_dtype=DTYPES[args.muon_momentum])
    act = None
    if args.gemm_offload == "hooks":
        act = ActivationOffload(model, int(args.act_offload * 2**30), select=gemm_offload.selector(cfg, args.gemm_offload_part), ahead=1)
    elif args.act_offload:
        act = ActivationOffload(model, int(args.act_offload * 2**30))
    workers = args.diloco_workers
    data_rank, data_world = (args.diloco_rank, workers) if workers > 1 else (rank, world)
    data = build_batches(args.data, args.seq_len, args.micro_bs, args.eos_id, data_rank, data_world, args.seed, pool_strides(args.pool))
    vals, icl_batch = {}, None
    if args.val_data:
        vals = build_val(args.val_data, args.seq_len, args.micro_bs, args.eos_id, rank, world, 1)
        icl_loader = build_val(args.val_data[:1], args.seq_len, args.micro_bs, args.eos_id, rank, world, 2)
        icl_batch = next(next(iter(icl_loader.values())))
    arith = None
    if args.arith_probe:
        if world > 1:
            log("[arith] probe needs one GPU: off")
        else:
            arith = arith_probe.load(args.arith_probe, args.micro_bs, args.seq_len, "cuda")
            log(f"[arith] probe {args.arith_probe}: {arith['n']} problems, {len(arith['batches'])} micro-batch(es), " f"every {args.arith_every} steps")
    names = getattr(data, "names", ["all"])
    domain_index = {n: i for i, n in enumerate(names)}
    train_ce = torch.zeros(2, len(names), device="cuda")

    step, tokens = 0, 0
    resumed_periodic, resumed_diloco = None, None
    resume = args.resume or (latest_checkpoint(args.ckpt_dir) if args.ckpt_dir else None)
    if resume:
        state = (load_offload_checkpoint if args.offload else load_checkpoint)(resume, model, opt)
        for change in check_against_checkpoint(cfg, state.get("model_config")):
            log(f"config change from checkpoint: {change}")
        step, tokens = state["step"], state["tokens"]
        resumed_diloco = state.get("diloco")
        if resumed_diloco and (resumed_diloco["workers"], resumed_diloco["rank"]) != (workers, args.diloco_rank):
            sys.exit(f"{resume} is DiLoCo worker {resumed_diloco['rank']} of {resumed_diloco['workers']}, " f"this run is worker {args.diloco_rank} of {workers}")
        rows, unit = state.get("rows_seen", state["micro_step"] * args.micro_bs * world), args.micro_bs * data_world
        if rows % unit:
            if (resumed_diloco or {}).get("workers", 1) == workers:
                raise ValueError(f"{rows} rows consumed is not a multiple of micro_bs x data world = {unit}")
            log(f"{rows} rows consumed, rounded down to a multiple of {unit} for the DiLoCo data split")
            rows -= rows % unit
        data.resume(rows, state.get("data"))
        log(f"resumed from {resume}: step {step}, {tokens / 1e9:.2f}B tokens")
        same_dir = args.ckpt_dir and os.path.dirname(os.path.abspath(resume)) == os.path.abspath(args.ckpt_dir)
        if state.get("periodic") and same_dir:
            resumed_periodic = resume

    micro_tokens = args.micro_bs * args.seq_len * world
    batch_share = workers if args.diloco_batch == "split" else 1
    fpt = flops_per_token(cfg, args.seq_len)
    peak = peak_bf16_flops(torch.cuda.get_device_name())
    monitor = G2Monitor()
    next_ckpt = (math.floor(tokens / args.ckpt_every) + 1) * args.ckpt_every
    next_val = (math.floor(tokens / args.val_every) + 1) * args.val_every

    dl = None
    if workers > 1:
        exchange_dir = os.path.join(args.diloco_dir, "xchg")
        peers = [p for p in (args.diloco_peers or "").split(",") if p]
        exchange = SshExchange(exchange_dir, peers, args.diloco_key) if peers else LocalExchange(exchange_dir)
        dl = DiLoCo(opt._master_cpu, [m.expert_bias for m in model.modules() if isinstance(m, Router)], opt.refresh_params, rank=args.diloco_rank, workers=workers, state_dir=args.diloco_dir, exchange=exchange, lr=args.diloco_lr, momentum=args.diloco_momentum, log=log)
        if dl.start(resumed_diloco):
            dl.sync()
    next_sync = (math.floor(tokens / args.diloco_every) + 1) * args.diloco_every
    synced_at = tokens
    t_last, tok_last, step_last = time.perf_counter(), tokens, step
    moe_ivl: dict[str, float] = {}

    periodic = [resumed_periodic] if resumed_periodic else []

    def checkpoint_now(is_periodic=False, at_sync=False):
        path = os.path.join(args.ckpt_dir, f"step_{step:08d}")
        trainer_state = {"step": step, "tokens": tokens, "micro_step": data.micro_step, "rows_seen": data.rows_seen(), "seq_len": args.seq_len, "model_config": asdict(cfg), "data": data.state(), "precision": args.precision, "periodic": is_periodic}
        if dl is not None:
            trainer_state["diloco"] = {"completed": dl.completed, "at_sync": at_sync, "workers": workers, "rank": args.diloco_rank}
        (save_offload_checkpoint if args.offload else save_checkpoint)(path, model, opt, trainer_state)
        log(f"saved {path}")
        return path

    def save_periodic(at_sync=False):
        periodic.append(checkpoint_now(is_periodic=True, at_sync=at_sync))
        while args.keep_ckpts and len(periodic) > args.keep_ckpts:
            old = periodic.pop(0)
            if rank == 0:
                shutil.rmtree(old, ignore_errors=True)
                log(f"removed {old}")

    def stop_requested():
        if args.ckpt_dir and os.path.exists(os.path.join(args.ckpt_dir, "STOP")):
            log(f"{args.ckpt_dir}/STOP found: stopping at step {step}, {tokens / 1e9:.3f}B tokens")
            torch.distributed.destroy_process_group()
            sys.exit(0)

    stop_at = min(args.total_tokens, args.stop_at_tokens or args.total_tokens)
    while tokens < stop_at:
        target = batch_tokens(tokens, args.batch_start, args.batch_end, args.batch_ramp_frac * args.total_tokens)
        accum = max(1, round(target / batch_share / micro_tokens))
        parts = torch.zeros(4, device="cuda")
        for m in range(accum):
            if not fsdp_offload:
                set_grad_sync(model, m == accum - 1)
            batch = next(data)
            domains = set(data.last_domains)
            long_frac = (batch["position_ids"] >= 8192).float().mean()
            with act.forward() if act else contextlib.nullcontext(), precision_context(args.precision):
                out = model(**to_device(batch))
            (out["loss"] / accum).backward()
            if fsdp_offload:
                opt.after_backward()
            parts += torch.stack([out["ce"], out["z"], out["aux"], long_frac.to(parts.device)]) / accum
            if len(domains) == 1:
                d = domain_index[domains.pop()]
                train_ce[0, d] += out["ce"].detach()
                train_ce[1, d] += 1
        if args.offload:
            grad_norm = opt.clip_grad_norm(args.clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        tokens += accum * micro_tokens * workers
        lr = wsd_lr(tokens, args.lr, args.warmup_tokens, args.total_tokens, args.decay_frac)
        opt.set_lr(lr)
        opt.step()
        opt.zero_grad()
        gamma = router_bias_rate(tokens, args.total_tokens, cfg.router_bias_rate, args.bias_freeze_frac, cfg.router_bias_rate_late, cfg.router_bias_late_tokens)
        moe_stats, counts = update_router_biases(model, gamma, cfg.router_bias_update)
        step += 1
        for k in ("moe/load_max_over_mean_avg", "moe/hot_switch"):
            if k in moe_stats:
                moe_ivl[k] = moe_ivl.get(k, 0.0) + moe_stats[k]

        ce, z, aux, long_frac = all_reduce_mean(parts).tolist()
        gn = grad_norm.item()
        for alarm in monitor.update(step, ce, gn, moe_stats, counts):
            log(f"[G2 ALARM] step {step}: {alarm}")
        if counts is not None and args.moe_report_every and step % args.moe_report_every == 0:
            log(f"[moe] step {step} gamma {gamma:.3g} {load_report(model, counts)}")
        if step % args.log_every == 0:
            now = time.perf_counter()
            n_ivl = step - step_last
            tok_s_gpu = (tokens - tok_last) / (now - t_last) / world / workers
            t_last, tok_last, step_last = now, tokens, step
            mem = torch.cuda.max_memory_allocated() / 2**30
            moe_stats = {**moe_stats, **{k.replace("moe/", "moe/ivl_"): v / max(1, n_ivl) for k, v in moe_ivl.items()}}
            moe_ivl.clear()
            extra = " ".join(f"{k}={v:.3g}" for k, v in moe_stats.items())
            mfu_s = f"{mfu(tok_s_gpu, fpt, peak):.3f}" if peak else "n/a"
            epochs = data.epochs()
            ep = " ".join(f"{k}={v:.2f}" for k, v in epochs.items())
            long_s = f" long_frac {long_frac:.3f}" if args.seq_len > 8192 else ""
            log(f"step {step} tok {tokens / 1e9:.3f}B ce {ce:.4f} z {z:.3g} aux {aux:.3g} gnorm {gn:.3g} " f"lr {lr:.3g} bs {accum * micro_tokens / 1e6:.2f}M tok/s/gpu {tok_s_gpu:.0f} mfu {mfu_s} " f"mem {mem:.1f}GiB{long_s} epochs {ep} {extra}")
            over = [k for k, v in epochs.items() if v > (5 if k.startswith("code") else 4)]
            if over:
                log(f"[WARN] step {step}: more epochs than the limit (4, code 5; spec 0.3) in {', '.join(over)}")
        if vals and tokens >= next_val:
            train = all_reduce_sum(train_ce).tolist()
            train_ce.zero_()
            fields, gaps = [], {}
            for name, loader in vals.items():
                v = evaluate(model, loader, args.val_batches, args.precision)
                s = f"ce {v:.4f}" if name == "val" else f"{name} {v:.4f}"
                if len(vals) == 1 and args.val_bytes_per_token:
                    s += f" bpb {v / math.log(2) / args.val_bytes_per_token:.4f}"
                d = domain_index.get(name)
                if d is not None and train[1][d] > 0:
                    t = train[0][d] / train[1][d]
                    gaps[name] = v - t
                    s += f" (train {t:.4f}, gap {v - t:+.4f})"
                fields.append(s)
            log(f"[val] step {step} tok {tokens / 1e9:.3f}B " + " ".join(fields))
            for alarm in monitor.update_gap(gaps, data.epochs()):
                log(f"[G2 ALARM] step {step}: {alarm}")
            next_val += args.val_every
        if arith is not None and step % args.arith_every == 0:
            try:
                log(f"[arith] step {step} tok {tokens / 1e9:.3f}B " + arith_probe.format_stats(arith_probe.run(model, arith, args.precision)))
            except Exception as e:
                log(f"[arith] probe failed, off for this run: {type(e).__name__}: {e}")
                arith = None
                model.train()
                torch.cuda.empty_cache()
        if icl_batch is not None and step % (args.icl_every if step <= 5000 else 10 * args.icl_every) == 0:
            probe = icl_probe(model, icl_batch, args.precision)
            if probe:
                log(f"[icl] step {step} tok {tokens / 1e9:.3f}B early_ce {probe[0]:.4f} late_ce {probe[1]:.4f} " f"icl {probe[1] - probe[0]:+.4f}")
        if dl is not None and tokens >= next_sync:
            save_periodic(at_sync=True)
            dl.sync()
            synced_at, next_sync = tokens, next_sync + args.diloco_every
            while next_sync <= tokens:
                next_sync += args.diloco_every
            stop_requested()
        elif dl is None and args.ckpt_dir and tokens >= next_ckpt:
            save_periodic()
            next_ckpt += args.ckpt_every
            stop_requested()

    if dl is not None and synced_at != tokens:
        save_periodic(at_sync=True)
        dl.sync()
    if args.ckpt_dir:
        checkpoint_now()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
