from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from apex3.act_offload import ActivationOffload
from apex3.config import ImplConfig, get_config, parse_overrides
from apex3.data import synthetic_batch
from apex3.flops import flops_per_token, gpu_hours, mfu, param_counts, peak_bf16_flops
from apex3.model import Apex3
from apex3.model import layers as L
from apex3.model.moe import Router, update_router_biases
from apex3.fsdp_offload import FSDPOffloadOptimizer, build_fsdp_offload_model
from apex3.offload import OffloadOptimizer, build_single_gpu_model
from apex3.optim import OptimizerSet
from apex3.parallel import all_reduce_max, apply_fsdp, build_mesh, init_distributed
from apex3.precision import PRECISIONS, is_te_precision, linear_for, moe_align_for, precision_context


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="C")
    p.add_argument("--attn-ratio", default=None, help="5:1 | 4:1 | 2:1 | full")
    p.add_argument("--n-layers", type=int, default=None, help="truncate the model for single-GPU compute benchmarks")
    p.add_argument("--set", nargs="*", default=[], help="ModelConfig overrides, e.g. rope_scale_global=8")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--doc-len", type=int, default=None, help="split rows into documents (default: one per row)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--precision", default="bf16", choices=PRECISIONS)
    p.add_argument("--attn", default="fa2", choices=["fa2", "fa3", "sdpa"])
    p.add_argument("--experts", default="grouped_mm", choices=["grouped_mm", "fp8_grouped", "loop", "te", "te_stacked"], help="fp8_grouped: routed experts in FP8 with rowwise scales (apex3/model/fp8.py)")
    p.add_argument("--fp8-recipe", default="rowwise", choices=["rowwise", "tensorwise", "rowwise_with_gw_hp"], help="--precision fp8_rowwise: torchao scaling recipe for the linears (apex3/model/fp8.py)")
    p.add_argument("--fp8-fast-accum", action="store_true", help="--precision fp8_rowwise: Hopper fast accumulation on the forward GEMMs")
    p.add_argument("--linear", default="torch", choices=["torch", "te"])
    p.add_argument("--recompute", default="none", choices=["none", "full"])
    p.add_argument("--moe-chunk", type=int, default=32768, help="tokens per MoE chunk for long micro-batches, 0 = off")
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw", "none"])
    p.add_argument("--muon-momentum", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--check-muon", action="store_true", help="compare the distributed Newton-Schulz with the gathered reference (G-sys numerical match)")
    p.add_argument("--reduce-dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--a2a", action="store_true", help="also time an all-to-all of the EP dispatch volume inside the node (spec 7.1: EP estimate)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--ce-impl", default="torch", choices=["torch", "triton"], help="LM head + cross-entropy: torch reference or the Triton kernel (model/ce_triton.py)")
    p.add_argument("--ce-chunk", type=int, default=1024, help="tokens per LM head chunk")
    p.add_argument("--act-budget", type=float, default=0.0,
                   help="with --compile-blocks and --recompute none: torch.compile picks what each block recomputes so its saved activations stay within this fraction of saving everything (torch._functorch.config.activation_memory_budget); cheap elementwise ops go first")
    p.add_argument("--offload-accum", default="gpu", choices=["gpu", "cpu"], help="--offload on one GPU: add gradients into the CPU accumulator with GPU kernels over the link, or copy them with the copy engine and add on the CPU (RUNLOG_GH200.md)")
    p.add_argument("--offload-gpu-share", type=float, default=0.0, help="with --offload-accum cpu: fraction of the parameter elements (first in model order) still added on the GPU (slower on GH200, keep 0 there: RUNLOG_GH200.md)")
    p.add_argument("--compile-blocks", action="store_true", help="torch.compile every block's forward (fuses the elementwise and routing work around the GEMMs)")
    p.add_argument("--act-offload", type=float, default=0.0, help="GiB of pinned CPU memory for activations: save them to CPU memory in forward and fetch them back in backward instead of recomputing (--offload on one GPU, --recompute none; GH200)")
    p.add_argument("--gemm-offload-part", default="all", choices=["all", "experts", "shared"], help="--gemm-offload: which gate/up outputs (experts: the routed ones only, half the bytes on C64; shared: the shared expert and dense FFN)")
    p.add_argument("--gemm-offload", default="none", choices=["none", "hooks", "compiler"], help="save the FFN gate/up GEMM outputs instead of recomputing them, in CPU memory until backward (apex3/gemm_offload.py; hooks: --act-offload GiB is the arena)")
    p.add_argument("--offload", action="store_true", help="optimizer state in CPU memory: one GPU (spec v1.10, 7.4) or FSDP2 on one node (7.6)")
    p.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step; with --offload on several GPUs use 8 or more, so the CPU gradient accumulation shows up in the micro-batch time")
    p.add_argument("--fp8-weights", action="store_true", help="--offload with FP8: TE weights stored in FP8 (fp8_model_init), FP32 masters in CPU memory")
    p.add_argument("--batch-tokens", type=float, default=8e6, help="--offload: training batch over which the optimizer step is spread in tokens_per_s_at_batch")
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--peak-tflops", type=float, default=None, help="BF16 dense peak if the GPU is not in the table")
    p.add_argument("--json", default=None, help="append the result as one JSON line")
    p.add_argument("--profile", type=int, default=0, help="profile the first measured iteration and print this many kernels by CUDA time")
    p.add_argument("--trace", default=None, help="write the GPU kernels of the first measured iteration's micro-batches (no optimizer step) as a Chrome trace for scripts/trace_breakdown.py; with --steps > 1 that iteration is not timed")
    return p.parse_args()


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def time_all_to_all(n_elements: int, group, iters: int = 20) -> float:
    world = dist.get_world_size(group)
    n = max(world, n_elements - n_elements % world)
    send = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    recv = torch.empty_like(send)
    for _ in range(3):
        dist.all_to_all_single(recv, send, group=group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_to_all_single(recv, send, group=group)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    args = parse_args()
    rank, world, _ = init_distributed()
    if args.precision != "bf16":
        args.linear = linear_for(args.precision)
    if args.precision == "fp8_rowwise" or args.experts == "fp8_grouped":
        from apex3.model import fp8

        if not fp8.available():
            sys.exit("--precision fp8_rowwise / --experts fp8_grouped need torchao and a Hopper GPU")
        fp8.FAST_ACCUM, fp8.RECIPE = args.fp8_fast_accum, args.fp8_recipe
    if world > 1 and args.experts == "te":
        args.experts = "te_stacked"
    if args.act_offload and not (args.offload and world == 1 and args.recompute == "none"):
        sys.exit("--act-offload needs --offload on one GPU and --recompute none")
    if args.fp8_weights and not (args.offload and world == 1 and is_te_precision(args.precision)):
        sys.exit("--fp8-weights needs --offload on one GPU and a Transformer Engine FP8 --precision")
    if args.trace and args.profile:
        sys.exit("--trace and --profile are separate runs")

    overrides = parse_overrides(args.set)
    if args.n_layers:
        overrides["n_layers"] = args.n_layers
    cfg = get_config(args.config, attn=args.attn_ratio, **overrides)
    impl = ImplConfig(attn=args.attn, experts=args.experts, linear=args.linear, recompute=args.recompute, moe_align=moe_align_for(args.precision),
                      moe_chunk=args.moe_chunk, precision=args.precision, compile_blocks=args.compile_blocks, ce_impl=args.ce_impl, ce_chunk=args.ce_chunk)
    if args.act_budget:
        if not (args.compile_blocks and args.recompute == "none"):
            sys.exit("--act-budget needs --compile-blocks and --recompute none")
        import torch._functorch.config as functorch_config

        functorch_config.activation_memory_budget = args.act_budget
    if args.gemm_offload != "none":
        if not (args.offload and world == 1 and args.compile_blocks and args.recompute == "none"):
            sys.exit("--gemm-offload needs --offload on one GPU, --compile-blocks and --recompute none")
        if args.gemm_offload == "hooks" and not args.act_offload:
            sys.exit("--gemm-offload hooks needs --act-offload GiB (its pinned CPU arena)")
        from apex3 import gemm_offload

        gemm_offload.install(cfg, args.gemm_offload, args.gemm_offload_part)
        gemm_offload.compiler_pool(args.gemm_offload).__enter__()
    if args.compile:
        L.enable_compile()
    torch.manual_seed(0)

    mesh, init = None, None
    if args.offload and world > 1:
        model = build_fsdp_offload_model(cfg, impl, world)
    elif args.offload:
        model, init = build_single_gpu_model(cfg, impl, fp8_weights=args.fp8_weights)
    elif world > 1:
        with torch.device("meta"):
            model = Apex3(cfg, impl)
        mesh = build_mesh(world, args.shard_size)
        apply_fsdp(model, mesh, reduce_dtype=DTYPES[args.reduce_dtype])
        model.to_empty(device="cuda")
        model.init_weights()
    else:
        with torch.device("cuda"):
            model = Apex3(cfg, impl)
        model.init_weights()
        model.to(torch.bfloat16)
        for m in model.modules():
            if isinstance(m, Router):
                m.float()
    model.train()
    if args.offload and world > 1:
        opt = None if args.optimizer == "none" else FSDPOffloadOptimizer(model, lr=1e-5, use_muon=args.optimizer == "muon", muon_momentum_dtype=DTYPES[args.muon_momentum])
    elif args.offload:
        if args.optimizer == "none" and init is not None:
            sys.exit("--fp8-weights needs an optimizer (the weights are set from its masters)")
        opt = None if args.optimizer == "none" else OffloadOptimizer(model, lr=1e-5, use_muon=args.optimizer == "muon", muon_momentum_dtype=DTYPES[args.muon_momentum], init=init, accumulate=args.offload_accum, gpu_share=args.offload_gpu_share)
        del init
    else:
        opt = None if args.optimizer == "none" else OptimizerSet(model, lr=1e-5, use_muon=args.optimizer == "muon", muon_momentum_dtype=DTYPES[args.muon_momentum], muon_check=args.check_muon)

    peak = args.peak_tflops * 1e12 if args.peak_tflops else peak_bf16_flops(torch.cuda.get_device_name())
    fpt = flops_per_token(cfg, args.seq_len)
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    torch.cuda.reset_peak_memory_stats()
    times, opt_times = [], []
    fsdp_offload = args.offload and world > 1 and opt is not None
    act = None
    if args.gemm_offload == "hooks":
        act = ActivationOffload(model, int(args.act_offload * 2**30), select=gemm_offload.selector(cfg, args.gemm_offload_part), ahead=1)
    elif args.act_offload:
        act = ActivationOffload(model, int(args.act_offload * 2**30))
    prof = None
    for i in range(args.warmup + args.steps):
        batches = [synthetic_batch(args.micro_bs, args.seq_len, cfg.vocab_size, args.doc_len, generator=gen) for _ in range(args.accum)]
        if args.profile and i == args.warmup:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        tracing = bool(args.trace) and i == args.warmup
        if tracing:
            tprof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            tprof.__enter__()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch in batches:
            with act.forward() if act else contextlib.nullcontext(), precision_context(args.precision):
                out = model(**batch)
            (out["loss"] / args.accum).backward()
            if fsdp_offload:
                opt.after_backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if tracing:
            tprof.__exit__(None, None, None)
            if rank == 0:
                tprof.export_chrome_trace(args.trace)
            del tprof
        if opt is not None:
            if args.offload:
                opt.clip_grad_norm(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        else:
            model.zero_grad(set_to_none=True)
        update_router_biases(model, cfg.router_bias_rate)
        torch.cuda.synchronize()
        if i >= args.warmup and not (tracing and args.steps > 1):
            times.append(time.perf_counter() - t0)
            opt_times.append(time.perf_counter() - t1)
        if prof is not None:
            prof.__exit__(None, None, None)
            if rank == 0:
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.profile))
            prof = None

    step = sorted(times)[len(times) // 2]
    opt_step = sorted(opt_times)[len(opt_times) // 2]
    tok_per_gpu = args.accum * args.micro_bs * args.seq_len / step
    mem = all_reduce_max(torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], dtype=torch.float64, device="cuda"))
    result = {"config": args.config, "n_layers": cfg.n_layers, "global_every": cfg.global_every, "seq_len": args.seq_len, "micro_bs": args.micro_bs, "accum": args.accum, "world": world, "precision": args.precision, "attn": args.attn, "experts": args.experts, "linear": args.linear, "recompute": args.recompute, "optimizer": args.optimizer, "compile": args.compile, "device": torch.cuda.get_device_name(), "fp8_weights": args.fp8_weights, "fp8_fast_accum": args.fp8_fast_accum, "fp8_recipe": args.fp8_recipe, "act_offload_gib": args.act_offload,
              "ce_impl": args.ce_impl, "ce_chunk": args.ce_chunk, "gemm_offload": args.gemm_offload, "gemm_offload_part": args.gemm_offload_part, "act_budget": args.act_budget, "active_params": param_counts(cfg)["active"], "gflop_per_token": fpt / 1e9, "step_s_median": step, "optimizer_s_median": opt_step, "tokens_per_s_per_gpu": tok_per_gpu, "tflops_per_gpu": tok_per_gpu * fpt / 1e12, "mfu_vs_bf16_peak": mfu(tok_per_gpu, fpt, peak) if peak else None, "max_mem_alloc_gib": mem[0].item() / 2**30, "max_mem_reserved_gib": mem[1].item() / 2**30}
    if act is not None:
        result.update({k.replace("_bytes", "_gib"): v / 2**30 if k.endswith("_bytes") else v for k, v in act.stats.items()})
    if args.offload:
        micro, micro_tokens = (step - opt_step) / args.accum, args.micro_bs * args.seq_len
        tok_at_batch = micro_tokens / (micro + opt_step * micro_tokens * world / args.batch_tokens)
        result.update({"offload": True, "offload_accum": args.offload_accum, "offload_gpu_share": args.offload_gpu_share, "micro_s_median": micro, "tokens_per_s_at_batch": tok_at_batch, "mfu_at_batch": mfu(tok_at_batch, fpt, peak) if peak else None, "days_1.1T": gpu_hours(1.1e12, tok_at_batch) / 24 / world, "days_0.6T": gpu_hours(0.6e12, tok_at_batch) / 24 / world})
    if args.n_layers is None:
        result["gpu_hours_per_1T_tokens"] = gpu_hours(1e12, tok_per_gpu)
        result["days_2T_on_64_gpus"] = gpu_hours(2e12, tok_per_gpu) / 64 / 24
    if opt is not None and opt.muon is not None and opt.muon.check_error is not None:
        result["muon_check_max_abs_err"] = opt.muon.check_error
    if args.a2a and mesh is not None and cfg.n_experts:
        group = mesh.get_group("shard")
        t = time_all_to_all(args.micro_bs * args.seq_len * cfg.top_k * cfg.d_model, group)
        result["ep_a2a_ms_per_call"] = t * 1e3
        result["ep_a2a_s_per_step"] = 4 * cfg.n_moe_layers * t
        result["ep_a2a_share_of_step"] = 4 * cfg.n_moe_layers * t / step
    if rank == 0:
        for k, v in result.items():
            print(f"{k:>24}: {v:.4g}" if isinstance(v, float) else f"{k:>24}: {v}")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(result) + "\n")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
