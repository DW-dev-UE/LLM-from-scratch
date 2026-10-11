#!/usr/bin/env bash
set -uo pipefail
PLAN=${PLAN:?plan file}
CODE=${CODE:-/workspace/apex3}
TAG=$(date -u +%m%d%H%M)
OUT=/ckpt/bench_plan_$TAG.jsonl
SUM=/ckpt/bench_plan_$TAG.txt
SET="n_experts=64 expert_ffn=1280 top_k=4 router_bias_rate=0.01 router_bias_update=proportional"
say() { echo "$(date -u) $*" | tee -a "$SUM"; }
inc() {
  sudo docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PYTHONPATH=$CODE apex3 bash -c "cd $CODE && $1"
}
sudo docker exec apex3 pgrep -f "scripts/train.py|bench_mfu.py" > /dev/null && { say "FAILED: GPU busy"; exit 1; }
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
say "GPU $(nvidia-smi --query-gpu=name,enforced.power.limit,memory.used --format=csv,noheader), plan $PLAN, code $CODE"
where=$(inc "python -W ignore -c 'import apex3; print(apex3.__file__)'" 2>&1 | tail -1)
say "imports: $where"
case "$where" in "$CODE"/*) ;; *) say "FAILED: apex3 does not import from $CODE"; exit 1 ;; esac
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  inc "python -m pytest -q -s -k 'not compiler' tests/test_fp8_rowwise.py tests/test_gpu_model.py tests/test_gemm_offload.py tests/test_offload.py tests/test_ce_triton.py" > "$SUM.pytest" 2>&1
  say "pytest exit $?: $(grep -E "passed|failed|error" "$SUM.pytest" | tail -1)"
fi
B="--config C --set $SET --offload --offload-accum cpu --compile-blocks --recompute none --act-budget 0.05 --moe-chunk 0 --micro-bs 6 --attn fa3 --accum 8 --ce-impl triton --ce-chunk 4096 --steps 4 --warmup 2"
bench() {
  local n=$1; shift
  say "bench $n: $*"
  inc "TORCH_LOGS=recompiles,graph_breaks numactl --membind=0 torchrun --nproc_per_node 1 scripts/bench_mfu.py $* --json $OUT" > "$SUM.bench_$n" 2>&1
  grep -E "micro_s_median|tokens_per_s_per_gpu|mfu_at_batch|max_mem_reserved|offloaded_gib|offloaded_tensors|overflow_gib|Error|OutOfMemory" "$SUM.bench_$n" | tail -8 | tee -a "$SUM"
}
while read -r num rest; do
  case "$num" in "" | \#*) continue ;; esac
  bench "$num" $B ${rest%%#*}
done < "$PLAN"
python3 - "$OUT" <<'EOF' | tee -a "$SUM"
import json, os, sys
rows = [json.loads(l) for l in open(sys.argv[1])] if os.path.exists(sys.argv[1]) else []
def tag(r):
    s = r["precision"] + (":" + r["fp8_recipe"] if r["precision"] != "bf16" and r.get("fp8_recipe", "rowwise") != "rowwise" else "")
    s += "+fp8_grouped" if r.get("experts") == "fp8_grouped" else ""
    s += f"+keep({r.get('gemm_offload_part', 'all')})" if r.get("gemm_offload", "none") != "none" else ""
    return s + f"@b{r.get('act_budget', 0.05):g}"
for r in rows:
    print(f"full {tag(r):40s} micro {r['micro_s_median']:.3f} s  mfu {r['mfu_at_batch']:.4f}  mem {r['max_mem_reserved_gib']:.1f} GiB")
EOF
say "BENCH_PLAN_DONE"
