#!/usr/bin/env bash
set -uo pipefail
if [ "${1:-}" != inner ]; then
  until grep -q FETCH_DONE ~/fetch.log 2>/dev/null; do sleep 300; done
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  grep -E "FilePages" /sys/devices/system/node/node1/meminfo
  echo "$(date -u) data ready, starting: set '${APEX_SET:-}' micro-bs ${APEX_MICRO_BS:-4} act-budget ${APEX_ACT_BUDGET:-0.1}"
  echo "$(date -u) main run extra: '${APEX_MAIN_EXTRA:-}' skip smoke: ${APEX_SKIP_SMOKE:-0}"
  exec sudo docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e APEX_SET="${APEX_SET:-}" -e APEX_MICRO_BS="${APEX_MICRO_BS:-4}" \
    -e APEX_ACT_BUDGET="${APEX_ACT_BUDGET:-0.1}" -e APEX_MAIN_EXTRA="${APEX_MAIN_EXTRA:-}" -e APEX_SKIP_SMOKE="${APEX_SKIP_SMOKE:-0}" apex3 bash /workspace/apex3/server/train_gh200.sh inner
fi

cd /workspace/apex3
B=/data/bin
V=/data/val
DATA=()
VAL=()
for dw in code:0.35 web:0.40 textbook:0.02 pdf:0.05 math:0.06 papers:0.06 medical:0.03 qa:0.02 books:0.01; do
  d=${dw%%:*}
  DATA+=("$d=${dw##*:}:$B/$d/*.bin")
  VAL+=("$d=$V/$d/*.bin")
done
SET=()
[ -n "${APEX_SET:-}" ] && SET=(--set ${APEX_SET})
COMMON=(--config C "${SET[@]}" --eos-id 203776 --total-tokens 1.1e12 --offload --offload-accum cpu --precision bf16 --compile-blocks --recompute none --act-budget "${APEX_ACT_BUDGET:-0.1}" --moe-chunk 0 --micro-bs "${APEX_MICRO_BS:-4}" --attn fa3 --data "${DATA[@]}" --val-data "${VAL[@]}")
RUN=(numactl --membind=0 torchrun --nproc_per_node 1 scripts/train.py)
mkdir -p /ckpt/logs /ckpt/main

if [ "${APEX_SKIP_SMOKE:-0}" != 1 ]; then
  SMOKE=(--ckpt-dir /ckpt/smoke --log-every 1 --val-every 4e6 --val-batches 2 --icl-every 1)
  rm -rf /ckpt/smoke
  "${RUN[@]}" "${COMMON[@]}" "${SMOKE[@]}" --stop-at-tokens 6e6 > /ckpt/logs/smoke1.log 2>&1 || { echo "$(date -u) SMOKE1_FAILED"; exit 2; }
  "${RUN[@]}" "${COMMON[@]}" "${SMOKE[@]}" --stop-at-tokens 1.2e7 > /ckpt/logs/smoke2.log 2>&1 || { echo "$(date -u) SMOKE2_FAILED"; exit 2; }
  grep -q "resumed from" /ckpt/logs/smoke2.log || { echo "$(date -u) SMOKE_NO_RESUME"; exit 2; }
  rm -rf /ckpt/smoke
  echo "$(date -u) smoke OK, main run"
fi

EXTRA=(${APEX_MAIN_EXTRA:-})
for attempt in $(seq 1 20); do
  "${RUN[@]}" "${COMMON[@]}" --ckpt-dir /ckpt/main --stop-at-tokens 0.51e12 --ckpt-every 1e9 --keep-ckpts 1 "${EXTRA[@]}" >> /ckpt/logs/main_phase1.log 2>&1
  rc=$?
  [ -e /ckpt/main/STOP ] && { echo "$(date -u) STOPPED (/ckpt/main/STOP), exit $rc"; exit 3; }
  [ $rc -eq 0 ] && { echo "$(date -u) PHASE1_DONE"; exit 0; }
  echo "$(date -u) main run exited (attempt $attempt), restarting from the latest checkpoint"
  sleep 60
done
echo "$(date -u) PHASE1_GAVE_UP"
exit 1
