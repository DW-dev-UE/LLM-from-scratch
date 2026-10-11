#!/usr/bin/env bash
set -uo pipefail
IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
say() { echo "$(date -u) $*"; }
case "${1:-}" in
1)
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq numactl > /dev/null && say "numactl $(numactl --version 2>&1 | head -1)"
  sudo mkdir -p /data /ckpt/wheels /ckpt/main /ckpt/logs && sudo chown ubuntu:ubuntu /data /ckpt /ckpt/wheels
  mkdir -p ~/apex3
  sudo docker pull -q $IMAGE && say "image pulled"
  sudo docker run -d --name apex3 --gpus all --ipc=host --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 -v ~/apex3:/workspace/apex3 -v /data:/data -v /ckpt:/ckpt $IMAGE sleep infinity && say "container apex3 up"
  sudo docker exec apex3 bash -c "nvidia-smi --query-gpu=name,power.limit --format=csv,noheader; numactl -H | head -3" 2>&1 | sed "s/^/  /"
  say "B_SETUP_1_DONE"
  ;;
2)
  ls ~/apex3/pyproject.toml /ckpt/wheels/flash_attn_3-*.whl > /dev/null || { say "FAILED: code or wheel missing"; exit 1; }
  sudo docker exec apex3 bash -c "apt-get install -y -qq numactl > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq numactl > /dev/null); pip install -q --no-deps -e /workspace/apex3 && pip install -q /ckpt/wheels/flash_attn_3-*.whl pytest py-spy" && say "pip installs done"
  sudo docker exec apex3 bash -c "cd / && python -W ignore -c 'import apex3, flash_attn_interface, torchao, torch; print(apex3.__file__, torchao.__version__, torch.__version__, torch.cuda.get_device_name())'; which numactl" | sed "s/^/  /"
  say "B_SETUP_2_DONE"
  ;;
*) echo "usage: $0 1 | 2"; exit 2 ;;
esac
