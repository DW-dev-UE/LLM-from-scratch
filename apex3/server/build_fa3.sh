#!/usr/bin/env bash
set -e
cd /tmp && rm -rf flash-attention && git clone -q --depth 1 https://github.com/Dao-AILab/flash-attention.git && cd flash-attention/hopper
export FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
    FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=24
pip wheel --no-build-isolation --no-deps -w /ckpt/wheels . 2>&1 | tail -5
pip install --no-deps --root-user-action ignore /ckpt/wheels/flash_attn_3*.whl
python -c "import flash_attn_interface; print(\"FA3 OK\")"
