# Apex-2 — 3.02B, 2× H100

ComposeLM(commit `003763d`, editable) 위의 dense decoder-only 사전학습.
저장소 루트의 `README.md`는 폐기된 7B / 128K vocab / `core/scripts/` 세대를
설명하므로 이 문서가 `apex2/` 패키지의 유일한 기준이다.

## 아키텍처

| 항목 | 설정 |
|---|---|
| Layers | 28 |
| Hidden size | 3,072 |
| Attention | GQA, Q 24 / KV 8 (3:1), head_dim 128 |
| FFN | SwiGLU, 8,192 (8/3 × d_model) |
| Norm | Pre-RMSNorm |
| Position | RoPE θ=10,000 → YaRN 4× (stage 2) |
| Vocabulary | 65,536 byte-level BPE, FIM 지원 |
| Embedding | tied, bias-free |

파라미터 (직접 계산):

| | |
|---|---|
| Attention / layer | 3072×3072×2 + 3072×1024×2 = 25,165,824 |
| FFN / layer | 3072×8192×3 = 75,497,472 |
| × 28 layers | 2,818,572,288 |
| Embedding (tied) | 65536×3072 = 201,326,592 |
| Norm (28×2+1)×3072 | 175,104 |
| **합계** | **3,020,073,984** |

### 3.02B를 고른 이유

1.0e22 FLOPs의 Chinchilla 최적점은 **8.2B / 165B 토큰**이다. 즉 3.02B는 이
예산의 loss 최적이 **아니며**, 원안 7.05B가 오히려 최적에 가까웠다.
3.02B는 450B에서 **149 tokens/param**(600B 연장 시 199)으로 의도적으로
over-train해 추론 비용을 약 2.7× 낮추는 선택이다. "예산에서 가능한 최대"가
아니라 "목표 품질을 만족하는 최소"로 기술해야 한다.

## 학습 설정

| 항목 | 값 |
|---|---|
| 하드웨어 | 2× H100 80GB SXM5 |
| 분산 | FSDP2 full shard |
| Precision | bf16_mixed (fp32 master) |
| Micro batch × accum | **2 × 128** |
| Tokens / step | 2,097,152 |
| LR | 3e-4, WSD (warmup 2K → stable → 1-sqrt decay, floor 0.1×) |
| max_steps | 286,000 (동결, 600B) |
| 학습 토큰 | 450B (214.5K step), `--decay-start`로 600B 연장 |

### 메모리 — micro batch가 4가 아니라 2인 이유

backward가 붙드는 텐서는 토큰·레이어당 약 51K bf16 elem
(qkv 5,120 + gate/up 16,384 + silu 8,192 + 잔차·norm 계열 ~21.5K) = **~100 KB**.

| micro batch | 토큰/micro | 활성값 | + 상태 24.2 GB | 여유 (80 GB) |
|---|---|---|---|---|
| 4 | 16,384 | 45.8 GB | 70.0 GB | all-gather 버퍼·단편화 전에 88% — **OOM** |
| **2** | **8,192** | **22.9 GB** | **47.1 GB** | 59% |

`loss_chunk_size=2048`이 로짓(토큰수 × 65,536)은 막지만 위 값은 막지 못한다.
tokens/step은 2,097,152로 동일하므로 LR·스텝 예산은 그대로 유효하다.

### 기간 — 106일이 아닌 이유

`6ND`는 attention을 뺀 값이다. 토큰당 `12 · n_layers · seq_len · d_model`
= 12 × 28 × 4096 × 3072 = **4.23 GFLOPs**를 더해야 하며, 이는 6N(18.1 GFLOPs)
대비 +23%다.

| | 450B | 600B |
|---|---|---|
| FLOPs | 1.005e22 | 1.34e22 |
| 100% MFU | 58.7일 | 78.3일 |
| **40–48% MFU (2-GPU 실측 범위)** | **122–147일** | **163–196일** |
| 비용 @ $8.37/h | $24,500–29,500 | $32,700–39,400 |

600B는 인스턴스 예약이 6개월을 커버할 때만 선택한다.

## 데이터 혼합

전 소스가 450B에서 1.0 epoch 미만, 600B에서 1.2 미만이다 (연장 시 같은 shard를
재사용하므로 두 값을 모두 확인할 것).

| 소스 | weight | 변경 | 이유 |
|---|---|---|---|
| fineweb-edu | 0.340 | — | ARC/MMLU 핵심 |
| dclm-baseline | **0.200** | 0.14 → | 상식 추론은 일반 웹 토큰 수에 비례 |
| stack-v3 | **0.200** | 0.26 → | 무필터 raw 코드는 토큰당 HumanEval 기여가 낮음 |
| cosmopedia-web-v1/v2 | 0.045 / 0.030 | — | |
| cosmopedia-math | **0.004** | 0.010 → | 2.25 epoch였음 (2B 코퍼스에 4.5B 목표) |
| cosmopedia-stanford | **0.002** | 0.005 → | 동일 |
| finemath-3plus | **0.059** | 0.050 → | 위에서 회수한 0.009 흡수 (34B 헤드룸) |
| infiwebmath-3plus | 0.030 | — | |
| pes2o / stackexchange / wikimedia | 0.040 / 0.030 / 0.020 | — | |

edu 34% + 코드 + 수학이 이미 특수 도메인 비중을 크게 만들어 놓은 상태라,
HellaSwag·PIQA를 목표대로 받으려면 일반 웹 비중이 필요하다.

## 문맥 확장 (stage 2)

base 런은 4,096 밖 위치를 한 번도 보지 않으므로 16K는 이 단계 전까지 존재하지
않는다. YaRN NTK-by-parts는 4× 점프에서 외삽보다 훨씬 덜 무너지지만, 목표 길이
파인튜닝 없이 쓰면 단문맥 성능까지 몇 점 잃는다.

| 항목 | 값 |
|---|---|
| 예산 | 8B 토큰 (사전학습의 1.8%), 7,629 step |
| micro × accum | 1 × 32, seq 16,384 → 1,048,576 tok/step |
| LR | 3e-5 (사전학습 peak의 0.1×), warmup 200 |
| Activation checkpointing | **on** (16,384 tok/micro는 4K의 bs=4와 같은 구간) |
| FLOPs | 토큰당 6N 18.1 + attn 16.9 GFLOPs, 재계산 +33% |
| 기간 | **5–6일** ($1,000–1,200) |

ComposeLM의 위치 버퍼는 전부 `persistent=False`
(`layers/pos_emb.py:184,380,609`)이므로 `rope → yarn` 전환에 파라미터 이름
변화가 없고 base 가중치가 `strict=True`로 로드된다.

## 목표 성능

레퍼런스는 **동급 파라미터 + 동급 토큰 예산**인 Pythia-2.8B(300B),
OpenLLaMA-3B(1T)다. Llama-3.2-3B(9T)·Qwen2.5-3B(18T)가 아니다.

| 벤치마크 | 목표 | 레퍼런스 |
|---|---|---|
| ARC-Easy (0) | 63–70 | Pythia 64.4 / OpenLLaMA 69.4 |
| ARC-Challenge (0) | 36–42 | Pythia 32.9 / OpenLLaMA 34.1 |
| HellaSwag (0) | 58–63 | 68은 ~1.5T 토큰 필요 |
| PIQA (0) | 73–77 | |
| Winogrande (0) | 58–63 | |
| MMLU (5) | 29–33 | 38은 9T급 영역 |
| HumanEval pass@1 | 15–22 | StarCoder2-3B는 3T 코드로 31.7 |
| MBPP pass@1 | 18–26 | |
| **0-shot 평균** | **53–57** | ship gate: > 49.62 (Apex-1) |

`eval.py`가 실행 시 각 항목을 이 범위와 대조해 `BELOW`를 표시한다.

## 검증

`tests/test_apex2_pipeline.py` — GPU·네트워크·실제 코퍼스 없이 39개 테스트.
(`tests/test_apex2_{data,train,diagnostics}.py`는 폐기된 128K 파이프라인용이다.)

```bash
python -m pytest tests/test_apex2_pipeline.py -q
```

고정하는 불변식: 파라미터 3,020,073,984 / tokens-per-step 2,097,152 /
mixture weight 합 1.0 및 전 소스 <2 epoch / bs=2의 메모리 여유 /
`rope→yarn` state_dict 키 동일 / 커서 재개 정확성 / rank 간 shard 분리 /
WSD 형태(1-sqrt < linear) / 목표치·오염감사·평가 태스크 목록 동기화 /
prune의 `latest` 보호 / shortfall 임계 2%.

### 홀드아웃 손실

`prepare_data.py`가 각 소스 스트림의 **앞** 200만 토큰을 `val.bin`으로 떼어낸다.
`PackedCorpus`는 `index["shards"]`만 읽으므로 분리는 검증 대상이 아니라 구조적이다.

```bash
python apex2/validate.py runs/apex2-3b/latest
```

학습 루프 안이 아니라 저장된 체크포인트를 상대로 돈다 — composelm은 콜백을
rank 0에서만 호출하므로(`train/trainer.py:1399`) 콜백 안의 forward는 FSDP2
all-gather에서 데드락한다. `--decay-start`를 언제 잡을지는 lm-eval이 아니라 이
가중 손실이 평평해지는 시점으로 판단한다. cron 30분 주기 권장.

### 오염 감사

```bash
python apex2/decontaminate.py --docs-per-source 200000   # DECONTAMINATION.json
python apex2/decontaminate.py --verify                   # 가중치 반영 후 재확인
```

GPT-3/Llama 방식의 13-gram(blake2b-64) 중복률. **필터링이 아니라 감사**다 —
450B 토큰을 Python 스캔으로 거르는 것은 실패하면 안 되는 잡에 붙이는 위험이고,
이 혼합의 기반 코퍼스들(FineWeb-Edu, DCLM, Pile) 역시 같은 이유로 보고만 한다.
`--threshold`(기본 0.1%) 초과 태스크는 모델 카드에 각주하거나 보고 대상에서
제외한다. 표본 감사이므로 20만 문서 기준 검출 하한은 5e-6, 0.0%는 "하한 미만"이지
"깨끗함"이 아니다.

## 실행 순서

```bash
python apex2/lock_datasets.py
python apex2/train_tokenizer.py --out tokenizer/apex2
python apex2/prepare_data.py --all --budget 450e9
python apex2/decontaminate.py
torchrun --standalone --nproc_per_node=2 apex2/train.py
torchrun --standalone --nproc_per_node=2 apex2/extend_context.py --init-from runs/apex2-3b/latest
python apex2/eval.py --ckpt runs/apex2-3b-16k/latest --tag final-16k --extended
```

배포 직전 `lock_datasets.py --verify`와 `decontaminate.py --verify`를 모두 통과시킨다.

cron:

```bash
*/30 * * * * cd /home/ubuntu/llm && python apex2/prune_ckpt.py runs/apex2-3b --keep 5 && python apex2/validate.py runs/apex2-3b/latest
```
