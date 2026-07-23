[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-0969DA?style=flat-square)](BENCHMARK-v2.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](BENCHMARK-v2.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](BENCHMARK-v2.ja.md)

[← README](README.md)

---

# 📊 BENCHMARK v2 · APEX-1 (xl, ~1.12B)

> v1(327M `base`)과는 다른 계열입니다. **영어 전용** 토크나이저(32K)·코퍼스로 처음부터 다시 학습한 **1B급 라인**입니다.
>
> 체크포인트·로그의 `xl`은 제품명이 아니라 `model.py`의 preset 키(nano/small/base/**xl**)입니다. `train.py`가 `{mode}_{preset}_{tag}.pt` 규칙으로 파일명을 자동 생성하기 때문에 `sft_xl_v1.pt`처럼 남습니다 — **이 preset(`xl`)으로 학습한 결과물의 제품명이 APEX-1**입니다.

| | |
|:--|:--|
| 🆕 **최신** | `sft_xl_v1` · 2026-07-22 |
| 📦 **모델** | APEX-1 · Decoder-only · 실측 **1,119.5M** (`xl`) |
| 🧪 **세트** | 영어 전용 **15문항 × THINKING on/off** (327M 라인과 별도 세트, 코딩 5문항만 동일 문항 유지) |
| 📁 **원본** | [`ckpt/benchmark_sft_xl_v1_raw.json`](ckpt/benchmark_sft_xl_v1_raw.json) |

> [!WARNING]
> 아직 **체크포인트 1개뿐인 첫 스냅샷**입니다. v1 라인 같은 버전 사다리는 없고, 벤치 목적은 "정렬 다음 단계(추가 SFT vs RL)를 감이 아니라 측정으로 정하는 것"이었습니다.

### 목차

1. [한눈에 보기](#1-한눈에-보기)
2. [학습 과정](#2-학습-과정)
3. [데이터셋](#3-데이터셋)
4. [채점 방법](#4-채점-방법)
5. [sft_xl_v1 결과](#5-sft_xl_v1-결과)
6. [다음 단계 — RLVR](#6-다음-단계--rlvr)
7. [원본 파일](#7-원본-파일)

---

## 1. 한눈에 보기

| | 💬 일반 채팅 (no-thinking) | 🧠 THINKING 켬 |
|:--|:--:|:--:|
| QA 정답(키워드) | 5/8 | 4/8 |
| 코딩 완전 통과 | **5/5** (테스트 25/25) | 3/5 (테스트 15/25) |
| 빈 답변 | 0건 | 0건 |
| instruction 준수 | 1/2 | 1/2 |
| 길이 초과 | 8/10 | 5/10 |
| 평균 반복률 | 0.032 (최대 0.109) | 0.042 (최대 0.129) |
| 평균 답변 길이 | 375자 | 527자 |

> [!IMPORTANT]
> 327M 라인의 최대 실패였던 "THINKING 핸드오프(답 칸 공란)"는 **이 라인에서는 처음부터 발생하지 않았습니다** — 빈 답 0건. 대신 새 병목은 **장황함·반복**입니다.
>
> THINKING을 켜면 오히려 코딩(5/5→3/5)·QA(5/8→4/8) 둘 다 떨어집니다 — 사고 과정이 답의 품질을 깎아먹는 역전 현상. `reverse_string`·`factorial` 두 문제는 사고 모드에서 코드가 중간에 잘려 문법 오류로 실패했는데, 답변 예산(`max_new` 256)을 사고가 먼저 소모한 결과로 보입니다.

---

## 2. 학습 과정

### 🧩 모델 카드

| 항목 | 값 |
|:-----|:---|
| 계열 | Decoder-only Transformer (LLaMA 스타일) |
| 파라미터 | 실측 **1,119.5M** |
| 깊이·폭 | 24 layer · d_model 2048 |
| Attention | GQA (Q16·KV4) + RoPE(θ=500,000) + **QK-Norm** |
| FFN | SwiGLU |
| 정규화 | RMSNorm (Pre-Norm) |
| 컨텍스트 | max 4096 (학습 seq 2048) |
| 기타 | weight tying · bias 없음 |

327M `base`(→ [ARCHITECTURE](ARCHITECTURE.md)) 대비 **QK-Norm**(고lr 안정화)과 **rope_theta 500K**(Llama-3 값, 추후 컨텍스트 확장 여지)가 새로 추가됐습니다.

### 🛤️ 학습 파이프라인 — 2단계 커리큘럼

계획 당시엔 "51K step 코사인 1회"였지만, 실제로는 **일반 믹스 41K + 수학·코드 상향 10K**로 나눠 실행했습니다 (GPU: H100 1장).

```text
① 코퍼스 다운로드 (v3-en, 80.80GB, 실패 0)
        ▼
② 토크나이저 32K (byte-level BPE, 영어 전용)
        ▼
③ 토큰화 — train 20.31B tok · val 0.13B tok (혼합 val)
        ▼
④a Pretrain stage-1 · 41,000 step · lr 3e-4 · batch8×seq2048×accum24
        ▼  pretrain_xl_v1 (stage1)
④b 커리큘럼 재구성 — fineweb/dclm 각 7GB 재샘플 + 코드 전체 + finemath4 + cosmopedia2 (12.85B tok)
        ▼
④c Pretrain stage-2 continue · 10,000 step · lr 6e-5 (동일 batch/accum)
        ▼  pretrain_xl_v1 ★ (최종 pretrain)
        ▼
⑤ SFT · smoltalk2 영어 서브셋 538K 예제 × 2epoch ÷ (batch8×accum16) ≈ 8,400 step · lr 3e-5
        ▼  sft_xl_v1 ★
```

| 단계 | steps | lr | 처리 토큰(추정) | init |
|:-----|------:|---:|---------------:|:-----|
| Pretrain stage-1 | 41,000 | 3e-4 | ~16.12B | — |
| Pretrain stage-2 | 10,000 | 6e-5 | ~3.93B | stage-1 |
| **pretrain 합계** | **51,000** | — | **~20.05B** | — |
| SFT | 8,400 | 3e-5 | 538K예제 × 2epoch | pretrain stage-2 |

두 단계 처리 토큰 합(~20.05B)이 애초 목표였던 "20B 토큰"과 거의 정확히 일치합니다.

> [!NOTE]
> stage-2는 "더 많이"가 아니라 **재가중**입니다 — fineweb_edu·dclm은 7GB씩만 다시 뽑고, 코드 전체·수학(finemath4)·합성 교과서(cosmopedia2)는 통째로 다시 섞어 비중을 끌어올렸습니다.

### 📉 Loss 추이

<details>
<summary><b>Pretrain · SFT loss 표 펼치기</b></summary>

<br/>

**Pretrain stage-1** (val, 대표 지점)

| step | val loss |
|----:|--------:|
| 500 | 3.59 |
| 10,000 | 2.30 |
| 20,000 | 1.86 |
| 30,000 | 2.18 |
| 40,500 | 2.16 |

**Pretrain stage-2** (val, 대표 지점 · init = stage-1 결과물)

| step | val loss |
|----:|--------:|
| 500 | 2.12 |
| 5,000 | 1.93 |
| 9,000 | 1.56 |
| 9,500 | 2.10 |

**SFT** (train, 대표 지점)

| step | train loss |
|----:|-----------:|
| 0 | 2.12 |
| 2,000 | 1.34 |
| 5,000 | 1.49 |
| 8,100 | 1.06 |

</details>

> [!WARNING]
> val loss가 지점마다 ±0.3~0.5 정도 튑니다. README 서두의 교훈 — "다중 배치 평균 없이는 등락을 실력 변화로 오독하기 쉽다" — 이 여기도 그대로 적용됩니다. 위 표는 **경향 참고용**이지 지점별 비교용이 아닙니다.

---

## 3. 데이터셋

### 🔤 토크나이저

byte-level BPE · vocab **32,000** · 영어 전용 (327M 라인의 64K 다국어 vocab과는 다른 별도 토크나이저)

### 📚 Pretrain 코퍼스 (v3-en) — 80.80GB

| 소스 | 용량 | 성격 |
|:-----|-----:|:-----|
| fineweb_edu | 21.00GB | 영어 웹 (교육적 필터링) |
| dclm | 21.00GB | 영어 웹 (DCLM baseline) |
| finemath4 | 10.00GB | 수학 |
| cosmopedia2 | 8.50GB | 합성 교과서 |
| code_python | 8.00GB | 코드 |
| finepdfs | 3.00GB | PDF 추출 텍스트 |
| code_js/java/c/cpp/sql/shell | 6.50GB (합) | 코드 (7개 언어 합산) |
| wikipedia_en | 1.30GB | 백과사전 |

327M 라인과 달리 **한국어·일본어 자연어 소스가 전혀 없습니다** — README 서두의 "작은 모델은 다국어보다 영어 단일 집중이 유리하다"는 교훈을 설계 단계부터 반영했습니다.

### 🗣️ SFT 믹스 — HuggingFaceTB/smoltalk2 (영어 서브셋)

| 분류 | 소스 | 목표 행수 |
|:-----|:-----|---------:|
| no-think | magpie_ultra | 150,000 |
| no-think | openhermes2.5 | 100,000 |
| no-think | tulu3_persona_if | 40,000 |
| no-think | systemchats_30k | 30,000 |
| no-think | smol_summarize | 30,000 |
| no-think | smol_rewrite | 30,000 |
| no-think | everyday_conversations | 10,000 |
| no-think | science (MoT) | 30,000 |
| think | openthoughts3 | 120,000 |
| think | systemchats (Qwen3-32B) | 30,000 |
| think | table_gpt (Qwen3-32B) | 20,000 |
| think | multiturn_if | 15,000 |
| think | s1k 1.1 | 1,000 |

목표 합 ~606K행 중 실제 **538K행**으로 학습(일부 소스 미달). think : no-think ≈ **1 : 2** — 327M `base` v6의 교훈("thinking 데이터가 부족하면 사고→답변 전달이 무너짐")을 설계 단계에서부터 반영한 비율입니다.

가중치는 전부 ×1 — 소스별 목표 행수 자체가 이미 믹스 비율입니다. 멀티턴은 첫 user/assistant 쌍만 사용, multilingual/aya 계열은 영어 전용 모델이라 제외했습니다.

---

## 4. 채점 방법

| 항목 | 내용 |
|:-----|:-----|
| 문항 | **영어 전용 15문항 × 2모드 = 30 생성** (fact 2 · math 3 · reasoning 2 · instruction 2 · open 1 · coding 5) |
| 코딩 5문항 | 327M 라인 14문항 세트와 **동일 문항** — 라인 간 비교용으로 고정 |
| 생성 | temp **0.0 greedy** · top_p 0.9 · max_new **256** · seed 0 |
| 채점 | QA는 키워드 포함 여부(자동) · 코딩은 유닛테스트 실행 통과 수 |
| 추가 계측 | instruction 준수, 답변 길이 초과 여부, 반복률(repetition rate) — 다음 후속학습(SFT vs RL)을 **측정으로** 정하기 위해 새로 추가 |
| 일시 | 2026-07-22 |

> [!WARNING]
> 키워드 매칭은 약한 대리 지표입니다. `en_math_rate`(사고 모드)는 최종 답이 "1 meter"로 틀렸는데도 사고 과정에 "120"이 등장해 `keyword_hit: true`로 잡혔고, `en_reason_order`(일반 모드)는 정답 "Joe"를 언급만 하고 결론은 틀린 "Ann"인데도 키워드 히트 처리됐습니다. §5 표에서는 이런 어긋남을 `*` 로 표시합니다.

---

## 5. sft_xl_v1 결과

### 5.1 QA · 일반 채팅 (THINKING 끔)

| 문항 | 판정 | 실제 답변 요지 |
|:-----|:----:|:-------------|
| 수도(프랑스) | ✅ | 정답, 다만 관광지 설명까지 확장(120자 상한 → 352자) |
| 광합성 기체 | ✅ | CO2·엽록소·포도당까지 정확 |
| 기차 속력×시간 | ❌ | "60÷60=1시간"으로 단위를 뒤바꿔 완전히 틀림 (정답 120km) |
| 펜 거스름돈 | ❌ | "$12−$20=−$4"로 부호·연산 모두 틀림 (정답 $8) |
| 사과 다단계 | ❌ | 문제를 오독해 "2개"로 답함 (정답 29개) |
| 삼단논법(장미) | ✅ | "No, it does not follow…" — 논리적으로 깔끔한 정답 |
| 키 순서(최단) | ✅* | 결론은 "Ann이 가장 작다"로 틀림(정답 Joe) — 추론 중 "Joe" 단어만 등장해 키워드는 히트 |
| 한 단어로 답하기 | ✅* | "A clear daytime sky is a beautiful **blue** color…" — 색은 맞아 키워드 히트, 한 단어 지시는 위반 |
| 3색 콤마 나열 | ⚠️ | 자동 채점은 지시 준수(✅)로 판정했지만, 실제로는 "그것들은 섞어서 만들 수 없는 기본색입니다" 등 금지된 설명을 덧붙임 — 채점기가 놓친 사례 |
| 한 문장 요약 | – | 원문을 사실상 그대로 재진술(요약이라기보다 축약 없는 복사) |

### 5.2 QA · THINKING 켬

| 문항 | 판정 | 실제 답변 요지 |
|:-----|:----:|:-------------|
| 수도(프랑스) | ✅ | 정답, 짧고 정확 |
| 광합성 기체 | ✅* | "photolysis"라는 틀린 용어를 섞어 설명(키워드 CO2는 등장해 히트) |
| 기차 속력×시간 | ✅* | 사고 중 "120"이 여러 번 등장하지만 최종 답은 "1 meter"로 오답 |
| 펜 거스름돈 | ❌ | "$32−$20=$12"로 틀림 (정답 $8) |
| 사과 다단계 | ❌ | "12−7=5"로 틀림 (정답 29) |
| 삼단논법(장미) | ✅* | "Yes… No, all roses do not fade at once"로 자기모순 답변, "no"만 히트 |
| 키 순서(최단) | ❌ | "Ann이 가장 작다"로 오답, "Joe" 언급도 없어 키워드도 미스 |
| 한 단어로 답하기 | ❌ | 지시는 지킴("clear" 한 단어)이지만 색이 틀림(정답 blue) |
| 3색 콤마 나열 | ❌ | "red blue yellow" 콤마 없이 나열 — 지시 위반 |
| 한 문장 요약 | – | 일반 모드와 동일하게 원문 재진술 |

일반 5/8 vs 사고 4/8 — 사고를 켜도 QA 정확도가 오르지 않고 오답 유형만 늘었습니다 (펜 거스름돈·사과 다단계는 양쪽 다 실패, 키 순서는 사고 모드가 더 나쁨).

### 5.3 코딩 · 두 모드 비교

일반 채팅은 **5/5 만점**. THINKING 모드는 2문제가 **코드 중간에 잘려** 실패했습니다.

| 문제 | 일반 채팅 | THINKING | 코멘트 |
|:-----|:--------:|:--------:|:-------|
| `is_prime` | 5/5 | 5/5 | 양쪽 모두 √n 시도분해로 통과 |
| `reverse_string` | **5/5** | **0/5** | 사고 모드: `''.join(reversed` 에서 **괄호가 닫히지 않고 코드가 잘림** (SyntaxError) |
| `factorial` | **5/5** | **0/5** | 사고 모드: `return` 까지만 쓰고 값 없이 잘림 |
| `is_palindrome` | 5/5 | 5/5 | 양쪽 모두 `s == s[::-1]` |
| `find_max` | 5/5 | 5/5 | 양쪽 모두 내장 `max()` 활용 |

<details>
<summary><b>THINKING 모드에서 잘린 코드 원문</b></summary>

<br/>

```python
# reverse_string — SyntaxError: '(' was never closed
def reverse_string(s):
    characters = s.split()
    reversed_chars = ''.join(reversed(characters))
    return ''.join(reversed

# factorial — return 문에 값이 없음
def factorial(n):
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    elif n == 0 or n == 1:
        return
```

</details>

> [!IMPORTANT]
> 두 실패 모두 "알고리즘을 몰라서"가 아니라 **`max_new=256` 예산 안에서 사고 서술이 길어져 코드 마무리 전에 잘린** 것으로 보입니다 — 327M 라인의 "핸드오프"(답 칸이 통째로 비는 버그)와는 다른 종류의 예산 문제입니다. 사고/답변 예산 분리 또는 확대가 다음 실험 후보입니다.

---

## 6. 다음 단계 — RLVR

벤치의 목적 자체가 "추가 SFT를 더 돌릴지, RL로 넘어갈지"를 감이 아니라 계측으로 정하는 것이었습니다.

| 근거 | 값 | 해석 |
|:-----|:--|:-----|
| 능력 | 일반 채팅 코딩 **5/5 완전 통과** | 알고리즘 자체는 충분히 학습됨 |
| 정렬 문제 | 반복률 평균 **0.032**, 길이 초과 **8/10** | 장황함·반복이 병목 — 능력이 아니라 스타일 문제 |

**→ 결정: RLVR** ([`ckpt/AUTO_DECISION_xl_v1.txt`](ckpt/AUTO_DECISION_xl_v1.txt))

> [!NOTE]
> 근거 원문: *"정렬(장황·반복): repetition 0.032, over_length 8/10 — 능력은 충분(coding 5/5)"*
>
> 능력 문제였다면 SFT 데이터·스텝을 늘리는 쪽이 맞지만, 지금 병목은 "무엇을 아는가"가 아니라 "어떻게 답하는가"이므로 다음 후속학습은 DPO·추가 SFT보다 **RLVR(검증 가능 보상 기반 RL)** 로 진행하기로 결정했습니다.

327M 라인이 겪은 "형식(핸드오프) → 정확도" 순의 병목 이동과 비교하면, 1B 라인은 핸드오프 문제 없이 시작해 곧장 **"정확도 + 장황함"** 단계에 있습니다.

---

## 7. 원본 파일

| 경로 | 내용 |
|:-----|:-----|
| [`ckpt/benchmark_sft_xl_v1_raw.json`](ckpt/benchmark_sft_xl_v1_raw.json) | sft_xl_v1 벤치 원본 (30개 생성 전체, thinking 텍스트 포함) |
| [`ckpt/AUTO_DECISION_xl_v1.txt`](ckpt/AUTO_DECISION_xl_v1.txt) | RLVR 결정 로그 |
| — | pretrain_xl_v1 단독(SFT 전) 채팅 벤치 **미측정** |

> 가중치(`.pt`)와 원본 코퍼스는 공개 저장소에 포함하지 않습니다.

---

<div align="center">

[README](README.md) · [BENCHMARK v1 (327M)](BENCHMARK-v1.md) · [ARCHITECTURE](ARCHITECTURE.md)

</div>
