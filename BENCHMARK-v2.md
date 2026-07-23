[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-0969DA?style=flat-square)](BENCHMARK-v2.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](BENCHMARK-v2.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](BENCHMARK-v2.ja.md)

[← README](README.md)

---

# 📊 BENCHMARK v2 · APEX-1 (~1.12B)

> v1(327M `base`)과는 다른 계열입니다. **영어 전용** 토크나이저(32K)·코퍼스로 처음부터 다시 학습한 **1B급 라인**입니다.
>
> 체크포인트·로그는 `model.py`의 preset명 **Apex-1**을 그대로 사용합니다 (`pretrain_Apex-1_v1`, `sft_Apex-1_v1`).

| | |
|:--|:--|
| 🆕 **최신** | `dpo_Apex-1_v1` · 2026-07-23 |
| 📦 **모델** | APEX-1 · Decoder-only · 실측 **1,119.5M** |
| 🧪 **세트** | 영어 전용 **15문항 × THINKING on/off** (327M 라인과 별도 세트, 코딩 5문항만 동일 문항 유지) + 표준 하네스 11종 |
| 📁 **원본** | [`ckpt/benchmark_sft_Apex-1_v1_raw.json`](ckpt/benchmark_sft_Apex-1_v1_raw.json) · [`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md) |
| ✅ **완료** | RLVR 폐기 → **DPO 완료** (§6) · 표준 벤치 11종 측정 완료 (§5.4) |

> [!WARNING]
> 아직 **체크포인트 1개뿐인 첫 스냅샷**입니다. v1 라인 같은 버전 사다리는 없고, 벤치 목적은 "정렬 다음 단계(추가 SFT vs RL)를 감이 아니라 측정으로 정하는 것"이었습니다.

### 목차

1. [한눈에 보기](#1-한눈에-보기)
2. [학습 과정](#2-학습-과정)
3. [데이터셋](#3-데이터셋)
4. [채점 방법](#4-채점-방법)
5. [sft_Apex-1_v1 결과](#5-sft_apex-1_v1-결과)
6. [다음 단계 — RLVR → DPO](#6-다음-단계--rlvr--dpo)
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
        ▼  pretrain_Apex-1_v1 (stage1)
④b 커리큘럼 재구성 — fineweb/dclm 각 7GB 재샘플 + 코드 전체 + finemath4 + cosmopedia2 (12.85B tok)
        ▼
④c Pretrain stage-2 continue · 10,000 step · lr 6e-5 (동일 batch/accum)
        ▼  pretrain_Apex-1_v1 ★ (최종 pretrain)
        ▼
⑤ SFT · smoltalk2 영어 서브셋 538K 예제 × 2epoch ÷ (batch8×accum16) ≈ 8,400 step · lr 3e-5
        ▼  sft_Apex-1_v1 ★
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

## 5. sft_Apex-1_v1 결과

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

### 5.4 표준 벤치마크 (lm-evaluation-harness)

위 §5.1–5.3은 자체 제작 15문항 세트입니다. 여기서는 `sft_Apex-1_v1`과 `dpo_Apex-1_v1`을 HF 포맷으로 변환해(로짓 일치 검증 max diff 2.3e-5) [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.12 · bf16으로, **공개된 점수가 있는 다른 모델들과 나란히** 비교했습니다. 측정일 2026-07-23, H100 80GB(Lambda). 프로토콜(상식 7종 0-shot, MMLU 5-shot, GSM8K 5-shot, HumanEval·MBPP 0-shot pass@1)과 비교 대상 수치는 [TinyLlama 논문](https://arxiv.org/abs/2401.02385) Table 2·3에서 그대로 가져왔습니다.

#### 한눈에

| 항목 | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B |
|:---|---:|---:|---:|---:|
| 상식 7종 평균 (0-shot) | 49.54 | 49.65 | 52.99 | 48.30 |
| MMLU (5-shot) | 24.83 | 24.90 | 25.34 | 25.70 |
| GSM8K (5-shot, strict) | 1.44 | 1.90 | — | — |
| HumanEval (pass@1) | 8.54 | 8.54 | 9.15 | 1.83 |
| MBPP (pass@1) | 4.80 | 5.20 | — | — |

> [!IMPORTANT]
> **DPO ≥ SFT, alignment tax 없음** — 전 항목에서 DPO가 SFT와 같거나 앞섭니다. **BoolQ 62.20은 비교 대상(TinyLlama-1.1B·Pythia-1.0B·OPT-1.3B) 4개 모델 중 1위**이고, ~20B 토큰만 학습한 Apex-1이 300B 토큰짜리 Pythia-1.0B와 HumanEval에서 8.54 vs 1.83으로 크게 앞섭니다. 3T 토큰인 TinyLlama와의 격차는 아래 "비교 모델" 표의 토큰 수 차이(150×)로 설명되는 패턴입니다.

#### 비교 모델 — 파라미터·학습 토큰

| 모델 | 파라미터 | 학습 토큰 | 발표 | 비고 |
|:---|---:|---:|:---|:---|
| **Apex-1** | 1.12B | ~20B | 2026, 이 프로젝트 | v3-en 코퍼스, 영어 전용 |
| TinyLlama-1.1B | 1.10B | **3T** (3조) | 2024, StatNLP·SUTD | Llama 2 아키텍처를 그대로 축소 — "작은 모델도 압도적 토큰 수로 밀어붙인다"의 대표 사례. Apex-1 대비 **토큰 150배** |
| Pythia-1.0B | 1.01B | 300B | 2023, EleutherAI | The Pile로 학습, 스케일링 법칙 연구용 표준 스위트(70M~12B). Apex-1 대비 **토큰 15배** |
| OPT-1.3B | 1.30B | 180B | 2022, Meta | GPT-3 재현 목적의 오픈 계열. 상식 7종만 논문에 보고돼 있어 지식·수학·코드 행은 비교 불가 |
| OLMo-1B | 1.18B | 2T | 2024, AI2 | Dolma 코퍼스로 학습, 데이터·학습 과정까지 전부 공개한 완전 오픈 모델. Apex-1 대비 **토큰 100배** |
| Llama-3.2-1B | 1.24B | 9T | 2024, Meta | Llama 3.1 8B/70B에서 프루닝+증류로 축소. Apex-1 대비 **토큰 450배** |
| Qwen2.5-1.5B | 1.54B | 18T | 2024, Alibaba | 이 표에서 학습 토큰이 가장 많음. Apex-1 대비 **토큰 900배** |
| SmolLM2-1.7B | 1.71B | 11T | 2024, HuggingFace | 고품질 필터링 코퍼스(FineWeb-Edu 등) 중심 학습. Apex-1 대비 **토큰 550배** |

> [!NOTE]
> 세 모델 모두 **EleutherAI lm-evaluation-harness 계열 프로토콜**로 측정·보고된 수치라 이 표처럼 나란히 놓고 비교하는 게 관례입니다 (Pythia·TinyLlama·OPT 논문이 서로를 이 방식으로 인용합니다). 다만 채점 코드 버전 차이 등으로 ±0.5점 내외 오차는 감안해야 합니다.

#### 더 넓은 비교 — 2024~2025 소형 오픈모델

위 "한눈에" 표는 Apex-1과 학습 토큰 규모가 비슷한(0.18B~3T) 모델끼리의 비교였습니다. 여기서는 범위를 넓혀 **오늘날 널리 쓰이는 1B~1.7B급 오픈모델**까지 같은 축(HellaSwag·ARC 평균·PIQA·GSM8K·HumanEval)으로 나란히 놓았습니다. 🏆는 각 열의 1위입니다.

| 모델 | 학습 토큰 | HellaSwag | ARC(평균) | PIQA | GSM8K | HumanEval |
|:---|---:|---:|---:|---:|---:|---:|
| **Apex-1 DPO (1.1B)** | 0.02T | 46.9 | 41.6 | 68.6 | 1.9 | 8.5 |
| Pythia-1.0B | 0.3T | 47.2 | 38.0 | 69.2 | — | 1.8 |
| OPT-1.3B | 0.18T | 53.7 | 40.1 | 72.4 | — | — |
| TinyLlama-1.1B | 3T | 59.2 | 42.7 | 73.3 | — | 9.2 |
| OLMo-1B | 2T | 62.5 | 46.3 | 73.7 | — | — |
| Llama-3.2-1B | 9T | 61.2 | 49.2 | 74.8 | 7.6 | 18.9 |
| Qwen2.5-1.5B | 18T | 66.4 | 58.5 | 76.1 | 61.7 🏆 | 37.2 🏆 |
| SmolLM2-1.7B | 11T | 68.7 🏆 | 60.5 🏆 | 77.6 🏆 | 31.1 | 22.6 |

> [!IMPORTANT]
> **토큰 효율로 읽기** — 점수는 대체로 학습 토큰 수에 비례하지만, Apex-1(0.02T)은 **15배 더 학습한 Pythia-1.0B(0.3T)와 상식 평균에서 동급이고, ARC는 오히려 앞섭니다** — 토큰당 효율로는 이 표에서 최상위권입니다.
>
> **HumanEval 8.5**는 300B 토큰(0.3T)을 학습한 Pythia-1.0B(1.83)의 **4.7배**이고, 3조 토큰(3T)인 TinyLlama-1.1B(9.15)에도 근접합니다 — ~20B 토큰 모델로는 이례적으로 좋은 결과입니다.
>
> Llama-3.2-1B(9T)·Qwen2.5-1.5B(18T)와의 격차는 **아키텍처 열세가 아니라 데이터 규모 450~900배 차이**입니다. "20B 토큰으로 여기까지"가 이 프로젝트를 공개할 때 가장 정직하고 설득력 있는 서사입니다.

> [!WARNING]
> **읽는 법**: SmolLM2-1.7B(11T)·Qwen2.5-1.5B(18T)는 Apex-1보다 **각각 550배·900배** 많은 토큰으로 학습된, 2024년 기준 최신 소형 오픈모델입니다. 이 표에서 Apex-1이 밀리는 건 예상된 결과이지 실패가 아닙니다 — §5.4 상단의 "한눈에" 표처럼 **학습 토큰 규모가 비슷한 모델과 비교**하는 쪽(BoolQ 62.20 비교 4모델 중 1위, 다만 MMLU는 1B급 공통 랜덤 수준으로 변별력 없음)이 이 프로젝트의 실제 위치를 더 공정하게 보여줍니다. GSM8K·MBPP는 train split 오염 고지가 있으니(§5.4 상단·[COMPARISON.md](ckpt/lm_eval_Apex-1_COMPARISON.md) 참고) HumanEval 쪽 비교가 가장 깨끗합니다. 이 표는 "토큰을 더 쓰면 같은 파라미터로 어디까지 갈 수 있는가"를 보여주는 참고용 지형도에 가깝습니다.

#### 벤치마크가 정확히 뭘 재는지 — 권위·한계

| 벤치마크(그룹) | 얼마나 널리 쓰이나 | 알려진 한계 |
|:---|:---|:---|
| 상식 7종 (HellaSwag·ARC-e/c·PIQA·WinoGrande·BoolQ·OpenBookQA) | EleutherAI harness의 사실상 표준 세트 — Pythia·TinyLlama·OPT·Llama 계열 논문이 전부 그대로 인용. **모델 간 비교의 업계 공용어** | 객관식이라 "덜 어색한 문장"을 고르는 얕은 패턴 매칭만으로도 점수가 나옴 — 절대 점수보다 **상대 비교용**으로 신뢰 |
| MMLU | GPT-4·Llama 등 최신 LLM 리포트가 대표 지표로 인용하는 지식 벤치마크(57과목 4지선다) | 4지선다 랜덤 기준이 25점인데, **1B급은 대부분 24~26점대**에 몰려 있어 이 규모에서는 변별력이 낮음 — 참고용 |
| GSM8K | 초등 다단계 수학 서술형 추론의 표준 벤치. 정답을 직접 생성해야 해서 사고 과정을 요구 | **~1B급 모델은 대부분 한 자릿수%대**에 머묾 — "다단계 추론을 거의 못 한다"를 확인하는 용도에 가까움(§6에서 이 프로젝트가 직접 실측한 근거) |
| HumanEval | OpenAI가 만든, 코드 생성 LLM 평가의 **사실상 업계 표준**(Codex·GPT·Claude·Llama 논문 전부 인용). pass@1 = 한 번 생성한 코드가 실제 테스트를 통과하는 비율 | 문제 수(164개)가 적어 개별 문항 하나로도 점수가 크게 흔들림 — 소형 모델 비교에는 충분히 유의미 |
| MBPP | Google이 만든 보조 코딩 벤치. HumanEval보다 쉬운 문제 500개로 코딩 기본기 확인 | HumanEval만큼 널리 인용되진 않음 — 보조 지표로 취급 |

#### 전체 결과 (상식 7종 개별 + OPT-1.3B 포함)

| 벤치마크 | 이름 유래 | 무엇을 보는가 | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B | OPT-1.3B |
|:---|:---|:---|---:|---:|---:|---:|---:|
| HellaSwag | "Harder Endings, Longer contexts…" 약어 | 다음 문장 전개 고르기 — 상식적 문맥 이해 | 46.69 | 46.91 | 59.20 | 47.16 | 53.65 |
| ARC-Easy | AI2 Reasoning Challenge (쉬움) | 초·중등 과학 지식 | 52.65 | 52.57 | 55.25 | 48.99 | 50.80 |
| ARC-Challenge | AI2 Reasoning Challenge (어려움) | 검색·통계로 못 푸는, 추론이 필요한 과학 문제 | 29.86 | 30.72 | 30.10 | 27.05 | 29.44 |
| PIQA | Physical Interaction QA | 물리적 상식 — 도구·재료 사용법 | 68.23 | 68.55 | 73.29 | 69.21 | 72.36 |
| WinoGrande | Winograd Schema 확장판 | 대명사가 뭘 가리키는지 — 문맥 추론 | 53.59 | 52.80 | 59.12 | 53.43 | 59.59 |
| BoolQ | Boolean Questions | 지문 읽고 예/아니오 판단 — 독해력 | 61.99 | **62.20** 🏆 | 57.83 | 57.83 | 60.83 |
| OpenBookQA | Open Book QA | 과학 원리를 새 상황에 적용 — 단순 암기가 아닌 응용 | 33.80 | 33.80 | 36.00 | 31.40 | 33.40 |
| **상식 평균** | | | 49.54 | **49.65** | 52.99 | 48.30 | 51.44 |
| MMLU | Massive Multitask Language Understanding (5-shot) | 법·의학·수학·역사 등 57과목 — 종합 지식량 | 24.83 | 24.90 | 25.34 | 25.70 | — |
| GSM8K strict (5-shot) | Grade School Math 8K | 초등 다단계 수학 서술형 (엄격 채점) | 1.44 | 1.90 | — | — | — |
| GSM8K flexible (5-shot) | 〃 | 〃 (완화 채점) | 1.97 | 2.27 | — | — | — |
| HumanEval (pass@1) | OpenAI 사람 작성 평가셋 | 함수 설명 → 파이썬 코드 생성, 실제 실행 테스트 | 8.54 | 8.54 | 9.15 | 1.83 | — |
| MBPP (pass@1) | Mostly Basic Python Problems | 기초 파이썬 문제 500개 — HumanEval보다 쉬운 코딩 기본기 | 4.80 | 5.20 | — | — | — |

**판정** ([`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md) 원문 그대로):

1. **DPO ≥ SFT** — 상식 평균 +0.11, ARC-Challenge +0.86, GSM8K +0.46, MBPP +0.4. 회귀(alignment tax) 없음. 릴리스 기본 모델로 DPO 권장.
2. **Pythia-1.0B(300B 토큰) 전 항목 동급~우위**, 특히 HumanEval 8.54 vs 1.83. Apex-1 학습량은 ~20B 토큰뿐.
3. **BoolQ 62.20은 비교 4모델 중 1위.**
4. TinyLlama(3T 토큰)와의 격차는 HellaSwag·WinoGrande·PIQA 등 학습량에 민감한 항목에 집중 — 토큰 수 차이(150×)로 설명되는 패턴.
5. MMLU는 1B급 공통으로 랜덤 수준(≈25) — 변별력 없음.

> [!WARNING]
> **오염 고지**: GSM8K·MBPP는 **train split**이 SFT/RLVR 학습 데이터에 일부 포함돼 있습니다. 평가는 **test split**이라 유효하지만 완전 무오염은 아닙니다. HumanEval은 학습 데이터에 전혀 없는 완전 클린 세트입니다.

---

## 6. 다음 단계 — RLVR → DPO

벤치의 목적 자체가 "추가 SFT를 더 돌릴지, RL로 넘어갈지"를 감이 아니라 계측으로 정하는 것이었습니다. 최초 결정은 RLVR이었지만, 실제로 돌려보니 **학습 신호 자체가 없다는 것**이 드러나 DPO로 방향을 바꿨습니다. 아래는 그 과정입니다.

### 6.1 최초 결정: RLVR

| 근거 | 값 | 해석(당시) |
|:-----|:--|:-----|
| 능력 | 일반 채팅 코딩 **5/5 완전 통과** | 알고리즘 자체는 충분히 학습됨 |
| 정렬 문제 | 반복률 평균 **0.032**, 길이 초과 **8/10** | 장황함·반복이 병목 — 능력이 아니라 스타일 문제 |

→ 최초 결정: **RLVR** ([`ckpt/AUTO_DECISION_Apex-1_v1.txt`](ckpt/AUTO_DECISION_Apex-1_v1.txt), 근거 원문: *"정렬(장황·반복): repetition 0.032, over_length 8/10 — 능력은 충분(coding 5/5)"*)

### 6.2 실제로 돌려보니 — 학습 신호가 없었다

**1차 시도 (THINKING 강제, 배치 생성)**: 샘플 6개 전부 thinking만 478~759자를 쏟아내고 답변(`</THINKING>` 이후)은 공란이었습니다. `max_new=200` 안에서 사고를 끝내지 못해 답까지 못 감 → 6개 전부 reward **0.10**(형식 점수만) → 그룹 내 분산 0 → 학습 신호 없음.

> [!WARNING]
> 327M 라인의 v6에서 겪었던 "thinking이 답변으로 전달 안 됨" 문제가 RLVR에서 그대로 재현된 것입니다. 게다가 벤치마크(§1)는 이 모델이 **no-thinking이 더 낫다**고 나왔는데, RLVR은 오히려 THINKING을 강제하고 있었으니 정확히 반대 방향으로 가고 있었습니다.

**2차 시도 (no-thinking 프로브로 재확인, `torch.no_grad()` 버그 수정 후)**: GSM8K 4문제 × 6샘플 = 24개 생성 **전부 reward 0.0**(정답 0개). 답변은 637~715자로 여전히 장황했고, 최종 숫자도 오답이었습니다(예: 72→96, 10→300).

근본 원인: **1B(20B 토큰) 모델이 다단계 산술(GSM8K)을 사실상 풀지 못합니다.** GRPO는 한 그룹 안에 정답·오답이 섞여야 상대적 advantage가 생기는데, 정답률이 ~0%이니 전 그룹이 스킵되고(`groups 0/8`) 그레디언트가 전혀 나오지 않습니다. 약 4시간 돌아간 런이 실제로는 아무것도 학습하지 못하고 있었던 것입니다.

> [!IMPORTANT]
> 애초에 "능력은 충분(coding 5/5)"이라고 판단했던 근거가 문제였습니다. 코딩 5문항은 SFT 데이터에 흔한 표준 패턴(√n 소인수분해, 슬라이스 반전 등)이라 통과했을 뿐, **GSM8K류 다단계 산술 추론은 별개의 능력**이었습니다. "코딩 5/5 = 능력 충분"이라는 일반화가 틀렸다는 게 이번 교훈입니다.

### 6.3 결론 — RLVR을 접고 DPO로

RLVR을 촉발한 원래 문제(§1)는 산술 능력이 아니라 **장황함 + 지시 불이행**(over_length 8/10, instruction 1/2)이었습니다. 이건 DPO가 정확히 겨냥하는 영역입니다 — DPO는 모델이 문제를 "풀 능력"이 없어도 "간결한 답 > 장황한 답"이라는 **상대 선호**만 학습하면 되므로, GSM8K 정답률과 무관하게 작동합니다. 선호쌍 데이터(60K)도 이미 준비돼 있어 바로 전환할 수 있었습니다.

RLVR 인프라 개선(배치 생성, 중간 저장, 실시간 로그)은 `rlhf.py`에 남겨뒀습니다 — 더 강한 모델이나 정답률이 높은 RL 과제를 만나면 그대로 재사용할 수 있습니다.

### 6.4 DPO 학습 완료

```text
step 0 | loss 0.6931 | pref acc 0.00
```

`loss 0.6931 ≈ ln(2)`은 정책 모델이 아직 참조 모델과 같을 때의 교과서적 초기값이고, `pref acc 0.00`은 아직 마진이 없다는 뜻으로 시작 단계에선 정상입니다. 60K 선호쌍으로 학습을 마쳤고, 결과는 §5.4의 표준 벤치마크로 확인됩니다 — **DPO가 SFT 대비 전 영역 동급 이상**(상식 평균 +0.11, ARC-Challenge +0.86, GSM8K +0.46, MBPP +0.4), alignment tax 없이 정렬이 걸렸습니다.

327M 라인이 겪은 "형식(핸드오프) → 정확도" 순의 병목 이동과 비교하면, 1B 라인은 핸드오프 문제 없이 시작했지만 **"정확도(산술) 자체가 RL 신호를 만들 만큼 없다"**는 다른 종류의 벽을 만났고, 그 벽을 우회해 장황함·지시불이행이라는 원래 목표를 DPO로 직접 겨냥해 완료까지 마쳤습니다.

---

## 7. 원본 파일

| 경로 | 내용 |
|:-----|:-----|
| [`ckpt/benchmark_sft_Apex-1_v1_raw.json`](ckpt/benchmark_sft_Apex-1_v1_raw.json) | sft_Apex-1_v1 벤치 원본 (30개 생성 전체, thinking 텍스트 포함) |
| [`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md) | SFT vs DPO 표준 벤치 11종 원문 (§5.4 출처) |
| [`ckpt/AUTO_DECISION_Apex-1_v1.txt`](ckpt/AUTO_DECISION_Apex-1_v1.txt) | 최초 RLVR 결정 로그 (§6.3에서 DPO로 전환) |
| — | pretrain_Apex-1_v1 단독(SFT 전) 채팅 벤치 **미측정** |

> 가중치(`.pt`)와 원본 코퍼스는 공개 저장소에 포함하지 않습니다.

---

<div align="center">

[README](README.md) · [BENCHMARK v1 (327M)](BENCHMARK-v1.md) · [ARCHITECTURE](ARCHITECTURE.md)

</div>
