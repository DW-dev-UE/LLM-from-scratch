[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-0969DA?style=flat-square)](BENCHMARK-v3.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](BENCHMARK-v3.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](BENCHMARK-v3.ja.md)

[← README](README.md)

---

# 📊 BENCHMARK v3 · APEX-2 (MoE 3.87B · 활성 1.45B)

> APEX-1(1.1B dense)과는 다른 계열입니다. **MoE(Mixture-of-Experts)** 구조로 처음부터 사전학습한 모델이고, 코드는 별도 저장소(GeneralLLM)에서 짰습니다.

| | |
|:--|:--|
| 🆕 **최종** | `sft2_chat` (SFT 2단계 step 536) · 2026-09-28 |
| 📦 **모델** | APEX-2 · Decoder-only MoE · 실측 **3,869.1M** (토큰당 활성 **1,453.2M**) |
| 🤗 **공개** | [huggingface.co/YOON1v/Apex-2](https://huggingface.co/YOON1v/Apex-2) — `transformers`·vLLM으로 바로 로드 (Qwen3MoeForCausalLM 매핑) |
| 🧪 **평가** | 코드 5종 · 수학 2종 · IFEval · MMLU · 상식 6종 (vLLM greedy, 채팅 템플릿) |
| 📁 **원본** | [`ckpt/benchmark_sft_Apex-2_v1.json`](ckpt/benchmark_sft_Apex-2_v1.json) · [`ckpt/benchmark_dpo_Apex-2_v1.json`](ckpt/benchmark_dpo_Apex-2_v1.json) |
| ✅ **결론** | Pretrain → SFT 완료. DPO는 성능이 떨어져 폐기 (§7) |

### 목차

1. [한눈에 보기](#1-한눈에-보기)
2. [모델 카드](#2-모델-카드)
3. [학습 과정](#3-학습-과정)
4. [데이터셋](#4-데이터셋)
5. [채점 방법](#5-채점-방법)
6. [비슷한 크기 모델과 비교](#6-비슷한-크기-모델과-비교)
7. [DPO · RLVR 기록](#7-dpo--rlvr-기록)
8. [한계](#8-한계)

---

## 1. 한눈에 보기

| 분야 | 벤치마크 | **SFT (최종)** | base | DPO (폐기) |
|:--|:--|--:|--:|--:|
| 코드 | HumanEval | **43.9** | 36.6 | 36.0 |
| 코드 | HumanEval+ | **41.5** | 32.9 | 32.3 |
| 코드 | MBPP | **56.3** | 54.8 | 45.2 |
| 코드 | MBPP+ | **48.9** | 46.3 | 37.3 |
| 코드 | MultiPL-E HumanEval C++ | **36.0** | — | 8.1 |
| 코드 | MultiPL-E MBPP C++ | **41.6** | — | 3.8 |
| 코드 | LiveCodeBench v5–v6 | 3.2 | — | 3.5 |
| 코드 | └ LCB easy (84) | 11.9 | — | 10.7 |
| 코드 | └ LCB medium (104) | 1.0 | — | 2.9 |
| 코드 | └ LCB hard (154) | 0.0 | — | 0.0 |
| 코드 | CRUXEval-O | 8.5 | — | 12.1 |
| 코드 | CRUXEval-I | 3.2 | — | 8.9 |
| 수학 | GSM8K | **32.4** (0-shot CoT) | 15.1 (8-shot) | 14.3 |
| 수학 | MATH-500 (0-shot CoT) | **21.0** | — | 14.8 |
| 지시 수행 | IFEval prompt strict | **44.7** | — | 35.7 |
| 지시 수행 | IFEval instruction strict | **56.6** | — | 49.2 |
| 지식 | MMLU (5-shot) | 28.6 | 28.2 | 28.3 |
| 상식 | HellaSwag (acc_norm) | 62.2 | 60.8 | 63.1 |
| 상식 | ARC-e (acc_norm) | 64.1 | 66.7 | 64.6 |
| 상식 | ARC-c (acc_norm) | 38.4 | 39.7 | 39.2 |
| 상식 | PIQA (acc_norm) | 74.8 | 73.9 | 74.4 |
| 상식 | WinoGrande (acc) | 61.8 | 60.1 | 63.3 |
| 상식 | LAMBADA (acc) | 52.8 | 54.5 | 51.2 |

- SFT가 코드·수학·지시 수행을 올렸고, 지식·상식은 사전학습 수준 그대로입니다 (±2 이내).
- base의 GSM8K는 8-shot이라 SFT의 0-shot CoT와 방식이 다릅니다.

## 2. 모델 카드

| 항목 | 값 |
|:--|:--|
| 구조 | Decoder-only Transformer, pre-norm, **모든 층이 MoE** (dense 층 없음) |
| 층 · d_model | 32 · 2,048 |
| 어텐션 | GQA 16Q / 4KV, head_dim 128, QK-RMSNorm, bias 없음 |
| 위치 | RoPE θ = 10,000 · 컨텍스트 4,096 (문서 경계 마스킹) |
| FFN | SwiGLU MoE: expert 16개, 토큰당 **top-4**, expert 폭 1,024 (토큰당 FFN 폭 4,096) |
| 라우터 | Linear 2048→16, fp32 softmax → top-4 → 확률 재정규화 |
| 부하 균형 | aux loss 0.01 (전역 배치 기준) + router z-loss 1e-3 |
| 어휘 | 151,936 (Qwen3 토크나이저), 임베딩–lm_head tie |
| HF 형식 | `Qwen3MoeForCausalLM` 1:1 매핑 (export 검증: logits 상대 차이 1.0e-6, 다음 토큰 예측 100% 일치) |

| 구성 | 파라미터 |
|:--|--:|
| 어텐션 | 335.6M |
| expert (32 × 16 × 3 × 2048 × 1024) | 3,221.2M |
| 라우터 · norm | 1.2M |
| 임베딩 (= lm_head) | 311.2M |
| **총** | **3,869,124,608** |
| **토큰당 활성** | **1,453,205,504** (비임베딩 1,142,040,576) |

## 3. 학습 과정

| 단계 | 토큰 (누적) | 설정 |
|:--|:--|:--|
| 사전학습 1 | 20.0B (20.0B) | GH200 1장, LR 4e-4 (warmup 1,000 step) |
| 사전학습 2 (CPT) | 46.1B (66.1B) | GH200 1장 → **GH200 2장 DiLoCo** (step 26,000부터, 250 step마다 가중치 평균), LR 4e-4 유지 |
| 사전학습 3 (decay) | 20.4B (**86.5B**) | GH200 2장 DiLoCo, LR 4e-4 → 0 선형, 9,750 step |
| SFT 1단계 | 2.21B (1 epoch) | H100 1장, LR 2e-5 → 0, 2,130 step |
| SFT 2단계 | 0.28B × 2 epoch | LR 1e-5 → 0, 536 step |

- global batch 1,048,576 토큰 (256 × 4,096), AdamW 8-bit (β 0.9/0.95), weight decay 0.1, grad clip 1.0, bf16 연산 + fp32 master.
- 사전학습 합계 54,250 step · 86,507,520,000 토큰.
- SFT 검증 loss: 1단계 0.627 → 0.577, 2단계 0.642 → 0.635 (2 epoch째에도 과적합 없음).

## 4. 데이터셋

**사전학습** (중복 제거된 보유 코퍼스 40개 소스)

| 구간 | 믹스 |
|:--|:--|
| 사전학습 1 | 웹 62 / 코드 22 / 수학 8 / 큐레이션 8 (FineWeb-Edu, DCLM, StarCoder 계열, FineMath, Wikipedia, StackExchange 등) |
| 사전학습 2 | 보유 코퍼스 전량 1회 (웹 · 코드 11개 언어 · 수학 · 논문 · StackExchange · 합성) |
| decay | 웹 30 / 코드 35 / 수학 15 / 큐레이션 20 |

**SFT** (ChatML, assistant 답변에만 loss, 평가셋 13-gram 오염 제거)

| 단계 | 구성 |
|:--|:--|
| 1단계 2.21B (413만 대화) | 범용 29% (Dolci-Instruct) / 코드 44% (OpenCoder SFT, OpenCodeInstruct 통과율 ≥ 0.9, self-oss, Code-Feedback) / 수학 27% (OpenMathInstruct-2, NuminaMath) |
| 2단계 0.28B | 실행 검증 코드 59% / NuminaMath 21% / 정밀 지시 수행 20% |

## 5. 채점 방법

- 생성: vLLM greedy, 채팅 템플릿, 프롬프트 + 답 ≤ 4,096 토큰.
- 모델이 쓴 코드는 bwrap 샌드박스 (네트워크 차단, `/` 읽기 전용)에서 실행해 채점. 정답을 넣은 자체 검사로 채점기를 먼저 검증.

| 벤치마크 | 방식 |
|:--|:--|
| HumanEval(+) · MBPP(+) | EvalPlus 0.3.1, 채팅 프롬프트 |
| LiveCodeBench | code_generation_lite release v5·v6 추가분 (2024-08 이후 342문제), 공식 프롬프트, pass@1 |
| CRUXEval-O / -I | CoT 후 `[ANSWER]assert …[/ANSWER]`, assert 실행 |
| MultiPL-E C++ | 함수 전체를 받아 테스트와 합쳐 `g++ -std=c++17` |
| GSM8K · MATH-500 | 0-shot CoT, `\boxed{}` (MATH는 math-verify) |
| IFEval | lm-eval 0.4.13, 채팅 템플릿 |
| MMLU · 상식 6종 | lm-eval loglikelihood (채팅 템플릿 없음), base와 같은 방식 |

## 6. 비슷한 크기 모델과 비교

다른 모델 점수는 공식 모델 카드·기술 보고서 값입니다. few-shot 수, CoT, IFEval 지표, LiveCodeBench 기간이 모델마다 달라 **대략적인 비교**로만 봐 주세요.

| 모델 | 사전학습 토큰 | HE+ | MBPP+ | GSM8K | IFEval | MMLU |
|:--|--:|--:|--:|--:|--:|--:|
| **Apex-2 SFT (3.87B · 활성 1.45B)** | **0.087T** | **41.5** | **48.9** | **32.4** | **44.7** | **28.6** |
| Qwen2.5-1.5B-Instruct | 18T | (HE 61.6) | (MBPP 63.2) | 73.2 | 42.5 | 50.7 |
| Qwen2.5-Coder-1.5B-Instruct | 5.5T | 66.5 | 59.4 | — | — | — |
| Qwen3-1.7B (non-thinking) | 36T | — | — | — | 68.2 | 64.4 |
| Llama-3.2-1B-Instruct | 9T | — | — | 44.4 | 59.5* | 49.3 |
| Gemma-3-1B-it | 2T | (HE 41.5) | (MBPP 35.2) | 62.8 | 80.2* | 38.8 |
| OLMoE-1B-7B (활성 1.3B) | 5.1T | 54.4 | — | 72.4 | 66.4* | 55.1 |
| DeepSeek-Coder-1.3B | 2T | 60.4 | 54.8 | — | — | — |

<sub>괄호 값은 HumanEval/MBPP. \* IFEval 지표가 다름 (Apex-2·Qwen: prompt-level strict). Qwen2.5·Qwen3 MMLU는 MMLU-Redux, OLMoE MMLU는 CoT.</sub>

**base끼리 비교** (사전학습 직후)

| 모델 | 사전학습 토큰 | HE / HE+ | MBPP / MBPP+ | MMLU | HellaSwag | PIQA | WinoGrande |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Apex-2 base** | **0.087T** | 36.6 / 32.9 | 54.8 / 46.3 | 28.2 | 60.8 | 73.9 | 60.1 |
| Qwen2.5-1.5B | 18T | 37.2 / 32.9 | 60.2 / 49.6 | 60.9 | 67.9 | — | 65.0 |
| Qwen2.5-Coder-1.5B | 5.5T | 43.9 / 36.6 | 69.2 / 58.6 | 53.6 | 61.8 | — | 60.7 |
| Gemma-2-2B | 2T | 17.7 / — | 29.6 / — | 51.3 | 73.0 | 77.8 | 70.9 |
| OLMoE-1B-7B | 5.1T | — | — | 56.3 | 81.7 | 78.7 | 70.6 |

- 사전학습 데이터가 비교 대상의 1/23~1/400인데, base 코드 점수는 Qwen2.5-1.5B(18T)와 같은 수준입니다 (HE+ 32.9).
- 격차가 큰 곳은 지식(MMLU)과 수학이고, 둘 다 사전학습 토큰 수가 좌우합니다.

<sub>출처: Qwen2.5 [blog](https://qwenlm.github.io/blog/qwen2.5-llm/) · [report](https://arxiv.org/abs/2412.15115) · Qwen2.5-Coder [report](https://arxiv.org/abs/2409.12186) · Qwen3 [report](https://arxiv.org/abs/2505.09388) · [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) · Gemma 3 [report](https://arxiv.org/abs/2503.19786) · [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b-it) · [OLMoE Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) · [OLMoE base](https://huggingface.co/allenai/OLMoE-1B-7B-0125) · DeepSeek-Coder [paper](https://arxiv.org/abs/2401.14196)</sub>

## 7. DPO · RLVR 기록

**DPO — 폐기**
- 데이터: allenai/Dolci-Instruct-DPO 22만 쌍 (off-policy), 길이 정규화 DPO (β 5), LR 1e-6.
- 검증 정확도는 step 100에서 81%로 거의 포화했는데, chosen NLL이 1.023 → 1.161로 계속 올랐습니다 (chosen까지 확률이 내려가는 likelihood displacement). step 300에서 멈췄습니다.
- 평가 결과 평균 답 길이가 320 → 733 토큰(2.3배)으로 늘어 길이 한도에 걸린 답이 크게 늘었고, 코드·수학·지시 수행이 모두 떨어졌습니다 (§1 표).

**RLVR (GSPO) — 중단**
- 코퍼스: OpenCodeInstruct (단위 테스트) · GSM8K · MATH · DeepMath · IFBench 학습셋 (모두 자유 라이선스), 보상은 샌드박스 실행 · math-verify · IFEval 제약.
- DPO 모델에서 시작했다가 DPO 결과를 보고 step 33에서 멈췄습니다.

## 8. 한계

- 영어 중심. 한국어 등 다른 언어는 거의 못 합니다 (SFT에서 다국어 제외).
- 지식이 부족해 사실 오류(환각)가 잦습니다 (MMLU 28.6).
- 경쟁 프로그래밍(LiveCodeBench medium 1%)과 코드 실행 추론(CRUXEval)이 약합니다.
- 컨텍스트 4,096 토큰.
