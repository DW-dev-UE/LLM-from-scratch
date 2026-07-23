[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-0969DA?style=flat-square)](README.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](README.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](README.ja.md)

# 밑바닥부터 만드는 LLM

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)

> [!NOTE]
> **이 저장소는 계속 다듬는 중입니다.**  
> 학습 결과, 벤치마크, 문서가 버전마다 바뀔 수 있습니다. 최신 커밋을 기준으로 봐 주세요. Issue / PR 환영합니다.

> 시작하기에 앞서, 이 글은 가독성이 좋지 않을 수 있습니다. 
> AI를 사용하여 "가독성이 좋게 다듬어줘" 라는 말을 할 수 있지만, 나중에 제가 어떤 생각을 했고 어떤 과정을 거쳤는지 없어질 수 있기때문입니다.
> 물론 AI를 통해 배우는것은 좋지만 문장을 몇 번씩 다듬으면서 나의 지식을 정리하는게 더 중요합니다.

> [!IMPORTANT]
> **APEX-1 (1B급) — Pretrain + SFT 완료**
>
> 실측 **1,119.5M** 파라미터. 모델명 **APEX-1** (코드상 preset명은 `xl` — `model.py`의 크기 티어 키일 뿐이며, 체크포인트 파일명은 `sft_xl_v1`처럼 preset 기준으로 자동 생성됩니다).
>
> - **구조**: 24층 · d_model 2048 · GQA(16Q/4KV) + RoPE(θ=500K) + SwiGLU + RMSNorm · weight tying
> - **컨텍스트**: max 4096 (학습 2048)
> - **Pretrain**: bf16 · lr 3e-4 코사인 · 51K step (20B 토큰) · 코퍼스 v3-en 80.8GB (영어+코드) · vocab 32K 영어 전용
> - **SFT**: `sft_xl_v1` · 8.4K step · lr 3e-5 · `pretrain_xl_v1` 위 SFT
> - **벤치 (영어 전용 15문항 · greedy)**: no-thinking 코딩 완전 통과 **5/5**(테스트 25/25) · QA 5/8 — thinking 모드는 코딩 3/5(테스트 15/25) · QA 4/8로 오히려 하락 → [BENCHMARK v2](BENCHMARK-v2.md)
> - **다음 단계**: 장황함·반복이 병목(over-length 8/10, repetition 0.032)이라 능력 자체는 충분 판단 → **RLVR 진행 결정** ([근거](ckpt/AUTO_DECISION_xl_v1.txt))
> - **미채택**: MoE · YaRN · FP8 · 멀티GPU (1B 검증 후 다음 스케일)

---

좋아요, 그럼 시작하겠습니다.

## 이 프로젝트가 하는 일

NVIDIA CUDA 위에서 돌아가는 **범용 LLM**과, 코딩·문서 작업처럼 실무에 쓸 수 있는 AI를 **밑바닥부터** 만드는 실험입니다.

- 토크나이저 학습
- 모델 아키텍처 (Decoder-only Transformer)
- 사전학습 · SFT · DPO · GRPO
- KV 캐시 추론 · 사고 모드 (THINKING)
- 사람 피드백 수집 UI · 승격 게이트

오픈소스 LLM을 받아 파인튜닝하는 편이 훨씬 빠르지만, 그럼에도 처음부터 다시 짠 이유는 하나뿐입니다.

LLM 내부를 손으로 짜 보지 않으면, 결국 남의 코드를 빌려 쓰는 쪽에 머문다.

> LLM 내부를 손으로 짜 보지 않으면, 결국 남의 코드를 빌려 쓰는 쪽에 머문다.

그래서 이 저장소는 **외부 상용/오픈 LLM 가중치에 기대지 않습니다.**

```ini
[326.7M급 모델을 개발하면서 느낀 교훈]

1. 다국어 비효율

작은 모델 조건부에서 다국어를 지원하는 것은 ~7B급 모델에서는 치명적이다. v7에서 보여줬듯, 소형 다국어는 셋 다 어중한간 항태로 수렴한다.

70B급 이상부터는 언어간 전이가 오히려 이득이 되지만, 작은 모델에서는 영어권 코퍼스가 압도적으로 양과 질이 좋으므로, 작은 모델에서는 영어에 초점을 맞춰야 한다.

2. 코퍼스의 질은 상당히 중요하다.

성능 = 파라미터 x 토큰 수 x 데이터 질이다. 기존에 공개된 코퍼스라 할지라도, 좋은 질의 코퍼스를 찾는것이 가장 급선무이다. 

3. 벤치마크에서의 노이즈

학습을 끝낸 후 벤치마크를 돌렸을 때, temperature 0.7값으로 문항당 1번만 샘플링했다. 같은 모델이 실행마다 다른 답을 내니 버전 간 점수 차이가 실력 차인지 주사위인지 구분이 불가능했다.

해결: v5부터 greedy 프로토콜을 사용하여 같은 모델은 항상 같은 답을 사용하게 하여 해결했다.

다만, val loss에도 속았다.

val이 코퍼스 파일 순서의 마지막 1%였는데, ko위키 꼬리 단일 도메인이라 전체 능력을 대표하지 못했다. (v2에서 ko위키 val은 나빠졌지만 다운스트림 벤치마크는 역대 최고)

측정이 무작위 16시퀀스 1배치라 +-0.25 노이즈가 있었는데, 이 등락을 실제 개선/악화로 오독해 두 번 판단을 실수했다. 즉, 두 번 속았다.

해결: 다중 배치 평균으로만 판단 + v3-en 다운로더에 소스별 1%씩 혼합 val을 구조적으로 내장했다.

4. thinking 붕괴 (v1~v5, 최대 실수)

정답을 <THINKING> 안에 써놓고 답변은 공란이었다. 6버전 내내 thinking 코딩 0/5
원인으로는 총 3개가 있었다.

> thinking 학습 데이터의 99%가 max_len=1024에서 사고 도중 절단되었다.
> THINK_SUFFIX가 학습/추론 간 불일치했다.
> infer.py에서 사고/답변 토큰 예산이 미분리라 사고가 예산을 다 먹으면 답변이 공란이었다.

해결: data.py에 thinking 꼬리 보존(길면 사고 중간을 잘라도 닫는 태그+답변은 보존: 47,273건, 너무 길면 no-thinking으로 강등: 17,540건) + THINK_SUFFIX 조건부 부착 + infer.py 예산 분리·EOS 가드

해결법을 적용하니 thinking 답변 10/14→14/14, 평균 0.50→2.93, 첫 코드 출력 4/5 점수를 받았다.

공통적인 교훈으로는 셋 다 모델이 멍청해서가 아니라, 측정기나 배관이 고장나있었다.
초점을 학습 모델에 맞추다 보니 생긴 문제였는데, 다음번부터 점수가 이상해도 샘플링, val 설계, 전처리 부터 의심하는 것을 순서로 둔다.
```

---

## 문서 안내

| 문서 | 무엇을 보나요 |
|:-----|:--------------|
| **README** (지금 문서) | 설계 철학, 파라미터 전략, 구조, 실행 |
| [GLOSSARY](GLOSSARY.md) | Transformer, RoPE, DPO 같은 용어 |
| [ARCHITECTURE](ARCHITECTURE.md) | 트랜스포머 워크스루 · 모델 설계 · 토크나이저 · 학습 · 추론 |
| [POST-TRAINING](POST-TRAINING.md) | 배포 후 인간 피드백 루프 |
| [BENCHMARK v2](BENCHMARK-v2.md) | APEX-1(1B) 벤치마크 · RLVR 결정 근거 |
| [BENCHMARK v1](BENCHMARK-v1.md) | Base 모델(327M) 벤치마크 (학습 과정 · Q&A 포함) |
| [ThinkingLab](ThinkingLab/ThinkingLab.md) | 가설 · 브레인스토밍 로그 (아직 검증되지 않은 생각들) |

---

## 1. 아키텍처

토큰 → 임베딩 → 어텐션 → FFN → logit 까지의 **숫자 예제 워크스루**는 [ARCHITECTURE — 트랜스포머 동작 원리](ARCHITECTURE.md#2-트랜스포머-동작-원리-교육용-워크스루)에 옮겼습니다. (교육용 예제와 실제 LLaMA 스타일 설계의 차이는 그 문서에서 이어서 설명합니다.)

### 처음에 생각했던 길

처음 계획은 단순했습니다.

1. 자연어 데이터만으로 LLM을 먼저 만든다
2. 그 위에 코딩 특화 데이터를 얹는다
3. 사람이 강화학습으로 다듬는다

코딩 AI라도 질문은 자연어로 오니까,  
**“말을 알아듣는 능력”을 먼저 완성해야 한다**고 봤기 때문입니다.

### 논문이 알려 준 것

[Llama 3](https://ar5iv.labs.arxiv.org/html/2407.21783) 를 보면 그 순서가 없습니다.  
사전학습 단계부터 이미 한 믹스에 섞습니다.

| 비중 (대략) | 데이터 |
|:-----------:|:-------|
| 50% | 자연어 |
| 25% | 수학 · 추론 |
| 17% | 코드 |
| 8% | 다국어 |

후속학습에서 instruction, preference, coding, tool-use 를 강화할 뿐입니다.  
**“자연어를 먼저 끝낸 뒤 코드를 얹는다” 단계는 없습니다.**

[DeepSeek-Coder](https://arxiv.org/html/2401.14196v1) 도 같은 방향입니다.  
코드 87% + 코드 관련 영어 10% + 기타 3% 로 처음부터 학습했고,  
자연어가 13%뿐인데도 당시 오픈 코드 LLM 중 상위권이었습니다.

### 그래서 바꾼 파이프라인

| | 순서 |
|:--|:-----|
| 예전에 생각했던 것 | 자연어만 → 코드만 → instruction |
| **지금 쓰는 것** | **혼합 pretrain** → 코드 비중 강화 → instruction → preference |

```text
  자연어 + 코드 + 수학 + 문서
            │
            ▼
   base pretraining
            │
            ▼
  code-heavy continued pretrain
            │
            ▼
    instruction tuning (SFT)
            │
            ▼
   preference (DPO / GRPO …)
```

### 데이터 믹스 가이드 (300M ~ 1B)

| 종류 | 비율 | 예시 |
|:-----|:----:|:-----|
| Raw source code | 35–45% | 소스 파일 |
| Code-adjacent | 15–20% | README, issue, PR, commit |
| 기술 자연어 | 20–30% | 영어 · 한국어 기술 문서 |
| Math / reasoning | 5–10% | 수학 · 논리 |
| Logs / tests / diffs | 5–10% | 로그, 테스트, diff |
| 한국어 · bilingual | 5–10% | 지시 · 이중 언어 |

## 2. 파라미터는 천천히 키운다

파라미터는 모델 크기입니다. (용어는 [GLOSSARY](GLOSSARY.md))  
클수록 여지는 커지지만, 이 프로젝트는 **300M → 1B → 3B** 처럼 단계를 나눕니다.

[Chinchilla](https://arxiv.org/abs/2203.15556) 요지:

> [!TIP]
> 같은 compute 예산이면 **모델 크기와 토큰 수를 같이** 키워야 한다.

당시 대형 모델은 크기에 비해 데이터가 부족한 경우가 많았고,  
더 작은 Chinchilla(70B, 1.4T 토큰)가 더 큰 Gopher보다 나은 점수를 내기도 했습니다.

그래서 원칙은 이렇습니다.

1. 작은 모델에서 **tokenizer · dataloader · loss · eval · checkpoint** 가 도는지 확인  
2. 그다음 규모를 올린다  

[StarCoder2](https://arxiv.org/abs/2402.19173) 도 3B / 7B / 15B 를 각각 다른 토큰 예산으로 나눠 학습한 사례입니다.

### 이 프로젝트가 밟는 단계

| 규모 | 목적 |
|:-----|:-----|
| 10M ~ 50M | 학습 루프, tokenizer, loss 감소 확인 |
| 100M ~ 300M | completion, FIM, 기본 instruction |
| 1B | 작은 coding assistant 실험 |
| 3B | 사내 도구 연동 후보 |
| 7B+ | 외부 노출을 고민할 최소 규모 |

두 라인을 병행 중입니다: 1B `xl` 라인(APEX-1) 최신은 **`sft_xl_v1`** (pretrain+SFT 완료 · RLVR 대기), 327M `base` 라인 최신은 **`sft_base_v6`** 입니다. → [§5 벤치마크 한눈에](#5-벤치마크-한눈에) · 버전별 상세 기록 [BENCHMARK v2](BENCHMARK-v2.md)

---

## 3. 모델 vs RAG · Tool

회사 지식을 전부 가중치에 넣으려 하면 감당이 안 됩니다.  
코드, 문서, 빌드 로그는 계속 바뀌기 때문입니다.

| 담당 | 맡는 것 |
|:-----|:--------|
| **모델 (가중치)** | 사고 방식, 코딩 패턴, 문맥 이해, 답변 톤, 도구 사용 **방법** |
| **RAG / Tools** | 최신 사실, 사내 문서, repo, 테스트 실행, 빌드 결과 |

> [!IMPORTANT]
> 모델 = **어떻게 생각하고 도구를 쓸지**  
> RAG / Tool = **지금 무엇이 사실인지**

(RAG · Tool 런타임 연동은 로드맵에 있고, 현재 공개 범위의 핵심은 학습 · 추론 · 정렬 파이프라인입니다.)

---

## 4. 프로젝트 구조

```text
AI/
├── README.md                 설계 철학 · 실행 안내
├── GLOSSARY.md               용어
├── ARCHITECTURE.md           모델 · 학습 · 추론
├── POST-TRAINING.md          피드백 후속학습
├── BENCHMARK-v2.md           APEX-1(1B) 벤치 리포트
├── BENCHMARK-v1.md           327M 벤치 리포트
└── llm/                      구현
    ├── model.py              RMSNorm + RoPE + SwiGLU + GQA
    ├── tokenizer.py
    ├── data.py
    ├── train.py              pretrain / sft / dpo
    ├── infer.py              채팅 · 로그
    ├── feedback.py           피드백 웹 UI
    ├── reward.py · rlhf.py   RM + GRPO / RLVR
    └── eval_gate.py          승격 게이트
```

> 대용량 가중치(`.pt`), 원본 코퍼스, 일부 소스는 저장소에 올리지 않습니다.  
> 문서로 설계와 벤치를 먼저 공개하는 형태입니다.

---

## 5. 벤치마크 한눈에

**학습 GPU**: H100, A100

두 라인이 병행됩니다: **1B `xl`**(APEX-1, 영어 전용) 와 **327M `base`** (한/일/영 다국어). 세트도 서로 다릅니다.

### 5.1 1B `xl` 라인 (APEX-1) ⭐ 주력

영어 전용 15문항 × THINKING on/off (코딩 5문항은 327M 세트와 동일 문항).  
최신이자 유일한 스냅샷: **`sft_xl_v1`** (`ckpt/benchmark_sft_xl_v1_raw.json`, 2026-07-22). Pretrain 51K step(20B 토큰) + SFT 8.4K step 완료.

**sft_xl_v1**

| | 💬 일반 채팅 | 🧠 THINKING 켬 |
|:--|:--:|:--:|
| QA 정답(키워드) | 5/8 | 4/8 |
| 코딩 완전 통과 | **5/5** (테스트 25/25) | 3/5 (테스트 15/25) |
| 빈 답변 | 0건 | 0건 |
| 길이 초과 | 8/10 | 5/10 |
| 평균 반복률 | 0.032 | 0.042 |

327M 라인의 최대 실패였던 THINKING 핸드오프(답 칸 공란)는 이 라인에서 **처음부터 발생하지 않았습니다.** 대신 새 병목은 장황함·반복이고, THINKING을 켜면 코딩·QA 모두 오히려 떨어집니다(사고가 답변 예산을 먼저 소모).

능력(코딩 5/5 만점)은 충분하다고 판단해 다음 단계는 추가 SFT가 아니라 **RLVR**로 결정했습니다.

상세 기록: [BENCHMARK-v2.md](BENCHMARK-v2.md)

<details>
<summary><b>5.2 327M `base` 라인 (펼쳐서 보기)</b></summary>

<br/>

`base` (~327M) 기준, 동일 14문항 × THINKING on/off.  
최신 스냅샷: **`sft_base_v6`** (`ckpt/benchmark_sft_base_v6.json`, 2026-07-15).

| 체크포인트 | 요약 |
|:-----------|:-----|
| pretrain v1 | 채팅 형식에서 사실상 0점 (지시 미학습) |
| sft v1 | 지시는 시도함. 정답률은 낮음. THINKING 답 0/14 (태그 미종료) |
| sft v2 | THINKING 종료 대부분 복구 (비어 있지 않은 답 13/14). 코딩은 여전히 약함 |
| sft v3 | 코딩(일반 채팅) 4/5 완전 통과. 영어 강세, 한국어 전 문항 0점. THINKING 코딩은 여전히 0/5 |
| sft v4 | 한국어 SFT 비중↑ (10.3%→22.5%). 벤치는 거의 횡보, 한국어 회복은 미미 |
| sft v5 | 베이스를 pretrain_v2로 교체 + greedy 디코딩. 코딩 첫 만점(5/5), 한국어 회복 시작(8/30) |
| **sft v6** | **THINKING 핸드오프 해결.** prep_sft 전처리 + 추론 예산 분리만으로 사고 평균 0.50→2.93, 사고 코딩 첫 통과(4/5) |

**sft_base_v6 · 일반 채팅 모드 (THINKING 끔)** · 평균 점수 3.93 / 5

| 영역 | 결과 |
|:-----|:-----|
| 한국어 | 평균 **2.67 / 5** (fact 문항 첫 만점) |
| 일본어 | 평균 3.00 / 5 |
| 영어 | 평균 **4.33 / 5** |
| 코딩 | 테스트 **25 / 25** (완전 통과 **5 / 5**) |
| THINKING 켬 | 평균 **2.93 / 5** · 비어 있지 않은 답 **14/14** · 코딩 완전 통과 **4/5** (+prime 부분 3/5) |

v6 하이라이트 (v5 대비):

- **THINKING 핸드오프 문제 해결**: 빈 답 4/14 → 0/14 (전부 답변 생성), THINKING 평균 0.50 → 2.93 (약 6배)
- THINKING 코딩 **0/5 → 4/5** — 프로젝트 6개 버전 만에 첫 실제 코드 출력 (`+prime` 부분 통과 3/5)
- 일반 채팅 평균도 동반 상승: **3.21 → 3.93**, 코딩 5/5 만점 유지
- 한국어 fact 최초 만점 · 일본어 수학 양쪽 모드 첫 정답 (한국어 합 8/30 → 10/30)
- 베이스·SFT 소스 믹스는 v5와 완전히 동일 — 변화는 오직 데이터 전처리(`prep_sft`)와 추론 예산 분리뿐
- 남은 문제: 한국어 산술은 여전히 붕괴, 긴 생성의 반복/퇴화, 일→영 번역 미형성

학습 과정·데이터셋·문항별 Q&A (버전별 상세 기록):

- [BENCHMARK-v1.md](BENCHMARK-v1.md) (한국어)  
- [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md)  
- [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md)  

아직 early-stage 입니다. 버전을 올릴 때마다 같은 문항으로 비교할 예정입니다.

</details>

---

## 6. 참고 문헌

| 논문 | 링크 |
|:-----|:-----|
| Llama 3 Herd of Models | https://ar5iv.labs.arxiv.org/html/2407.21783 |
| DeepSeek-Coder | https://arxiv.org/html/2401.14196v1 |
| Chinchilla (compute-optimal) | https://arxiv.org/abs/2203.15556 |
| StarCoder 2 | https://arxiv.org/abs/2402.19173 |

---

<div align="center">

**이어서 읽기**

[용어](GLOSSARY.md) · [아키텍처](ARCHITECTURE.md) · [후속학습](POST-TRAINING.md) · [벤치마크 v1](BENCHMARK-v1.md) · [벤치마크 v2](BENCHMARK-v2.md) · [생각 실험실](ThinkingLab/ThinkingLab.md)

</div>
