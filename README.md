<div align="center">

# LLM-from-scratch

**토크나이저부터 학습 · 추론 · 피드백 루프까지, 외부 LLM 없이 직접 구현**

GPT-2급 Decoder-only Transformer · 범용 LLM · 회사용 AI 실험

<br/>

![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?style=flat-square&logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![From scratch](https://img.shields.io/badge/from%20scratch-no%20external%20LLM-7c3aed?style=flat-square)
![Status](https://img.shields.io/badge/status-actively%20updated-22c55e?style=flat-square)

<br/>

[한국어](README.md) · [English](README.en.md) · [日本語](README.ja.md)

**Repo:** [github.com/DW-dev-UE/LLM-from-scratch](https://github.com/DW-dev-UE/LLM-from-scratch)

</div>

---

> **이 저장소는 계속 다듬는 중입니다.**  
> 학습 결과, 벤치마크, 문서가 버전마다 바뀔 수 있습니다. 최신 커밋을 기준으로 봐 주세요. Issue / PR 환영합니다.

---

## 이 프로젝트가 하는 일

NVIDIA CUDA 위에서 돌아가는 **범용 LLM**과, 코딩·문서 작업처럼 실무에 쓸 수 있는 AI를  
**밑바닥부터** 만드는 실험입니다.

- 토크나이저 학습  
- 모델 아키텍처 (Decoder-only Transformer)  
- 사전학습 · SFT · DPO · GRPO  
- KV 캐시 추론 · 사고 모드 (`THINKING`)  
- 사람 피드백 수집 UI · 승격 게이트  

오픈소스 LLM을 받아 파인튜닝하는 편이 훨씬 빠릅니다.  
그래도 처음부터 다시 짠 이유는 하나뿐입니다.

> LLM 내부를 손으로 짜 보지 않으면, 결국 남의 코드를 빌려 쓰는 쪽에 머문다.

그래서 이 저장소는 **외부 상용/오픈 LLM 가중치에 기대지 않습니다.**

---

## 문서 안내

| 문서 | 무엇을 보나요 |
|:-----|:--------------|
| **README** (지금 문서) | 설계 철학, 파라미터 전략, 구조, 실행 |
| [GLOSSARY](GLOSSARY.md) | Transformer, RoPE, DPO 같은 용어 |
| [ARCHITECTURE](ARCHITECTURE.md) | 모델 · 토크나이저 · 학습 · 추론 |
| [POST-TRAINING](POST-TRAINING.md) | 배포 후 인간 피드백 루프 |
| [BENCHMARK v1](BENCHMARK-v1.md) | Base 모델 벤치마크 (학습 과정 · Q&A 포함) |

구현 코드: [`llm/`](llm/)  
명령어 모음: [`llm/명령어-정리.md`](llm/명령어-정리.md)

---

## 목차

1. [설계를 이렇게 바꿨다](#1-설계를-이렇게-바꿨다)  
2. [파라미터는 천천히 키운다](#2-파라미터는-천천히-키운다)  
3. [모델 vs RAG · Tool](#3-모델-vs-rag--tool)  
4. [프로젝트 구조](#4-프로젝트-구조)  
5. [실행 방법](#5-실행-방법)  
6. [벤치마크 한눈에](#6-벤치마크-한눈에)  
7. [참고 문헌](#7-참고-문헌)

---

## 1. 설계를 이렇게 바꿨다

### 처음에 생각했던 길

처음 계획은 단순했습니다.

| 단계 | 내용 |
|:----:|:-----|
| **1** | 자연어 데이터만으로 LLM을 먼저 만든다 |
| **2** | 그 위에 코딩 특화 데이터를 얹는다 |
| **3** | 사람이 강화학습으로 다듬는다 |

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

---

## 2. 파라미터는 천천히 키운다

파라미터는 모델 크기입니다. (용어는 [GLOSSARY](GLOSSARY.md))  
클수록 여지는 커지지만, 이 프로젝트는 **300M → 1B → 3B** 처럼 단계를 나눕니다.

[Chinchilla](https://arxiv.org/abs/2203.15556) 요지:

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

현재 벤치에 올린 실학습 체크포인트는 **base 약 327M** 입니다. → [BENCHMARK v1](BENCHMARK-v1.md)

---

## 3. 모델 vs RAG · Tool

회사 지식을 전부 가중치에 넣으려 하면 감당이 안 됩니다.  
코드, 문서, 빌드 로그는 계속 바뀌기 때문입니다.

| 담당 | 맡는 것 |
|:-----|:--------|
| **모델 (가중치)** | 사고 방식, 코딩 패턴, 문맥 이해, 답변 톤, 도구 사용 **방법** |
| **RAG / Tools** | 최신 사실, 사내 문서, repo, 테스트 실행, 빌드 결과 |

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
├── BENCHMARK-v1.md           벤치 리포트
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

## 5. 실행 방법

자세한 명령은 [llm/명령어-정리.md](llm/명령어-정리.md) 에 있습니다.  
데모 한 바퀴 예시:

```bash
cd llm
pip install torch tokenizers numpy

python data.py demo
python tokenizer.py --input data/corpus.txt data/sft.jsonl
python data.py pretrain
python train.py --mode pretrain --preset nano --steps 2000
python data.py sft
python train.py --mode sft --init ckpt/pretrain_nano.pt --steps 300
python infer.py --ckpt ckpt/sft_nano.pt
```

### 피드백 한 사이클 (요약)

```bash
python infer.py --ckpt ckpt/sft_nano.pt
python feedback.py

# A) 교정 SFT
python data.py sft --input data/sft.jsonl data/feedback.jsonl --weight 1 3
python train.py --mode sft --init ckpt/sft_nano.pt --steps 100 --tag v1

# B) DPO
python train.py --mode dpo --init ckpt/sft_nano.pt --steps 100 --tag v1

# C) RM + GRPO / RLVR
python reward.py --init ckpt/sft_nano.pt --steps 200
python rlhf.py --mode rm   --init ckpt/sft_nano.pt --rm ckpt/rm_nano.pt --data data/sft.jsonl
python rlhf.py --mode rlvr --init ckpt/sft_nano.pt --data data/sft.jsonl

python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano_v1.pt
```

실전에서는 코퍼스 · SFT jsonl 을 공개 데이터로 바꾸고  
`small` / `base` 프리셋으로 키우면 됩니다.

---

## 6. 벤치마크 한눈에

`base` (~327M) 기준, 동일 14문항 × THINKING on/off.

| 체크포인트 | 요약 |
|:-----------|:-----|
| pretrain v1 | 채팅 형식에서 사실상 0점 (지시 미학습) |
| **sft v1** | 지시는 시도함. 정답률은 아직 낮음 |

**sft_base_v1 · 일반 채팅 모드 (THINKING 끔)**

| 영역 | 결과 |
|:-----|:-----|
| 한국어 | 평균 1.33 / 5 |
| 일본어 | 평균 0.33 / 5 |
| 영어 | 평균 2.67 / 5 |
| 코딩 | 테스트 4 / 17 (완전 통과 1/5) |
| THINKING 켬 | 답 0 / 14 (태그 미종료 이슈) |

학습 과정, 데이터셋, 문항별 질문·답변은 여기:

- [BENCHMARK-v1.md](BENCHMARK-v1.md) (한국어)  
- [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md)  
- [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md)  

아직 early-stage 입니다. 버전을 올릴 때마다 같은 문항으로 비교할 예정입니다.

---

## 7. 참고 문헌

| 논문 | 링크 |
|:-----|:-----|
| Llama 3 Herd of Models | https://ar5iv.labs.arxiv.org/html/2407.21783 |
| DeepSeek-Coder | https://arxiv.org/html/2401.14196v1 |
| Chinchilla (compute-optimal) | https://arxiv.org/abs/2203.15556 |
| StarCoder 2 | https://arxiv.org/abs/2402.19173 |

---

<div align="center">

**이어서 읽기**

[용어](GLOSSARY.md) · [아키텍처](ARCHITECTURE.md) · [후속학습](POST-TRAINING.md) · [벤치마크](BENCHMARK-v1.md)

</div>
