<div align="center">

# 범용 LLM 및 회사용 AI 아키텍처

<img src="https://img.shields.io/badge/CUDA-enabled-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/from--scratch-no%20external%20LLM-blueviolet?style=flat" alt="From scratch" />
<img src="https://img.shields.io/badge/status-actively%20updated-brightgreen?style=flat" alt="Actively updated" />

**[한국어](README.md)** &nbsp;·&nbsp; [English](README.en.md) &nbsp;·&nbsp; [日本語](README.ja.md)

</div>

---

> **🚧 이 저장소는 계속 업데이트 중입니다.**  
> 모델 학습·벤치마크·문서·파이프라인을 주기적으로 손보고 있다. 체크포인트 버전(v1 → v2 …),
> 벤치마크 리포트, 명령어 정리도 같이 갱신될 수 있으니 최신 커밋을 기준으로 읽어 달라.
> 이슈·PR 환영.

NVIDIA CUDA 위에서 돌아가는 범용 LLM과, 코딩·디자인처럼 회사 업무에 바로 쓸 수 있는 AI를 밑바닥부터
만드는 프로젝트다. 토크나이저부터 학습·추론 엔진, 배포 후 사람 피드백으로 모델을 고쳐나가는 파이프라인까지
전부 직접 짰다.

기존 오픈소스 LLM을 가져다 파인튜닝하는 쪽이 훨씬 빠르고 실용적이라는 건 알고 있다. 그런데도 처음부터
다시 만드는 이유는 하나다 — LLM이 안에서 어떻게 돌아가는지 직접 손으로 짜보지 않으면 결국 남의 코드를
빌려 쓰는 사람으로 남기 때문이다. 그래서 이 저장소는 외부 LLM에 기대지 않는다.

문서는 아래처럼 나눠져 있다. 순서대로 읽어도 되고, 필요한 것만 찾아봐도 된다.

| 문서 | 내용 |
|---|---|
| **README.md** (지금 보는 문서) | 설계 철학, 파라미터 전략, RAG·Tool 역할 분리, 프로젝트 구조, 실행 방법 |
| [GLOSSARY.md](GLOSSARY.md) | Transformer, weight, RoPE, DPO 같은 용어가 낯설다면 여기부터 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 모델 구조, 토크나이저, 학습·추론 엔진 — 실제 코드 포함 |
| [POST-TRAINING.md](POST-TRAINING.md) | 배포 후 사람 피드백으로 모델을 계속 고쳐나가는 순환 구조 |
| [BENCHMARK-v1.md](BENCHMARK-v1.md) | Base V1 벤치마크 리포트 (pretrain vs SFT, 언어·코딩 상세) |

실제 구현 코드는 [llm/](llm/) 폴더에 있고, 명령어는 [llm/명령어-정리.md](llm/명령어-정리.md)에
정리해 뒀다.

## 목차

1. [아키텍처 설계 과정](#1-아키텍처-설계-과정)
2. [파라미터의 개수](#2-파라미터의-개수)
3. [RAG와 Tool](#3-rag와-tool)
4. [프로젝트 구조](#4-프로젝트-구조)
5. [실행 방법](#5-실행-방법)
6. [벤치마크 (V1)](#6-벤치마크-v1)
7. [참고 문헌](#참고-문헌)

---

## 1. 아키텍처 설계 과정

처음엔 이렇게 갈 생각이었다.

1️⃣ 순수 자연어 데이터셋으로 LLM을 먼저 만든다
2️⃣ 그 위에 코딩 특화 데이터셋을 얹는다
3️⃣ 사람이 직접 강화학습으로 다듬는다

이유는 간단했다. 코딩을 아무리 잘하는 AI라도 사람은 결국 자연어로 질문을 던진다. 그러니 "사용자가
무슨 맥락에서 뭘 원하는지 알아듣는 능력"이 제일 먼저 갖춰져야 한다고 생각했다.

그런데 Llama 3 논문(*The Llama 3 Herd of Models*)을 읽어보니, 이 순서 자체가 없었다.

> https://ar5iv.labs.arxiv.org/html/2407.21783

Llama 3는 dense Transformer 구조이고, 사전학습 단계에서 next-token prediction으로 언어 구조와
세계 지식을 익힌 뒤, 후속학습 단계에서 instruction following·preference alignment·coding·
reasoning·tool-use를 강화한다. 그런데 그 사전학습 자체가 처음부터 자연어(약 50%) + 수학·추론(25%)
+ 코드(17%) + 다국어(8%)를 한 데이터 믹스에 섞어서 진행된다. "자연어를 먼저 완성하고 코드를 얹는다"는
단계가 아예 없는 셈이다.

DeepSeek-Coder는 반대쪽에서 같은 얘기를 한다. 1.3B~33B 모델을 2T 토큰으로 처음부터 학습시켰고,
repository 단위 코퍼스와 16K 컨텍스트에서의 fill-in-the-blank(FIM) 과제를 썼다. 데이터 구성은 87%
소스 코드, 10% 코드 관련 영어, 3% 코드와 무관한 중국어였다.

> https://arxiv.org/html/2401.14196v1

자연어 비중이 13%밖에 안 되는데도 당시 오픈소스 code LLM 중 제일 좋은 성능을 냈다. 두 논문 다 "자연어를
따로 먼저 완성한다"는 단계 없이 처음부터 자연어와 코드를 같이 학습시킨다.

결국 원래 세운 순서가 틀렸다는 얘기다.

**버린 순서**

```
자연어만 학습 → 코드만 학습 → instruction tuning
```

자연어 능력은 물론 필요하다. 다만 그걸 따로 떼어내 먼저 "완성"할 필요는 없었다. 그래서 순서를 바꿨다.

**바꾼 순서**

```
자연어 + 코드 + 수학 + 문서 혼합 base pretraining
  → code-heavy continued pretraining
    → instruction tuning
      → preference optimization
```

### 데이터 구성 비율 (300M ~ 1B 기준)

| 데이터 종류 | 비율 | 비고 |
|---|---|---|
| Raw source code | 35% ~ 45% | |
| Code-adjacent text | 15% ~ 20% | README, docs, issue, PR, commit |
| General technical natural language | 20% ~ 30% | 영어 + 한국어 기술 문서 |
| Math / reasoning | 5% ~ 10% | |
| Logs / tests / errors / diffs | 5% ~ 10% | |
| Korean instruction / bilingual data | 5% ~ 10% | |

---

## 2. 파라미터의 개수

파라미터는 모델의 크기다(용어가 헷갈리면 [GLOSSARY.md](GLOSSARY.md) 참고). 크면 클수록 똑똑해질
가능성은 높지만, 이 프로젝트는 300M → 1B → 3B로 천천히 키우는 쪽을 택했다.

Chinchilla 논문(*Training Compute-Optimal Large Language Models*)은 같은 compute 예산이라면
모델 크기와 학습 토큰 수를 같이 늘려야 한다고 말한다.

> https://arxiv.org/abs/2203.15556

70M~16B 이상 파라미터, 5B~500B 토큰 구간에서 400개 넘는 모델을 돌려본 결과, 모델 크기가 2배가
되면 토큰 수도 2배가 되어야 compute-optimal이라는 결론이었다. GPT-3, Gopher, Jurassic-1,
Megatron-Turing NLG 같은 당시 모델들은 실제로 크기에 비해 데이터가 부족했고, Chinchilla(70B,
1.4T 토큰)는 4배 큰 Gopher(280B, 300B 토큰)와 같은 compute로 MMLU 등에서 더 좋은 점수를 냈다.

그러니 무작정 파라미터부터 키우지 말고, 작은 모델에서 tokenizer·dataloader·loss curve·eval·
checkpoint 파이프라인이 제대로 도는지 먼저 확인하고 규모를 올리는 게 안전하다. StarCoder2가 이걸
실제로 보여주는 사례다 — 3B/7B/15B를 각각 다른 토큰 수(약 3.1T/3.5T/4.1T)로 따로 학습해서, 배포
예산에 맞게 고를 수 있는 모델 패밀리를 한 번에 내놨다. (15B 모델은 2배 넘게 큰 CodeLlama-34B와
맞먹는 성능을 냈다.)

> https://arxiv.org/abs/2402.19173

이 프로젝트가 실제로 밟는 단계는 이렇다.

| 규모 | 목적 |
|---|---|
| 10M ~ 50M | 학습 루프, tokenizer, loss 감소 확인 |
| 100M ~ 300M | 코드 completion, FIM, 기본 instruction 실험 |
| 1B | 작은 coding assistant 실험 |
| 3B | 사내 도구와 연동 가능한 실사용 후보 |
| 7B+ | 외부 서비스로 노출 가능한 최소 규모 |

---

## 3. RAG와 Tool

기업 특화 LLM이라고 해서 모든 지식을 모델 가중치에 우겨넣으려 하면 안 된다. 최신 코드, 최신 문서,
테스트 결과, 빌드 로그처럼 계속 바뀌는 정보는 RAG와 tool 호출로 연결하는 게 낫다. 새 기술이 나올
때마다 모델 전체를 재학습하는 건 감당이 안 된다.

역할은 이렇게 나눈다.

| 담당 | 항목 |
|---|---|
| **Model (가중치)** | 사고 방식, 코딩 패턴, 문맥 이해, 답변 스타일, tool 사용 능력 |
| **RAG / Tools (외부 연결)** | 최신 지식, 사내 문서, 실제 repo, 테스트 실행, 빌드 결과, 배포 환경 |

모델은 "어떻게 생각하고 어떻게 도구를 쓸지"를 맡고, RAG/Tool은 "지금 뭐가 사실인지"를 맡는다.

---

## 4. 프로젝트 구조

```
AI/
├── README.md              # 이 문서 — 설계 철학, 파라미터 전략, 프로젝트 구조, 실행 방법
├── README.en.md            # English
├── README.ja.md            # 日本語
├── GLOSSARY.md              # 용어 정리 (ko / en / ja)
├── GLOSSARY.en.md
├── GLOSSARY.ja.md
├── ARCHITECTURE.md          # 모델 아키텍처·학습·추론 (ko / en / ja)
├── ARCHITECTURE.en.md
├── ARCHITECTURE.ja.md
├── POST-TRAINING.md         # 후속 학습(인간 피드백) (ko / en / ja)
├── POST-TRAINING.en.md
├── POST-TRAINING.ja.md
├── BENCHMARK-v1.md          # Base V1 벤치마크 리포트 (ko / en / ja)
├── BENCHMARK-v1.en.md
├── BENCHMARK-v1.ja.md
└── llm/                      # 실제 구현
    ├── model.py               # 아키텍처 — RMSNorm+RoPE+SwiGLU+GQA, KV캐시, hidden()
    ├── tokenizer.py            # 토크나이저 학습/로드
    ├── data.py                # 데이터 준비 — demo / pretrain / sft·교정·혼합 / ratings / hygiene
    ├── train.py                # 학습 루프 — pretrain / sft / dpo, --tag 버전·스냅샷
    ├── infer.py                # 추론 엔진 + CLI 채팅, 대화 로그 자동 저장
    ├── feedback.py             # LLM Feedback Studio — 피드백 수집 웹 UI
    ├── reward.py               # 보상모델 — GPT 백본 + 스칼라 head, 랭킹 손실
    ├── rlhf.py                 # GRPO 정책 최적화 (RLVR 규칙보상 / RM 스칼라보상)
    ├── eval_gate.py            # 승격 게이트 — 홀드아웃 회귀 loss + win-rate
    ├── 명령어-정리.md            # 전체 파이프라인 실행 명령어 모음
    ├── tokenizer.json
    ├── data/                   # corpus, sft/preference/feedback/ratings jsonl, rated_sft.jsonl, train/val bin
    ├── ckpt/                   # pretrain/sft/dpo/rm/grpo 체크포인트 + 벤치마크 JSON
    └── logs/                   # chat_log.jsonl — 추론 대화 로그
```

---

## 5. 실행 방법

전체 명령어는 [llm/명령어-정리.md](llm/명령어-정리.md)에 단계별로 정리돼 있다. 처음부터 끝까지
한 번 돌려보려면:

```bash
cd llm
pip install torch tokenizers numpy

python data.py demo                                        # 1. 샘플 데이터 생성
python tokenizer.py --input data/corpus.txt data/sft.jsonl # 2. 토크나이저 학습
python data.py pretrain                                    # 3. 사전학습 데이터 준비
python train.py --mode pretrain --preset nano --steps 2000 # 4. 사전학습
python data.py sft                                         # 5. SFT 데이터 준비
python train.py --mode sft --init ckpt/pretrain_nano.pt --steps 300  # 6. SFT
python infer.py --ckpt ckpt/sft_nano.pt                    # 7. 채팅 (로그 자동 기록)
```

배포 후 사람 피드백으로 한 사이클을 돌리는 방법은 [POST-TRAINING.md](POST-TRAINING.md)에서 다룬다.
짧게만 보면:

```bash
python infer.py --ckpt ckpt/sft_nano.pt          # 1. 대화 → logs/chat_log.jsonl 수집
python feedback.py                               # 2. 웹 UI에서 평가·교정·선호 라벨링

# 트랙 A — 교정 SFT
python data.py sft --input data/sft.jsonl data/feedback.jsonl --weight 1 3
python train.py --mode sft --init ckpt/sft_nano.pt --steps 100 --tag v1

# 트랙 B — DPO
python train.py --mode dpo --init ckpt/sft_nano.pt --steps 100 --tag v1

# 트랙 C — 보상모델 + GRPO / RLVR
python reward.py --init ckpt/sft_nano.pt --steps 200
python rlhf.py --mode rm   --init ckpt/sft_nano.pt --rm ckpt/rm_nano.pt --data data/sft.jsonl
python rlhf.py --mode rlvr --init ckpt/sft_nano.pt --data data/sft.jsonl

# 승격 게이트 통과 시 새 체크포인트로 1번부터 반복
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano_v1.pt
```

실전에서는 `data/corpus.txt`를 실제 코퍼스(TinyStories, FineWeb-Edu 등)로, `data/sft.jsonl`을
공개 instruction/reasoning 데이터로 바꾸고 `--preset small|base`로 키우면 된다.

---

## 6. 벤치마크 (V1)

`base` 프리셋(~327M)으로 학습한 **pretrain_base_v1** 과 **sft_base_v1** 을 동일 14문항·2모드
(THINKING on/off)로 측정한 결과다. 자세한 표·해석·문항별 노트는 언어별 리포트를 보면 된다.

| 문서 | 언어 |
|---|---|
| [BENCHMARK-v1.md](BENCHMARK-v1.md) | 한국어 |
| [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md) | English |
| [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md) | 日本語 |

**한 줄 결과 (sft_base_v1, no-thinking)**

| 영역 | 결과 |
|---|---|
| 한국어 평균 | 1.33 / 5 |
| 일본어 평균 | 0.33 / 5 |
| 영어 평균 | 2.67 / 5 |
| 코딩 테스트 | 4 / 17 (완전 통과 1/5 — `is_palindrome`) |
| THINKING 모드 | **답변 0/14** (`</THINKING>` 미종료 — `--no-thinking` 권장) |

pretrain만으로는 28문항 전부 0점(지시 미학습의 교과서적 실패)이고, SFT 5,000 step 이후
“지시 시도” 단계로 올라간다. 아직 early-stage이며, 후속 버전·벤치마크는 계속 추가할 예정이다.

원본 JSON: [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json),
[`llm/ckpt/benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json)

---

## 참고 문헌

- Llama Team, AI @ Meta. *The Llama 3 Herd of Models.* — https://ar5iv.labs.arxiv.org/html/2407.21783
- Guo, D. et al. *DeepSeek-Coder: When the Large Language Model Meets Programming — The Rise of Code Intelligence.* — https://arxiv.org/html/2401.14196v1
- Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* — https://arxiv.org/abs/2203.15556
- Lozhkov, A. et al. *StarCoder 2 and The Stack v2: The Next Generation.* — https://arxiv.org/abs/2402.19173

<div align="center">

---

**다음 읽을 문서**

[📖 GLOSSARY.md](GLOSSARY.md) — 용어부터 확인하기 &nbsp;|&nbsp;
[🏗️ ARCHITECTURE.md](ARCHITECTURE.md) — 모델 구조·학습·추론 &nbsp;|&nbsp;
[🔁 POST-TRAINING.md](POST-TRAINING.md) — 인간 피드백 후속 학습 &nbsp;|&nbsp;
[📊 BENCHMARK-v1.md](BENCHMARK-v1.md) — Base V1 벤치마크

</div>
