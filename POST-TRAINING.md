<div align="center">

# 후속 학습(Post-Training) — 인간 피드백 기반 정렬

<img src="https://img.shields.io/badge/RLHF-DPO%20%7C%20GRPO%20%7C%20RLVR-76B900?style=flat" alt="RLHF" />

**[한국어](POST-TRAINING.md)** &nbsp;·&nbsp; [English](POST-TRAINING.en.md) &nbsp;·&nbsp; [日本語](POST-TRAINING.ja.md)

[← README.md](README.md) &nbsp;|&nbsp; 용어가 낯설면 [GLOSSARY.md](GLOSSARY.md)부터

</div>

---

배포된 모델의 답변을 사람이 검토하고 "이 질문엔 이렇게 답해야 한다"는 피드백을 모아서, 모델을
계속 고쳐나가는 순환 파이프라인을 다룬다. ChatGPT가 쓰는 RLHF, 오픈소스 계열이 주로 쓰는 RLAIF·DPO와
같은 구조를 튜토리얼 규모로 축소했다.

## 목차

1. [전체 순환 구조](#1-전체-순환-구조-feedback-loop)
2. [피드백 수집 설계](#2-피드백-수집-설계)
3. [후속 학습 3트랙](#3-후속-학습-3트랙-난이도-순)
4. [트랙 선택 가이드](#4-트랙-선택-가이드)
5. [안전장치 및 평가](#5-안전장치-및-평가)
6. [1사이클 실행 흐름](#6-1사이클-실행-흐름)

---

## 1. 전체 순환 구조 (Feedback Loop)

```
                ┌─────────────────────────────────────────────┐
                │                                             │
   [배포/추론]  →  [피드백 수집]  →  [피드백 데이터셋]  →  [후속 학습]
   infer.py       사람이 답변 평가    feedback.jsonl        3가지 트랙
   (로그 저장)    ① 교정  ② 선호     preference.jsonl      (아래 §3)
                  ③ 점수                                     │
                │                                             ↓
                └────────────── 새 체크포인트 배포 ←──────────┘
                                (1 사이클 = 1 iteration, 반복)
```

여기서 중요한 건 한 번에 끝내지 않는다는 것이다. 수집 → 학습 → 배포 → 재수집을 계속 반복하고,
매 사이클마다 이전 사이클 데이터를 누적해서 다시 학습한다.

## 2. 피드백 수집 설계

### 2.1 추론 로그 (수집의 출발점)

`infer.py`가 모든 대화를 자동으로 기록한다.

```json
// logs/chat_log.jsonl — 1줄 = 1턴
{"ts": "2026-07-07T12:00:00", "user": "2+3*4=?", "thinking": "...", "answer": "165.", "ckpt": "sft_nano.pt"}
```

### 2.2 인간 피드백 3종 (라벨링 형식)

| 유형 | 사람이 하는 일 | 데이터 형식 | 사용처 |
|---|---|---|---|
| **① 교정 (Correction)** | 틀린 답을 보고 "이렇게 답해야 한다"고 정답을 직접 작성 | `{user, bad_answer, corrected_answer, corrected_thinking}` | 교정 SFT |
| **② 선호 (Preference)** | 같은 질문에 대한 답변 2개 중 더 나은 쪽 선택 | `{user, chosen, rejected}` | DPO / 보상모델 |
| **③ 점수 (Rating)** | 답변에 1~5점 부여 (👍/👎도 가능) | `{user, answer, score}` | 보상모델, 필터링 |

②는 같은 프롬프트를 temperature만 다르게 해서 2회 생성시키면 쌍을 만들기가 쉽다. ①의 교정 답변은
`<THINKING>` 사고 과정까지 같이 써주면 사고 품질도 함께 좋아진다.

```json
// data/feedback.jsonl (교정)
{"user": "2 + 3 * 4 = ?", "bad_answer": "165.",
 "corrected_thinking": "곱셈 우선. 3*4=12, 2+12=14.", "corrected_answer": "답은 14입니다."}

// data/preference.jsonl (선호쌍)
{"user": "퀵소트를 파이썬으로 구현해줘.",
 "chosen":   "def quicksort(a): ...(정확한 구현)",
 "rejected": "sort()를 쓰세요."}
```

### 2.3 라벨링 UI (최소 구성)

- 튜토리얼: **CLI** — `infer.py` 답변 직후 `[g]ood / [b]ad / [c]orrect / [p]refer` 키 입력 → jsonl에 append
- 확장: 간단한 웹 UI(Gradio/Streamlit)로 A/B 비교 화면 제공
- 실서비스 구조: 사용자 👍/👎(암묵 피드백) + 전문 라벨러의 교정(명시 피드백)을 같이 쓴다

## 3. 후속 학습 3트랙 (난이도 순)

### 트랙 A — 교정 SFT (Rejection-sampling SFT, 가장 간단·권장 시작점)

인간이 작성한 교정 답변을 그대로 SFT 데이터로 바꿔서 재학습한다.

```
feedback.jsonl → {user, thinking: corrected_thinking, assistant: corrected_answer}
              → data.py sft 형식과 동일 → train.py --mode sft --init ckpt/sft_nano.pt
```

- 기존 파이프라인을 그대로 재사용한다. 새 코드가 필요 없다.
- 기존 SFT 데이터와 교정 데이터를 **1 : 3** 비율로 섞는다(교정 쪽에 가중치) — 기존 능력을 잊어버리는
  현상(catastrophic forgetting)을 막기 위해서다.
- 점수 ③ 데이터는 4~5점 답변만 골라 SFT 데이터로 승격한다 (rejection sampling).

### 트랙 B — DPO (Direct Preference Optimization, 본 설계의 메인)

선호쌍 ②로 보상모델 없이 바로 정렬한다. RLHF보다 구현·안정성이 압도적으로 쉬워서 튜토리얼에 제일
잘 맞는다.

원리는 간단하다. chosen 확률은 높이고 rejected 확률은 낮추되, 원래(참조) 모델에서 너무 멀어지지는
않게 한다.

```
L_DPO = -log σ( β·[ (logπ_θ(chosen) - logπ_ref(chosen))
                  - (logπ_θ(rejected) - logπ_ref(rejected)) ] )
```

```python
# train.py에 --mode dpo 추가 (스켈레톤)
def seq_logprob(model, ids, labels):
    """시퀀스의 응답 부분 로그확률 합 (labels=-100 위치 제외)."""
    logits, _ = model(ids[:, :-1])                      # (B,T,V) — 전체 logits 필요
    logp = F.log_softmax(logits.float(), -1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    picked = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)                      # (B,)

def dpo_loss(policy, ref, batch, beta=0.1):
    # batch: chosen/rejected 각각 (ids, labels) — SFT와 동일한 마스킹 규칙
    pc = seq_logprob(policy, *batch["chosen"])
    pr = seq_logprob(policy, *batch["rejected"])
    with torch.no_grad():                               # 참조 모델은 고정(frozen)
        rc = seq_logprob(ref, *batch["chosen"])
        rr = seq_logprob(ref, *batch["rejected"])
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
```

| 하이퍼파라미터 | 권장값 |
|---|---|
| β (KL 강도) | 0.1 (0.05~0.5 탐색) |
| LR | 5e-7 ~ 1e-6 (SFT보다 훨씬 낮게) |
| Epoch | 1~3 (과적합 주의 — chosen 확률만 오르는지 모니터링) |
| 참조 모델 | 학습 시작 시점 체크포인트 사본 (VRAM 부족 시 사전에 ref logprob 캐싱) |

### 트랙 C — 보상모델 + RL (RLHF/GRPO, 확장 단계)

**보상모델(RM)**: 기존 GPT 백본을 그대로 쓰고, LM head만 스칼라 head로 바꾼다.

```python
class RewardModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = GPT(cfg)                        # SFT 체크포인트로 초기화
        self.backbone.lm_head = nn.Identity()
        self.v_head = nn.Linear(cfg.d_model, 1, bias=False)
    def forward(self, ids):                             # 마지막 토큰 hidden → 점수
        h = self.backbone.hidden(ids)                   # (B,T,D)
        return self.v_head(h[:, -1]).squeeze(-1)        # (B,)

# 학습: 선호쌍 랭킹 손실
loss = -F.logsigmoid(rm(chosen) - rm(rejected)).mean()
```

**RL 단계 — GRPO 권장** (PPO보다 단순하고, value 모델이 필요 없다):

```
프롬프트당 G개 답변 샘플링 → RM 점수 → 그룹 평균 대비 advantage
A_i = (r_i - mean(r)) / std(r)
→ advantage 가중 policy gradient + KL(π_θ ‖ π_ref) 페널티
```

- 수학·코딩처럼 **정답을 자동으로 채점할 수 있는 도메인**은 RM 대신 규칙 기반 보상
  (정답 일치 +1, 코드는 테스트 통과 +1)을 쓴다 → RLVR. `<THINKING>` 사고 품질이 실제로 좋아지는
  구간이 바로 여기다.
- 보상 해킹(reward hacking) 감시가 필요하다 — RM 점수는 오르는데 사람이 보기엔 나빠지면 바로 중단.

## 4. 트랙 선택 가이드

```
피드백이 "정답 작성" 형태          → 트랙 A (교정 SFT)          ← 여기서 시작
피드백이 "둘 중 선택" 형태         → 트랙 B (DPO)               ← 메인
자동 채점 가능(수학·코딩) / 대규모  → 트랙 C (RM+GRPO / RLVR)    ← 확장
실전 권장 조합: A로 바닥 다지기 → B 반복 → (여력 되면) C
```

## 5. 안전장치 및 평가

| 항목 | 설계 |
|---|---|
| **망각 방지** | 후속 학습마다 고정 회귀 셋(기존 SFT 검증셋) loss 측정 — 5% 이상 악화되면 롤백 |
| **KL 가드** | DPO의 β / GRPO의 KL 페널티로 참조 모델과의 거리를 제한 |
| **승격 게이트** | 새 체크포인트는 (1) 회귀 셋 통과 (2) 홀드아웃 선호쌍 win-rate > 55%일 때만 배포 |
| **데이터 위생** | 동일 프롬프트 중복 제거, 교정↔선호 라벨 충돌 검수, 라벨러 간 일치도 확인 |
| **버전 관리** | `ckpt/{mode}_{preset}_v{n}.pt` + 학습에 쓴 데이터 스냅샷을 같이 보관 |

> **구현 상태**: 트랙 A(혼합 SFT)·B(DPO)·C(RM+GRPO/RLVR) 전부 구현돼 있다. GRPO 손실은
> `mean_token(-A·logπ_θ + β·KL_k3(π_ref‖π_θ))`, advantage는 `A=(r-mean)/std`, RLVR 보상은
> 형식 보상 + 정답 일치로 계산한다. `python rlhf.py --mode {rm,rlvr}`로 실행한다.

## 6. 1사이클 실행 흐름

트랙 A+B 기준으로, 후속 학습 한 사이클을 처음에 이렇게 스케치했다. 실제 구현에서 쓰는 정확한
명령어·플래그는 [README.md의 §5 실행 방법](README.md#5-실행-방법)을 따른다.

```bash
python infer.py --ckpt ckpt/sft_nano.pt --log          # 1. 배포·대화 로그 수집
python feedback.py review                              # 2. 사람이 답변 검토·라벨링
python data.py sft --input data/feedback.jsonl         # 3a. 교정 → SFT 데이터
python train.py --mode sft --init ckpt/sft_nano.pt     # 4a. 트랙 A 학습
python train.py --mode dpo --init ckpt/sft_nano.pt \
                --data data/preference.jsonl           # 4b. 트랙 B 학습
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano.pt  # 5. 승격 게이트
# 통과 시 새 체크포인트로 1번부터 반복
```

<div align="center">

---

[← ARCHITECTURE.md](ARCHITECTURE.md) &nbsp;|&nbsp; [README.md](README.md) &nbsp;|&nbsp; [GLOSSARY.md](GLOSSARY.md)

</div>
