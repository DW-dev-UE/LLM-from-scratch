[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-0969DA?style=flat-square)](POST-TRAINING.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](POST-TRAINING.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](POST-TRAINING.ja.md)

# POST-TRAINING

[← README](README.md) · 용어가 막히면 [GLOSSARY](GLOSSARY.md)

---

배포만 하고 끝내면 모델은 거기서 멈춥니다.
사람이 "이 질문엔 이렇게 답해야 한다"고 말해주면, 그걸 모아 다시 학습할 수 있습니다.

이 문서가 다루는 건 그 순환입니다.
ChatGPT 쪽 RLHF, 오픈소스 쪽 RLAIF·DPO 같은 걸 **작은 규모로** 옮겨 둔 버전이라고 보면 됩니다.

---

## 1. 전체 순환 구조

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

한 번에 끝내는 구조가 아닙니다.
수집 → 학습 → 배포 → 재수집을 반복하고, 이전 사이클 데이터도 누적해서 다시 학습합니다.

---

## 2. 피드백 수집 설계

### 2.1 추론 로그

`infer.py`가 대화를 자동으로 기록합니다.

```json
// logs/chat_log.jsonl — 1줄 = 1턴
{"ts": "2026-07-07T12:00:00", "user": "2+3*4=?", "thinking": "...", "answer": "165.", "ckpt": "sft_nano.pt"}
```

### 2.2 인간 피드백 3종

| 유형 | 사람이 하는 일 | 데이터 형식 | 사용처 |
|---|---|---|---|
| **① 교정 (Correction)** | 틀린 답을 보고 "이렇게 답해야 한다"고 정답을 직접 작성 | `{user, bad_answer, corrected_answer, corrected_thinking}` | 교정 SFT |
| **② 선호 (Preference)** | 같은 질문에 대한 답 2개 중 나은 쪽 선택 | `{user, chosen, rejected}` | DPO / 보상모델 |
| **③ 점수 (Rating)** | 답에 1~5점 (👍/👎도 가능) | `{user, answer, score}` | 보상모델, 필터링 |

②는 temperature만 바꿔 2번 생성하면 쌍을 만들기 쉽습니다.
① 교정에 `<THINKING>` 사고 과정까지 같이 쓰면 사고 품질도 같이 올라갑니다.

```json
// data/feedback.jsonl (교정)
{"user": "2 + 3 * 4 = ?", "bad_answer": "165.",
 "corrected_thinking": "곱셈 우선. 3*4=12, 2+12=14.", "corrected_answer": "답은 14입니다."}

// data/preference.jsonl (선호쌍)
{"user": "퀵소트를 파이썬으로 구현해줘.",
 "chosen":   "def quicksort(a): ...(정확한 구현)",
 "rejected": "sort()를 쓰세요."}
```

### 2.3 라벨링 UI

- 튜토리얼: **CLI** — 답 직후 `[g]ood / [b]ad / [c]orrect / [p]refer` → jsonl append
- 확장: Gradio/Streamlit 으로 A/B 화면
- 실서비스: 사용자 👍/👎(암묵) + 라벨러 교정(명시)를 같이 씀

---

## 3. 후속 학습 3트랙 (쉬운 순)

### 트랙 A — 교정 SFT (가장 단순, 여기서 시작)

인간이 쓴 교정 답을 그대로 SFT 데이터로 바꿔 재학습합니다.

```
feedback.jsonl → {user, thinking: corrected_thinking, assistant: corrected_answer}
              → data.py sft 형식과 동일 → train.py --mode sft --init ckpt/sft_nano.pt
```

- 기존 파이프라인 그대로. 새 코드가 필요 없습니다.
- 기존 SFT : 교정 = **1 : 3** 으로 섞습니다 (교정 쪽 가중).

> [!WARNING]
> 안 그러면 기존 능력을 잊어버리는 catastrophic forgetting 이 납니다.

- 점수 ③ 중 4~5점만 골라 SFT 로 승격 (rejection sampling).

### 트랙 B — DPO (이 설계의 메인)

선호쌍 ②로 보상모델 없이 바로 정렬합니다.
RLHF보다 구현·안정성이 훨씬 쉬워서 튜토리얼에 잘 맞습니다.

원리는 간단합니다. chosen 확률은 올리고 rejected 는 낮추되, 참조 모델에서 너무 멀어지지 않게 합니다.

```
L_DPO = -log σ( β·[ (logπ_θ(chosen) - logπ_ref(chosen))
                  - (logπ_θ(rejected) - logπ_ref(rejected)) ] )
```

```python
# train.py에 --mode dpo 추가 (스켈레톤)
def seq_logprob(model, ids, labels):
    """시퀀스의 응답 부분 로그확률 합 (labels=-100 위치 제외)."""
    logits, _ = model(ids[:, :-1])                      # (B,T,V)
    logp = F.log_softmax(logits.float(), -1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    picked = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)                      # (B,)

def dpo_loss(policy, ref, batch, beta=0.1):
    pc = seq_logprob(policy, *batch["chosen"])
    pr = seq_logprob(policy, *batch["rejected"])
    with torch.no_grad():                               # 참조 모델은 고정
        rc = seq_logprob(ref, *batch["chosen"])
        rr = seq_logprob(ref, *batch["rejected"])
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
```

| 하이퍼파라미터 | 권장값 |
|---|---|
| β (KL 강도) | 0.1 (0.05~0.5 탐색) |
| LR | 5e-7 ~ 1e-6 (SFT보다 훨씬 낮게) |
| Epoch | 1~3 (과적합 주의 — chosen 확률만 오르는지 모니터링) |
| 참조 모델 | 학습 시작 시점 체크포인트 사본 (VRAM 부족 시 ref logprob 사전 캐싱) |

### 트랙 C — 보상모델 + RL (확장)

**보상모델(RM)**: GPT 백본을 쓰고 LM head만 스칼라 head로 바꿉니다.

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

**RL 단계 — GRPO 권장** (PPO보다 단순, value 모델 불필요):

```
프롬프트당 G개 답변 샘플링 → RM 점수 → 그룹 평균 대비 advantage
A_i = (r_i - mean(r)) / std(r)
→ advantage 가중 policy gradient + KL(π_θ ‖ π_ref) 페널티
```

- 수학·코딩처럼 **정답을 자동 채점**할 수 있으면 RM 대신 규칙 보상  
  (정답 일치 +1, 테스트 통과 +1) → **RLVR**. `<THINKING>` 품질이 실제로 올라가는 구간이 여기입니다.
- 보상 해킹 감시: RM 점수는 오르는데 사람이 보기엔 나빠지면 바로 중단.

---

## 4. 트랙 선택

```
피드백이 "정답 작성" 형태          → 트랙 A (교정 SFT)          ← 여기서 시작
피드백이 "둘 중 선택" 형태         → 트랙 B (DPO)               ← 메인
자동 채점 가능(수학·코딩) / 대규모  → 트랙 C (RM+GRPO / RLVR)    ← 확장
실전 권장: A로 바닥 다지기 → B 반복 → (여력 되면) C
```

---

## 5. 안전장치 및 평가

| 항목 | 설계 |
|---|---|
| **망각 방지** | 후속 학습마다 고정 회귀 셋 loss 측정 — 5% 이상 악화면 롤백 |
| **KL 가드** | DPO의 β / GRPO의 KL 페널티로 참조 모델과의 거리 제한 |
| **승격 게이트** | (1) 회귀 셋 통과 (2) 홀드아웃 선호쌍 win-rate > 55% 일 때만 배포 |
| **데이터 위생** | 동일 프롬프트 중복 제거, 교정↔선호 라벨 충돌 검수, 라벨러 일치도 |
| **버전 관리** | `ckpt/{mode}_{preset}_v{n}.pt` + 학습 데이터 스냅샷을 같이 보관 |

트랙 A(혼합 SFT)·B(DPO)·C(RM+GRPO/RLVR) 는 구현돼 있습니다.
GRPO 손실은 `mean_token(-A·logπ_θ + β·KL_k3(π_ref‖π_θ))`, advantage 는 `A=(r-mean)/std`,  
RLVR 은 형식 보상 + 정답 일치. 실행: `python rlhf.py --mode {rm,rlvr}`.

---

## 6. 1사이클 실행 흐름

트랙 A+B 기준 스케치입니다. 정확한 플래그는 [README §5](README.md#5-실행-방법) 을 따릅니다.

```bash
python infer.py --ckpt ckpt/sft_nano.pt --log          # 1. 배포·대화 로그
python feedback.py review                              # 2. 사람이 검토·라벨링
python data.py sft --input data/feedback.jsonl         # 3a. 교정 → SFT 데이터
python train.py --mode sft --init ckpt/sft_nano.pt     # 4a. 트랙 A
python train.py --mode dpo --init ckpt/sft_nano.pt \
                --data data/preference.jsonl           # 4b. 트랙 B
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano.pt  # 5. 승격 게이트
# 통과 시 새 체크포인트로 1번부터 반복
```

---

[ARCHITECTURE](ARCHITECTURE.md) · [README](README.md) · [GLOSSARY](GLOSSARY.md) · [BENCHMARK](BENCHMARK-v1.md)
