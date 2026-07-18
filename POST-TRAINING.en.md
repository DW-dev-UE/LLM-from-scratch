[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](POST-TRAINING.md) [![English](https://img.shields.io/badge/English-0969DA?style=flat-square)](POST-TRAINING.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](POST-TRAINING.ja.md)

# POST-TRAINING

[← README](README.en.md) · Stuck on a term? [GLOSSARY](GLOSSARY.en.md)

---

If you only deploy and stop, the model freezes there.
When a person says “for this question, it should answer like this,” you can collect that and train again.

This doc is about that loop.
Think of it as a **small-scale** version of ChatGPT-side RLHF, or open-source RLAIF / DPO.

---

## 1. Overall feedback loop

```
                ┌─────────────────────────────────────────────┐
                │                                             │
   [Deploy/Inference]  →  [Collect feedback]  →  [Feedback dataset]  →  [Post-training]
   infer.py             human rates the answer   feedback.jsonl          3 tracks
   (logs saved)          ① correction ② preference  preference.jsonl     (§3 below)
                          ③ rating                                        │
                │                                             ↓
                └────────────── deploy new checkpoint ←──────────┘
                                (1 cycle = 1 iteration, repeat)
```

This is not a one-shot structure.
Collect → train → deploy → collect again, and keep stacking data from previous cycles into the next train.

---

## 2. Feedback collection design

### 2.1 Inference logs

`infer.py` logs conversations automatically.

```json
// logs/chat_log.jsonl — 1 line = 1 turn
{"ts": "2026-07-07T12:00:00", "user": "2+3*4=?", "thinking": "...", "answer": "165.", "ckpt": "sft_nano.pt"}
```

### 2.2 Three kinds of human feedback

| Type | What the human does | Data format | Used for |
|---|---|---|---|
| **① Correction** | Sees a wrong answer and writes how it should have answered | `{user, bad_answer, corrected_answer, corrected_thinking}` | Correction SFT |
| **② Preference** | Picks the better of two answers to the same question | `{user, chosen, rejected}` | DPO / reward model |
| **③ Rating** | Scores an answer 1–5 (👍/👎 works too) | `{user, answer, score}` | Reward model, filtering |

② is easy to build: sample twice with different temperatures.
For ①, writing the `<THINKING>` trace with the correction also lifts reasoning quality.

```json
// data/feedback.jsonl (correction)
{"user": "2 + 3 * 4 = ?", "bad_answer": "165.",
 "corrected_thinking": "Multiplication first. 3*4=12, 2+12=14.", "corrected_answer": "The answer is 14."}

// data/preference.jsonl (preference pair)
{"user": "Implement quicksort in Python.",
 "chosen":   "def quicksort(a): ...(a correct implementation)",
 "rejected": "Just use sort()."}
```

### 2.3 Labeling UI

- Tutorial: **CLI** — right after an answer, `[g]ood / [b]ad / [c]orrect / [p]refer` → append jsonl
- Scale-up: Gradio/Streamlit A/B screen
- Production: user 👍/👎 (implicit) + labeler corrections (explicit) together

---

## 3. Three post-training tracks (easiest first)

### Track A — Correction SFT (simplest; start here)

Turn human corrections into SFT data and retrain as-is.

```
feedback.jsonl → {user, thinking: corrected_thinking, assistant: corrected_answer}
              → same format as data.py sft → train.py --mode sft --init ckpt/sft_nano.pt
```

- Same pipeline as before. No new code required.
- Mix original SFT : correction = **1 : 3** (weight toward corrections).

> [!WARNING]
> Otherwise catastrophic forgetting eats prior ability.

- From ③ ratings, promote only 4–5 star answers into SFT (rejection sampling).

### Track B — DPO (main track in this design)

Align directly from preference pairs ② with no reward model.
Much easier to implement and more stable than full RLHF, so it fits a tutorial well.

Idea is simple: raise chosen probability, lower rejected, without drifting too far from the reference model.

```
L_DPO = -log σ( β·[ (logπ_θ(chosen) - logπ_ref(chosen))
                  - (logπ_θ(rejected) - logπ_ref(rejected)) ] )
```

```python
# add --mode dpo to train.py (skeleton)
def seq_logprob(model, ids, labels):
    """Sum of log-probs over the response span (exclude labels=-100 positions)."""
    logits, _ = model(ids[:, :-1])                      # (B,T,V)
    logp = F.log_softmax(logits.float(), -1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    picked = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)                      # (B,)

def dpo_loss(policy, ref, batch, beta=0.1):
    pc = seq_logprob(policy, *batch["chosen"])
    pr = seq_logprob(policy, *batch["rejected"])
    with torch.no_grad():                               # reference model is frozen
        rc = seq_logprob(ref, *batch["chosen"])
        rr = seq_logprob(ref, *batch["rejected"])
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
```

| Hyperparameter | Recommended |
|---|---|
| β (KL strength) | 0.1 (explore 0.05–0.5) |
| LR | 5e-7 ~ 1e-6 (much lower than SFT) |
| Epochs | 1–3 (watch overfitting — check that only chosen prob rises) |
| Reference model | Frozen copy of the start-of-training checkpoint (if VRAM is tight, pre-cache ref logprobs) |

### Track C — Reward model + RL (scale-up)

**Reward model (RM)**: reuse the GPT backbone; swap only the LM head for a scalar head.

```python
class RewardModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = GPT(cfg)                        # init from SFT checkpoint
        self.backbone.lm_head = nn.Identity()
        self.v_head = nn.Linear(cfg.d_model, 1, bias=False)
    def forward(self, ids):                             # last token hidden → score
        h = self.backbone.hidden(ids)                   # (B,T,D)
        return self.v_head(h[:, -1]).squeeze(-1)        # (B,)

# training: preference-pair ranking loss
loss = -F.logsigmoid(rm(chosen) - rm(rejected)).mean()
```

**RL stage — prefer GRPO** (simpler than PPO; no value model):

```
Sample G answers per prompt → RM scores → advantage vs group mean
A_i = (r_i - mean(r)) / std(r)
→ advantage-weighted policy gradient + KL(π_θ ‖ π_ref) penalty
```

- On math/coding where you can **auto-grade**, skip RM and use rule rewards  
  (exact match +1, tests pass +1) → **RLVR**. This is where `<THINKING>` quality often actually moves.
- Watch reward hacking: if RM score rises but humans think answers got worse, stop immediately.

---

## 4. Picking a track

```
Feedback is “write the correct answer”    → Track A (Correction SFT)  ← start here
Feedback is “pick between two”            → Track B (DPO)             ← main track
Auto-gradable (math/coding) / large scale → Track C (RM+GRPO / RLVR)  ← scale-up
Real-world combo: firm up with A → iterate B → add C if you have room
```

---

## 5. Safeguards and evaluation

| Item | Design |
|---|---|
| **Forgetting prevention** | After each post-train, measure loss on a fixed regression set — roll back if it degrades >5% |
| **KL guard** | Bound distance from reference via DPO β / GRPO KL penalty |
| **Promotion gate** | Ship only if (1) regression set passes and (2) holdout preference win-rate > 55% |
| **Data hygiene** | Dedupe same prompts, check correction↔preference label conflicts, inter-labeler agreement |
| **Version control** | Keep `ckpt/{mode}_{preset}_v{n}.pt` with a snapshot of the training data |

Tracks A (mixed SFT), B (DPO), and C (RM+GRPO/RLVR) are implemented.
GRPO loss is `mean_token(-A·logπ_θ + β·KL_k3(π_ref‖π_θ))`, advantage `A=(r-mean)/std`,  
RLVR is format reward + answer match. Run: `python rlhf.py --mode {rm,rlvr}`.

---

## 6. One full cycle end to end

Sketch for tracks A+B. Exact flags follow [README §5](README.en.md#5-how-to-run).

```bash
python infer.py --ckpt ckpt/sft_nano.pt --log          # 1. Deploy and log chats
python feedback.py review                              # 2. Human review / labels
python data.py sft --input data/feedback.jsonl         # 3a. Correction → SFT data
python train.py --mode sft --init ckpt/sft_nano.pt     # 4a. Track A
python train.py --mode dpo --init ckpt/sft_nano.pt \
                --data data/preference.jsonl           # 4b. Track B
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano.pt  # 5. Promotion gate
# on pass, repeat from step 1 with the new checkpoint
```

---

[ARCHITECTURE](ARCHITECTURE.en.md) · [README](README.en.md) · [GLOSSARY](GLOSSARY.en.md) · [BENCHMARK](BENCHMARK-v1.en.md)
