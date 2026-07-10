<div align="center">

# Post-Training — Aligning the Model with Human Feedback

<img src="https://img.shields.io/badge/RLHF-DPO%20%7C%20GRPO%20%7C%20RLVR-76B900?style=flat" alt="RLHF" />

[한국어](POST-TRAINING.md) &nbsp;·&nbsp; **[English](POST-TRAINING.en.md)** &nbsp;·&nbsp; [日本語](POST-TRAINING.ja.md)

[← README.en.md](README.en.md) &nbsp;|&nbsp; New to the terms? Start with [GLOSSARY.en.md](GLOSSARY.en.md)

</div>

---

This covers the loop where a human reviews what the deployed model says, collects feedback along the
lines of "this is how it should have answered," and keeps patching the model with it. It's the same basic
structure as the RLHF that ChatGPT runs on, or the RLAIF/DPO combo most open-source projects lean on —
just shrunk down to tutorial scale.

## Table of Contents

1. [The Overall Feedback Loop](#1-the-overall-feedback-loop)
2. [Feedback Collection Design](#2-feedback-collection-design)
3. [Three Post-Training Tracks](#3-three-post-training-tracks-by-difficulty)
4. [How to Pick a Track](#4-how-to-pick-a-track)
5. [Safeguards and Evaluation](#5-safeguards-and-evaluation)
6. [One Full Cycle, End to End](#6-one-full-cycle-end-to-end)

---

## 1. The Overall Feedback Loop

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

The key thing here is that this never wraps up in one shot. Collect → train → deploy → collect again,
on repeat — and every cycle folds the previous cycle's data back in before retraining.

## 2. Feedback Collection Design

### 2.1 Inference Logs (Where Collection Starts)

`infer.py` automatically logs every conversation.

```json
// logs/chat_log.jsonl — 1 line = 1 turn
{"ts": "2026-07-07T12:00:00", "user": "2+3*4=?", "thinking": "...", "answer": "165.", "ckpt": "sft_nano.pt"}
```

### 2.2 Three Types of Human Feedback (Labeling Formats)

| Type | What the human does | Data format | Used for |
|---|---|---|---|
| **① Correction** | Sees a wrong answer and writes the correct one directly — "here's how it should have answered" | `{user, bad_answer, corrected_answer, corrected_thinking}` | Correction SFT |
| **② Preference** | Given two answers to the same question, picks the better one | `{user, chosen, rejected}` | DPO / reward model |
| **③ Rating** | Scores an answer 1–5 (👍/👎 works too) | `{user, answer, score}` | Reward model, filtering |

② is the easy one to generate — sample the same prompt twice at different temperatures and you've got a
pair for free. For ①, if the human also writes out the `<THINKING>` trace alongside the corrected answer,
the model's reasoning quality improves right along with its final answers.

```json
// data/feedback.jsonl (correction)
{"user": "2 + 3 * 4 = ?", "bad_answer": "165.",
 "corrected_thinking": "Multiplication first. 3*4=12, 2+12=14.", "corrected_answer": "The answer is 14."}

// data/preference.jsonl (preference pair)
{"user": "Implement quicksort in Python.",
 "chosen":   "def quicksort(a): ...(a correct implementation)",
 "rejected": "Just use sort()."}
```

### 2.3 Labeling UI (Minimal Setup)

- Tutorial version: **CLI** — right after `infer.py` prints an answer, hit `[g]ood / [b]ad / [c]orrect / [p]refer` and it appends to the jsonl
- Scale-up version: a simple web UI (Gradio/Streamlit) with an A/B comparison screen
- Production version: combine implicit feedback (user 👍/👎) with explicit feedback (corrections from trained labelers)

## 3. Three Post-Training Tracks (by Difficulty)

### Track A — Correction SFT (Rejection-Sampling SFT; simplest, the recommended starting point)

Turn the human-written corrections straight into SFT data and retrain on them.

```
feedback.jsonl → {user, thinking: corrected_thinking, assistant: corrected_answer}
              → same format as data.py sft → train.py --mode sft --init ckpt/sft_nano.pt
```

- Reuses the existing pipeline as-is — no new code required.
- Mix the original SFT data with the correction data at a **1 : 3** ratio, weighted toward the
  corrections — this guards against catastrophic forgetting, where the model loses abilities it already had.
- For the ③ rating data, only promote 4–5 star answers to SFT data (rejection sampling).

### Track B — DPO (Direct Preference Optimization; the main track in this design)

Aligns the model directly from preference pairs (②), with no reward model in the loop. It's dramatically
simpler to implement and far more stable than RLHF, which makes it the best fit for a tutorial.

The idea is simple: raise the probability of the chosen answer, lower it for the rejected one, but don't
let the model drift too far from the original (reference) model while doing it.

```
L_DPO = -log σ( β·[ (logπ_θ(chosen) - logπ_ref(chosen))
                  - (logπ_θ(rejected) - logπ_ref(rejected)) ] )
```

```python
# add --mode dpo to train.py (skeleton)
def seq_logprob(model, ids, labels):
    """Sum of log-probabilities over the response portion of the sequence (excludes labels=-100 positions)."""
    logits, _ = model(ids[:, :-1])                      # (B,T,V) — need full logits
    logp = F.log_softmax(logits.float(), -1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    picked = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)                      # (B,)

def dpo_loss(policy, ref, batch, beta=0.1):
    # batch: chosen/rejected each (ids, labels) — same masking rule as SFT
    pc = seq_logprob(policy, *batch["chosen"])
    pr = seq_logprob(policy, *batch["rejected"])
    with torch.no_grad():                               # reference model is fixed (frozen)
        rc = seq_logprob(ref, *batch["chosen"])
        rr = seq_logprob(ref, *batch["rejected"])
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
```

| Hyperparameter | Recommended value |
|---|---|
| β (KL strength) | 0.1 (explore 0.05–0.5) |
| LR | 5e-7 ~ 1e-6 (much lower than SFT) |
| Epochs | 1–3 (watch for overfitting — check whether only the chosen probability is rising) |
| Reference model | A frozen copy of the checkpoint from the start of training (if VRAM is tight, precompute and cache the ref logprobs) |

### Track C — Reward Model + RL (RLHF/GRPO; the scale-up stage)

**Reward model (RM)**: reuse the existing GPT backbone unchanged, just swap the LM head for a scalar head.

```python
class RewardModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = GPT(cfg)                        # initialize from the SFT checkpoint
        self.backbone.lm_head = nn.Identity()
        self.v_head = nn.Linear(cfg.d_model, 1, bias=False)
    def forward(self, ids):                             # last token hidden → score
        h = self.backbone.hidden(ids)                   # (B,T,D)
        return self.v_head(h[:, -1]).squeeze(-1)        # (B,)

# training: preference-pair ranking loss
loss = -F.logsigmoid(rm(chosen) - rm(rejected)).mean()
```

**RL stage — GRPO is the pick** (simpler than PPO, and skips the value model entirely):

```
Sample G answers per prompt → score with RM → advantage relative to group mean
A_i = (r_i - mean(r)) / std(r)
→ advantage-weighted policy gradient + KL(π_θ ‖ π_ref) penalty
```

- For domains where correctness can be graded automatically — math, coding — skip the RM and use a
  rule-based reward instead (+1 for matching the answer, +1 for code that passes its tests) → this is
  RLVR. This is also exactly where `<THINKING>` reasoning quality starts visibly improving.
- Watch for reward hacking — if the RM score keeps climbing while the answers look worse to an actual
  human, stop right there.

## 4. How to Pick a Track

```
Feedback is "write the correct answer"    → Track A (Correction SFT)  ← start here
Feedback is "pick between two"            → Track B (DPO)             ← the main track
Auto-gradable (math/coding) / large scale → Track C (RM+GRPO / RLVR)  ← scale-up
Recommended real-world combo: nail the basics with A → iterate on B → add C if you have the resources
```

## 5. Safeguards and Evaluation

| Item | Design |
|---|---|
| **Forgetting prevention** | Measure loss on a fixed regression set (the original SFT validation set) after every round of post-training — roll back if it degrades by more than 5% |
| **KL guard** | Bound the distance from the reference model via DPO's β / GRPO's KL penalty |
| **Promotion gate** | A new checkpoint only ships if it (1) passes the regression set and (2) beats a >55% win rate on a held-out preference set |
| **Data hygiene** | Dedupe identical prompts, check for conflicts between correction and preference labels, verify inter-labeler agreement |
| **Version control** | Keep `ckpt/{mode}_{preset}_v{n}.pt` alongside a snapshot of the data it was trained on |

> **Implementation status**: Tracks A (mixed SFT), B (DPO), and C (RM+GRPO/RLVR) are all implemented.
> The GRPO loss is `mean_token(-A·logπ_θ + β·KL_k3(π_ref‖π_θ))`, the advantage is `A=(r-mean)/std`, and
> the RLVR reward is computed as format reward + answer match. Run it with `python rlhf.py --mode {rm,rlvr}`.

## 6. One Full Cycle, End to End

Here's how I first sketched out one post-training cycle, based on Tracks A+B. For the exact commands and
flags the real implementation uses, see [§5 How to Run It in README.en.md](README.en.md#5-how-to-run-it).

```bash
python infer.py --ckpt ckpt/sft_nano.pt --log          # 1. Deploy and collect conversation logs
python feedback.py review                              # 2. Human reviews and labels answers
python data.py sft --input data/feedback.jsonl         # 3a. Correction → SFT data
python train.py --mode sft --init ckpt/sft_nano.pt     # 4a. Train Track A
python train.py --mode dpo --init ckpt/sft_nano.pt \
                --data data/preference.jsonl           # 4b. Train Track B
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano.pt  # 5. Promotion gate
# on pass, repeat from step 1 with the new checkpoint
```

<div align="center">

---

[← ARCHITECTURE.en.md](ARCHITECTURE.en.md) &nbsp;|&nbsp; [README.en.md](README.en.md) &nbsp;|&nbsp; [GLOSSARY.en.md](GLOSSARY.en.md)

</div>
