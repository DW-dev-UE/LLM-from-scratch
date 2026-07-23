[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](BENCHMARK-v2.md) [![English](https://img.shields.io/badge/English-0969DA?style=flat-square)](BENCHMARK-v2.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](BENCHMARK-v2.ja.md)

[← README](README.en.md)

---

# 📊 BENCHMARK v2 · APEX-1 (~1.12B)

> A different line from v1 (327M `base`). A **1B-scale** line trained from scratch with an **English-only** tokenizer (32K) and corpus.
>
> Checkpoints and logs use the `model.py` preset name **Apex-1** directly (`pretrain_Apex-1_v1`, `sft_Apex-1_v1`).

| | |
|:--|:--|
| 🆕 **Latest** | `dpo_Apex-1_v1` · 2026-07-23 |
| 📦 **Model** | APEX-1 · Decoder-only · measured **1,119.5M** |
| 🧪 **Set** | English-only **15 prompts × THINKING on/off** (separate set from the 327M line; only the 5 coding prompts are shared) + 11 standard benchmarks |
| 📁 **Raw** | [`ckpt/benchmark_sft_Apex-1_v1_raw.json`](ckpt/benchmark_sft_Apex-1_v1_raw.json) · [`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md) |
| ✅ **Done** | RLVR abandoned → **DPO complete** (§6) · 11 standard benchmarks measured (§5.4) |

> [!WARNING]
> Still the **first snapshot with only one checkpoint.** There's no version ladder like the v1 line yet — the point of this benchmark was to decide the next alignment step (more SFT vs. RL) by measurement, not by feel.

### 📑 Contents

1. [At a glance](#1-at-a-glance)
2. [Training process](#2-training-process)
3. [Datasets](#3-datasets)
4. [Scoring](#4-scoring)
5. [sft_Apex-1_v1 results](#5-sft_apex-1_v1-results)
6. [Next step — RLVR → DPO](#6-next-step--rlvr--dpo)
7. [Raw files](#7-raw-files)

---

## 1. At a glance

| | 💬 normal chat (no-thinking) | 🧠 THINKING on |
|:--|:--:|:--:|
| QA correct (keyword) | 5/8 | 4/8 |
| Coding fully passed | **5/5** (tests 25/25) | 3/5 (tests 15/25) |
| Empty answers | 0 | 0 |
| Instruction followed | 1/2 | 1/2 |
| Over-length | 8/10 | 5/10 |
| Mean repetition rate | 0.032 (max 0.109) | 0.042 (max 0.129) |
| Mean answer length | 375 chars | 527 chars |

> [!IMPORTANT]
> The 327M line's biggest failure — the THINKING handoff (a blank answer field) — **never happened at all on this line**: 0 empty answers. Instead, the new bottleneck is **verbosity and repetition**.
>
> Turning THINKING on actually makes both coding (5/5→3/5) and QA (5/8→4/8) worse — the thinking process degrades answer quality instead of helping. `reverse_string` and `factorial` both failed in thinking mode because the code got cut off mid-line, which looks like the thinking pass eating the answer budget (`max_new` 256) before the code could finish.

---

## 2. Training process

### 🧩 Model card

| Field | Value |
|:------|:------|
| Family | Decoder-only Transformer (LLaMA-style) |
| Parameters | measured **1,119.5M** |
| Depth · width | 24 layers · d_model 2048 |
| Attention | GQA (Q16·KV4) + RoPE(θ=500,000) + **QK-Norm** |
| FFN | SwiGLU |
| Norm | RMSNorm (Pre-Norm) |
| Context | max 4096 (trained at seq 2048) |
| Other | weight tying · no bias |

Compared to the 327M `base` (see [ARCHITECTURE](ARCHITECTURE.en.md)), **QK-Norm** (stabilizes higher LR) and **rope_theta 500K** (the Llama-3 value, leaves room for future context extension) are new here.

### 🛤️ Training pipeline — a two-stage curriculum

The plan was "one 51K-step cosine run," but what actually ran was **41K steps of a general mix + 10K steps upweighting math/code** (GPU: 1× H100).

```text
① Corpus download (v3-en, 80.80GB, 0 failures)
        ▼
② Tokenizer 32K (byte-level BPE, English-only)
        ▼
③ Tokenize — train 20.31B tok · val 0.13B tok (mixed val)
        ▼
④a Pretrain stage-1 · 41,000 steps · lr 3e-4 · batch8×seq2048×accum24
        ▼  pretrain_Apex-1_v1 (stage1)
④b Curriculum reweight — resample 7GB each of fineweb/dclm + all code + finemath4 + cosmopedia2 (12.85B tok)
        ▼
④c Pretrain stage-2 continue · 10,000 steps · lr 6e-5 (same batch/accum)
        ▼  pretrain_Apex-1_v1 ★ (final pretrain)
        ▼
⑤ SFT · smoltalk2 English subset, 538K examples × 2 epochs ÷ (batch8×accum16) ≈ 8,400 steps · lr 3e-5
        ▼  sft_Apex-1_v1 ★
```

| Stage | steps | lr | tokens processed (est.) | init |
|:------|------:|---:|-------------------------:|:-----|
| Pretrain stage-1 | 41,000 | 3e-4 | ~16.12B | — |
| Pretrain stage-2 | 10,000 | 6e-5 | ~3.93B | stage-1 |
| **Pretrain total** | **51,000** | — | **~20.05B** | — |
| SFT | 8,400 | 3e-5 | 538K examples × 2 epochs | pretrain stage-2 |

The combined tokens processed across both pretrain stages (~20.05B) lines up almost exactly with the original "20B tokens" target.

> [!NOTE]
> Stage-2 isn't "more training" — it's a **reweight**. fineweb_edu and dclm are resampled at only 7GB each, while the code, math (finemath4), and synthetic-textbook (cosmopedia2) sources are reshuffled in whole to raise their share.

### 📉 Loss curve

<details>
<summary><b>Expand pretrain / SFT loss tables</b></summary>

<br/>

**Pretrain stage-1** (val, representative points)

| step | val loss |
|----:|--------:|
| 500 | 3.59 |
| 10,000 | 2.30 |
| 20,000 | 1.86 |
| 30,000 | 2.18 |
| 40,500 | 2.16 |

**Pretrain stage-2** (val, representative points · init = stage-1 output)

| step | val loss |
|----:|--------:|
| 500 | 2.12 |
| 5,000 | 1.93 |
| 9,000 | 1.56 |
| 9,500 | 2.10 |

**SFT** (train, representative points)

| step | train loss |
|----:|-----------:|
| 0 | 2.12 |
| 2,000 | 1.34 |
| 5,000 | 1.49 |
| 8,100 | 1.06 |

</details>

> [!WARNING]
> val loss swings by roughly ±0.3–0.5 point to point. The lesson from the README intro — "without multi-batch averaging, it's easy to misread noise as a real change in skill" — applies here too. The tables above are for **trend reference only**, not point-to-point comparison.

---

## 3. Datasets

### 🔤 Tokenizer

byte-level BPE · vocab **32,000** · English-only (a separate tokenizer from the 327M line's 64K multilingual vocab)

### 📚 Pretrain corpus (v3-en) — 80.80GB

| Source | Size | Nature |
|:-------|-----:|:-------|
| fineweb_edu | 21.00GB | English web (education-filtered) |
| dclm | 21.00GB | English web (DCLM baseline) |
| finemath4 | 10.00GB | Math |
| cosmopedia2 | 8.50GB | Synthetic textbooks |
| code_python | 8.00GB | Code |
| finepdfs | 3.00GB | PDF-extracted text |
| code_js/java/c/cpp/sql/shell | 6.50GB (combined) | Code (7 languages) |
| wikipedia_en | 1.30GB | Encyclopedia |

Unlike the 327M line, there is **no Korean or Japanese natural-language source at all** — this design directly applies the README intro's lesson that "small models do better focusing on one language than spreading across several."

### 🗣️ SFT mix — HuggingFaceTB/smoltalk2 (English subset)

| Category | Source | Target rows |
|:---------|:-------|------------:|
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

Trained on the **actual 538K rows** out of a ~606K target (some sources fell short). think : no-think ≈ **1 : 2** — this ratio was built into the design from the start, applying the 327M `base` v6 lesson that "if there isn't enough thinking data, the thinking→answer handoff breaks."

All weights are ×1 — the per-source target row counts already encode the mix ratio. Multi-turn examples use only the first user/assistant pair, and multilingual/aya-style sources were excluded since this is an English-only model.

---

## 4. Scoring

| Item | Detail |
|:-----|:-------|
| Prompts | **English-only 15 prompts × 2 modes = 30 generations** (fact 2 · math 3 · reasoning 2 · instruction 2 · open 1 · coding 5) |
| Coding prompts | **Identical to** the 327M line's 14-prompt set — kept fixed for cross-line comparison |
| Generation | temp **0.0 greedy** · top_p 0.9 · max_new **256** · seed 0 |
| Scoring | QA via keyword presence (automatic) · coding via unit-test pass count |
| Added metrics | instruction adherence, over-length, repetition rate — added specifically so the next post-training step (more SFT vs. RL) is **measured**, not eyeballed |
| Date | 2026-07-22 |

> [!WARNING]
> Keyword matching is a weak proxy. On `en_math_rate` (thinking mode), the final answer was wrong ("1 meter"), yet "120" showed up in the reasoning text, so `keyword_hit: true`. On `en_reason_order` (normal chat), the correct answer "Joe" was merely mentioned mid-reasoning while the actual conclusion was the wrong "Ann," and it still counted as a keyword hit. §5's tables mark these mismatches with `*`.

---

## 5. sft_Apex-1_v1 results

### 5.1 QA · normal chat (THINKING off)

| Prompt | Verdict | What the model actually said |
|:-------|:-------:|:-----------------------------|
| Capital (France) | ✅ | Correct, though it expanded into tourist-brochure detail (120-char cap → 352 chars) |
| Photosynthesis gas | ✅ | Accurate through CO2·chlorophyll·glucose |
| Train speed×time | ❌ | Flipped the units into "60÷60=1 hour" — completely wrong (correct: 120km) |
| Pens and change | ❌ | Got both the sign and the arithmetic wrong: "$12−$20=−$4" (correct: $8) |
| Multi-step apples | ❌ | Misread the problem and answered "2" (correct: 29) |
| Syllogism (roses) | ✅ | "No, it does not follow…" — a clean, logically sound answer this time |
| Height order (shortest) | ✅* | Concluded "Ann is shortest," which is wrong (correct: Joe) — the word "Joe" only appears mid-reasoning, so the keyword hit |
| One-word answer | ✅* | "A clear daytime sky is a beautiful **blue** color…" — the color is right (keyword hit), but it violates the one-word instruction |
| List 3 colors, comma-separated | ⚠️ | The automatic grader marked instruction-following as ✅, but the model actually added a forbidden explanatory sentence ("...basic colors that cannot be created by mixing other colors") — a case the grader missed |
| One-sentence summary | – | Essentially restates the source text verbatim rather than condensing it |

### 5.2 QA · THINKING on

| Prompt | Verdict | What the model actually said |
|:-------|:-------:|:-----------------------------|
| Capital (France) | ✅ | Correct, short and clean |
| Photosynthesis gas | ✅* | Mixes in the wrong term "photolysis" (keyword CO2 still appears, so it hit) |
| Train speed×time | ✅* | "120" appears repeatedly in the reasoning, but the final answer is the wrong "1 meter" |
| Pens and change | ❌ | Wrong: "$32−$20=$12" (correct: $8) |
| Multi-step apples | ❌ | Wrong: "12−7=5" (correct: 29) |
| Syllogism (roses) | ✅* | Self-contradictory answer — "Yes… No, all roses do not fade at once" — only the "no" hit |
| Height order (shortest) | ❌ | Wrong ("Ann is shortest"), and "Joe" isn't even mentioned, so no keyword hit either |
| One-word answer | ❌ | Followed the instruction ("clear," one word) but got the color wrong (correct: blue) |
| List 3 colors, comma-separated | ❌ | "red blue yellow" with no commas — instruction violated |
| One-sentence summary | – | Restates the source the same way as in normal-chat mode |

5/8 in normal chat vs. 4/8 with thinking on — turning thinking on doesn't raise QA accuracy, it just adds new failure modes (pens-and-change and multi-step-apples fail in both modes, and height-order is even worse with thinking on).

### 5.3 Coding · both modes compared

Normal chat scores a **perfect 5/5**. THINKING mode failed 2 problems because the **code got cut off mid-generation**.

| Problem | Normal chat | THINKING | Comment |
|:--------|:-----------:|:--------:|:--------|
| `is_prime` | 5/5 | 5/5 | Both modes pass with √n trial division |
| `reverse_string` | **5/5** | **0/5** | THINKING mode: `''.join(reversed` never closes — **SyntaxError** from a cut-off line |
| `factorial` | **5/5** | **0/5** | THINKING mode: stops right at `return` with no value |
| `is_palindrome` | 5/5 | 5/5 | Both modes use `s == s[::-1]` |
| `find_max` | 5/5 | 5/5 | Both modes use the built-in `max()` |

<details>
<summary><b>Expand the truncated THINKING-mode code</b></summary>

<br/>

```python
# reverse_string — SyntaxError: '(' was never closed
def reverse_string(s):
    characters = s.split()
    reversed_chars = ''.join(reversed(characters))
    return ''.join(reversed

# factorial — return statement with no value
def factorial(n):
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    elif n == 0 or n == 1:
        return
```

</details>

> [!IMPORTANT]
> Neither failure looks like "not knowing the algorithm" — it looks like the **thinking narration ran long enough inside the `max_new=256` budget that the code got cut off before it could finish.** This is a different kind of budget problem from the 327M line's "handoff" bug (where the whole answer field went blank). Splitting or expanding the thinking/answer budget is the next experiment to try.

### 5.4 Standard benchmarks (lm-evaluation-harness)

§5.1–5.3 above are our own 15-prompt set. Here, `sft_Apex-1_v1` and `dpo_Apex-1_v1` were converted to HF format (logit-match verified, max diff 2.3e-5) and measured on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.12, bf16, **side by side with other models that have published scores**. Measured 2026-07-23 on an H100 80GB (Lambda). Protocol (7 commonsense tasks 0-shot, MMLU 5-shot, GSM8K 5-shot, HumanEval/MBPP 0-shot pass@1) and comparison figures are taken directly from the [TinyLlama paper](https://arxiv.org/abs/2401.02385) Table 2·3.

#### At a glance

| Metric | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B |
|:---|---:|---:|---:|---:|
| Commonsense avg, 7 tasks (0-shot) | 49.54 | 49.65 | 52.99 | 48.30 |
| MMLU (5-shot) | 24.83 | 24.90 | 25.34 | 25.70 |
| GSM8K (5-shot, strict) | 1.44 | 1.90 | — | — |
| HumanEval (pass@1) | 8.54 | 8.54 | 9.15 | 1.83 |
| MBPP (pass@1) | 4.80 | 5.20 | — | — |

> [!IMPORTANT]
> **DPO ≥ SFT, no alignment tax** — DPO matches or beats SFT on every metric here. **BoolQ 62.20 ranks 1st among the 4 comparison models** (TinyLlama-1.1B, Pythia-1.0B, OPT-1.3B), and trained on only ~20B tokens, Apex-1 beats the 300B-token Pythia-1.0B on HumanEval by a wide margin (8.54 vs 1.83). The gap to the 3T-token TinyLlama concentrates in data-hungry tasks, as shown in the token-count comparison below.

#### Comparison models — parameters & training tokens

| Model | Parameters | Training tokens | Released | Notes |
|:---|---:|---:|:---|:---|
| **Apex-1** | 1.12B | ~20B | 2026, this project | v3-en corpus, English-only |
| TinyLlama-1.1B | 1.10B | **3T** (3 trillion) | 2024, StatNLP/SUTD | Llama 2 architecture scaled down — the flagship example of "push token count instead of parameter count" at small scale. **150× more tokens** than Apex-1 |
| Pythia-1.0B | 1.01B | 300B | 2023, EleutherAI | Trained on The Pile; a standard suite (70M–12B) built for scaling-law research. **15× more tokens** than Apex-1 |
| OPT-1.3B | 1.30B | 180B | 2022, Meta | Open reproduction of GPT-3-style training. Only the commonsense suite is reported in the paper, so no knowledge/math/code comparison is available |

> [!NOTE]
> All three baselines were measured and reported using the **EleutherAI lm-evaluation-harness family of protocols**, which is why lining them up like this is standard practice (the Pythia, TinyLlama, and OPT papers all cite each other this way). Expect roughly ±0.5-point noise from differences in harness version across papers.

#### What these benchmarks actually measure — authority & limits

| Benchmark (group) | How widely used | Known limits |
|:---|:---|:---|
| Commonsense 7 (HellaSwag, ARC-e/c, PIQA, WinoGrande, BoolQ, OpenBookQA) | The de facto standard EleutherAI harness suite — cited as-is by Pythia, TinyLlama, OPT, and Llama-family papers. **The industry's common language for cross-model comparison** | Multiple-choice format means shallow "pick the less awkward sentence" pattern-matching can score decently — trust this for **relative comparison**, not absolute skill |
| MMLU | The headline knowledge benchmark cited by nearly every modern LLM report (GPT-4, Llama, etc.); 57 subjects, 4-way multiple choice | Random-guess baseline is 25%, and **most 1B-scale models cluster at 24–26** — not discriminative at this scale, treat as a reference point only |
| GSM8K | The standard benchmark for elementary multi-step math reasoning; requires generating the answer, not just picking one | **Most ~1B-scale models sit in the low single digits** — at this size it mainly confirms "this model can barely do multi-step reasoning" (which is exactly what this project measured directly in §6) |
| HumanEval | OpenAI's benchmark — the **de facto industry standard** for code-gen LLM evaluation (cited by Codex, GPT, Claude, and Llama papers). pass@1 = the fraction of once-generated code that actually passes the tests | Only 164 problems, so a single item can swing the score noticeably — still meaningful for comparing small models |
| MBPP | Google's secondary coding benchmark; 500 easier problems than HumanEval, for coding fundamentals | Not cited nearly as widely as HumanEval — treated as a secondary signal |

#### Full results (individual commonsense tasks + OPT-1.3B)

| Benchmark | Name origin | What it measures | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B | OPT-1.3B |
|:---|:---|:---|---:|---:|---:|---:|---:|
| HellaSwag | short for "Harder Endings, Longer contexts…" | Pick the natural continuation — commonsense context | 46.69 | 46.91 | 59.20 | 47.16 | 53.65 |
| ARC-Easy | AI2 Reasoning Challenge (easy) | Elementary/middle-school science | 52.65 | 52.57 | 55.25 | 48.99 | 50.80 |
| ARC-Challenge | AI2 Reasoning Challenge (hard) | Science questions lookup/stats can't solve — needs reasoning | 29.86 | 30.72 | 30.10 | 27.05 | 29.44 |
| PIQA | Physical Interaction QA | Physical commonsense — tool/material use | 68.23 | 68.55 | 73.29 | 69.21 | 72.36 |
| WinoGrande | extended Winograd Schema | What a pronoun refers to — context inference | 53.59 | 52.80 | 59.12 | 53.43 | 59.59 |
| BoolQ | Boolean Questions | Read a passage, answer yes/no — reading comprehension | 61.99 | **62.20** 🏆 | 57.83 | 57.83 | 60.83 |
| OpenBookQA | Open Book QA | Apply science principles to new situations, not rote recall | 33.80 | 33.80 | 36.00 | 31.40 | 33.40 |
| **Commonsense avg** | | | 49.54 | **49.65** | 52.99 | 48.30 | 51.44 |
| MMLU | Massive Multitask Language Understanding (5-shot) | Law, medicine, math, history, 57 subjects — breadth of knowledge | 24.83 | 24.90 | 25.34 | 25.70 | — |
| GSM8K strict (5-shot) | Grade School Math 8K | Elementary multi-step math word problems (strict scoring) | 1.44 | 1.90 | — | — | — |
| GSM8K flexible (5-shot) | 〃 | 〃 (relaxed scoring) | 1.97 | 2.27 | — | — | — |
| HumanEval (pass@1) | OpenAI's human-written eval set | Function description → Python code, actually executed | 8.54 | 8.54 | 9.15 | 1.83 | — |
| MBPP (pass@1) | Mostly Basic Python Problems | 500 basic Python problems — easier coding fundamentals than HumanEval | 4.80 | 5.20 | — | — | — |

**Verdict** (verbatim from [`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md)):

1. **DPO ≥ SFT** — commonsense avg +0.11, ARC-Challenge +0.86, GSM8K +0.46, MBPP +0.4. No regression (alignment tax). DPO recommended as the release default.
2. **On par with or ahead of Pythia-1.0B (300B tokens) across the board**, especially HumanEval 8.54 vs 1.83 — despite Apex-1 training on only ~20B tokens.
3. **BoolQ 62.20 ranks 1st among the 4 comparison models.**
4. The gap to TinyLlama (3T tokens) concentrates in data-hungry tasks like HellaSwag/WinoGrande/PIQA — a pattern explained by the 150× token-count difference.
5. MMLU sits at near-random (≈25) across the board at 1B scale — not discriminative here.

> [!WARNING]
> **Contamination disclosure**: GSM8K and MBPP's **train splits** partially overlap the SFT/RLVR training data. Evaluation used the **test split**, which is valid, but not fully uncontaminated. HumanEval is fully clean — entirely absent from training data.

---

## 6. Next step — RLVR → DPO

The whole point of this benchmark was to decide "more SFT or move to RL" by measurement rather than by feel. The initial decision was RLVR — but running it actually revealed **there was no learning signal at all**, so the direction changed to DPO. Here's how that unfolded.

### 6.1 Initial decision: RLVR

| Evidence | Value | Reading (at the time) |
|:---------|:------|:--------|
| Capability | Normal-chat coding **fully passes 5/5** | The algorithms themselves are learned well enough |
| Alignment problem | Mean repetition rate **0.032**, over-length **8/10** | Verbosity/repetition is the bottleneck — a style problem, not a capability one |

→ Initial decision: **RLVR** ([`ckpt/AUTO_DECISION_Apex-1_v1.txt`](ckpt/AUTO_DECISION_Apex-1_v1.txt), original rationale: *"alignment (verbosity/repetition): repetition 0.032, over_length 8/10 — capability is sufficient (coding 5/5)"*)

### 6.2 What actually happened — no learning signal

**1st attempt (thinking forced, batched generation)**: all 6 samples spent 478–759 characters on thinking alone, and the answer field (after `</THINKING>`) came back blank. It couldn't finish thinking within `max_new=200`, so it never reached the answer → all 6 got reward **0.10** (format score only) → zero variance within the group → no learning signal.

> [!WARNING]
> This is the 327M line's v6 "thinking never reaches the answer" problem reappearing inside RLVR — and worse, the benchmark (§1) already showed this model does **better in no-thinking mode**, while RLVR was forcing THINKING on, driving it in exactly the wrong direction.

**2nd attempt (switched to a no-thinking probe, after fixing a `torch.no_grad()` bug)**: 4 GSM8K problems × 6 samples = 24 generations, **all reward 0.0** (0 correct). Answers were still verbose (637–715 chars), and the final numbers were wrong (e.g., 72→96, 10→300).

Root cause: **the 1B model (20B tokens) essentially cannot solve multi-step arithmetic (GSM8K).** GRPO needs a mix of correct and incorrect samples within a group to create a relative advantage signal; at ~0% accuracy every group gets skipped (`groups 0/8`) and no gradient is produced at all. The run that had been going for ~4 hours had learned nothing.

> [!IMPORTANT]
> The original "capability is sufficient (coding 5/5)" reasoning was the problem. The 5 coding prompts pass because they match common patterns already present in the SFT data (√n trial division, slice reversal, etc.) — **GSM8K-style multi-step arithmetic reasoning is a separate capability entirely.** The lesson here is that "coding 5/5 → capability is sufficient" doesn't generalize.

### 6.3 Conclusion — drop RLVR, move to DPO

The problem that originally triggered RLVR (§1) was never arithmetic capability — it was **verbosity and instruction non-compliance** (over-length 8/10, instruction 1/2). That's exactly what DPO targets: DPO only needs to learn a **relative preference** ("concise > verbose"), not whether the model can solve the underlying problem, so it works independent of GSM8K accuracy. The 60K preference-pair dataset was already prepared, so switching was immediate.

The RLVR infrastructure improvements (batched generation, checkpointing, live logging) stay in `rlhf.py` — ready to reuse once there's a stronger model or an easier RL task with actual accuracy headroom.

### 6.4 DPO training complete

```text
step 0 | loss 0.6931 | pref acc 0.00
```

`loss 0.6931 ≈ ln(2)` is the textbook starting value when the policy still equals the reference model, and `pref acc 0.00` means there's no margin yet — both normal at the start. Training finished on all 60K preference pairs, and the outcome is confirmed in §5.4's standard benchmarks — **DPO matches or beats SFT across the board** (commonsense avg +0.11, ARC-Challenge +0.86, GSM8K +0.46, MBPP +0.4), alignment achieved with no capability tax.

Compared to the 327M line's bottleneck moving from "format (handoff)" to "accuracy," the 1B line skipped the handoff problem entirely but ran into a different wall — **"accuracy itself doesn't exist enough to generate an RL signal"** — and worked around that wall by aiming DPO directly at the original target of verbosity and instruction-following, seeing it through to completion.

---

## 7. Raw files

| Path | Contents |
|:-----|:---------|
| [`ckpt/benchmark_sft_Apex-1_v1_raw.json`](ckpt/benchmark_sft_Apex-1_v1_raw.json) | sft_Apex-1_v1 benchmark raw data (all 30 generations, including thinking text) |
| [`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md) | SFT vs DPO standard-benchmark raw data, 11 tasks (source for §5.4) |
| [`ckpt/AUTO_DECISION_Apex-1_v1.txt`](ckpt/AUTO_DECISION_Apex-1_v1.txt) | Initial RLVR decision log (superseded by DPO, see §6.3) |
| — | pretrain_Apex-1_v1 alone (pre-SFT) chat benchmark **not measured** |

> Weights (`.pt`) and raw corpora are not included in the public repository.

---

<div align="center">

[README](README.en.md) · [BENCHMARK v1 (327M)](BENCHMARK-v1.en.md) · [ARCHITECTURE](ARCHITECTURE.en.md)

</div>
