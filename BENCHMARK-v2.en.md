[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](BENCHMARK-v2.md) [![English](https://img.shields.io/badge/English-0969DA?style=flat-square)](BENCHMARK-v2.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](BENCHMARK-v2.ja.md)

[← README](README.en.md)

---

# 📊 BENCHMARK v2 · APEX-1 (xl, ~1.12B)

> A different line from v1 (327M `base`). A **1B-scale** line trained from scratch with an **English-only** tokenizer (32K) and corpus.

| | |
|:--|:--|
| 🆕 **Latest** | `sft_xl_v1` · 2026-07-22 |
| 📦 **Model** | APEX-1 · Decoder-only · measured **1,119.5M** (`xl`) |
| 🧪 **Set** | English-only **15 prompts × THINKING on/off** (separate set from the 327M line; only the 5 coding prompts are shared) |
| 📁 **Raw** | [`ckpt/benchmark_sft_xl_v1_raw.json`](ckpt/benchmark_sft_xl_v1_raw.json) |

> [!WARNING]
> Still the **first snapshot with only one checkpoint.** There's no version ladder like the v1 line yet — the point of this benchmark was to decide the next alignment step (more SFT vs. RL) by measurement, not by feel.

### 📑 Contents

1. [At a glance](#1-at-a-glance)
2. [Training process](#2-training-process)
3. [Datasets](#3-datasets)
4. [Scoring](#4-scoring)
5. [sft_xl_v1 results](#5-sft_xl_v1-results)
6. [Next step — RLVR](#6-next-step--rlvr)
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
        ▼  pretrain_xl_v1 (stage1)
④b Curriculum reweight — resample 7GB each of fineweb/dclm + all code + finemath4 + cosmopedia2 (12.85B tok)
        ▼
④c Pretrain stage-2 continue · 10,000 steps · lr 6e-5 (same batch/accum)
        ▼  pretrain_xl_v1 ★ (final pretrain)
        ▼
⑤ SFT · smoltalk2 English subset, 538K examples × 2 epochs ÷ (batch8×accum16) ≈ 8,400 steps · lr 3e-5
        ▼  sft_xl_v1 ★
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

## 5. sft_xl_v1 results

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

---

## 6. Next step — RLVR

The whole point of this benchmark was to decide "more SFT or move to RL" by measurement rather than by feel.

| Evidence | Value | Reading |
|:---------|:------|:--------|
| Capability | Normal-chat coding **fully passes 5/5** | The algorithms themselves are learned well enough |
| Alignment problem | Mean repetition rate **0.032**, over-length **8/10** | Verbosity/repetition is the bottleneck — a style problem, not a capability one |

**→ Decision: RLVR** ([`ckpt/AUTO_DECISION_xl_v1.txt`](ckpt/AUTO_DECISION_xl_v1.txt))

> [!NOTE]
> Original rationale text: *"alignment (verbosity/repetition): repetition 0.032, over_length 8/10 — capability is sufficient (coding 5/5)"*
>
> If this were a capability problem, more SFT data/steps would be the right call. But the current bottleneck is "how it answers," not "what it knows," so the next post-training step is **RLVR (RL with verifiable rewards)** rather than DPO or additional SFT.

Compared to the 327M line's bottleneck moving from "format (handoff)" to "accuracy," the 1B line skipped the handoff problem entirely and starts directly at the **"accuracy + verbosity"** stage.

---

## 7. Raw files

| Path | Contents |
|:-----|:---------|
| [`ckpt/benchmark_sft_xl_v1_raw.json`](ckpt/benchmark_sft_xl_v1_raw.json) | sft_xl_v1 benchmark raw data (all 30 generations, including thinking text) |
| [`ckpt/AUTO_DECISION_xl_v1.txt`](ckpt/AUTO_DECISION_xl_v1.txt) | RLVR decision log |
| — | pretrain_xl_v1 alone (pre-SFT) chat benchmark **not measured** |

> Weights (`.pt`) and raw corpora are not included in the public repository.

---

<div align="center">

[README](README.en.md) · [BENCHMARK v1 (327M)](BENCHMARK-v1.en.md) · [ARCHITECTURE](ARCHITECTURE.en.md)

</div>
