<div align="center">

# Benchmark Report — Base V1

<img src="https://img.shields.io/badge/checkpoint-sft__base__v1-blue?style=flat" alt="sft_base_v1" />
<img src="https://img.shields.io/badge/params-~327M-informational?style=flat" alt="params" />
<img src="https://img.shields.io/badge/status-early--stage-orange?style=flat" alt="early-stage" />

[한국어](BENCHMARK-v1.md) &nbsp;·&nbsp; **[English](BENCHMARK-v1.en.md)** &nbsp;·&nbsp; [日本語](BENCHMARK-v1.ja.md)

[← README.en.md](README.en.md) &nbsp;|&nbsp; Raw JSON: [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json) · [`benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json)

</div>

---

> **TL;DR**  
> `pretrain_base_v1` scores **0/28** on every graded item — it does not follow chat format.  
> `sft_base_v1` (5,000 SFT steps) **starts trying** to follow instructions, but accuracy is still low.  
> **THINKING mode returns empty answers on all 14 prompts (0/14)** — use `--no-thinking` for now.

## Table of Contents

1. [Evaluation setup](#1-evaluation-setup)
2. [Checkpoints](#2-checkpoints)
3. [Headline results](#3-headline-results)
4. [Pretrain vs SFT](#4-pretrain-vs-sft)
5. [Results by category](#5-results-by-category)
6. [Coding benchmark](#6-coding-benchmark)
7. [Interpretation & next steps](#7-interpretation--next-steps)
8. [Source data](#8-source-data)

---

## 1. Evaluation setup

| Item | Detail |
|---|---|
| Prompt set | 14 prompts × 2 modes (THINKING on / off) = **28 generations** |
| Categories | Korean QA/summary · Japanese QA/translation · English QA/summary · 5 Python coding tasks |
| QA / open scoring | 0–5 (qualitative grading of full transcripts) |
| Coding scoring | Unit tests (objective pass/fail) |
| Generated | 2026-07-09 |
| Scored | 2026-07-10 |

The same questions and rubric are used for both the pretrain and SFT checkpoints.

---

## 2. Checkpoints

| Name | Path | Training | Role |
|---|---|---|---|
| **pretrain_base_v1** | `ckpt/pretrain_base_v1.pt` | 18,000 pretrain steps, `base` preset | Pure next-token pretraining |
| **sft_base_v1** | `ckpt/sft_base_v1.pt` | 5,000 SFT steps on top of pretrain | Chat / instruction tune (**primary subject**) |

- Parameters: ~**326.7M** (`base` preset)
- Tokenizer: BPE vocab 64k (production train)
- Meta: [`llm/ckpt/SFT_base_v1/sft_base_v1.json`](llm/ckpt/SFT_base_v1/sft_base_v1.json)

---

## 3. Headline results

### sft_base_v1 — THINKING mode

| Metric | Value |
|---|---|
| Answers produced | **0 / 14 (0%)** |
| Average score | **0.0** |

> **Critical finding**  
> On every THINKING run the model emits `<|eos|>` before closing `</THINKING>`.  
> `infer.py` then puts the whole generation in the thinking bucket and returns an **empty answer**.  
> → Use this checkpoint with **`--no-thinking`** until a later SFT fix.

### sft_base_v1 — no-thinking mode

| Area | Average / pass rate |
|---|---|
| Korean (3 items) | **1.33 / 5** |
| Japanese (3 items) | **0.33 / 5** |
| English (3 items) | **2.67 / 5** |
| Coding tests | **4 / 17 (23.5%)** |
| Coding problems fully passed | **1 / 5** (`is_palindrome` only) |

English factual recall (Paris) is the strongest signal. Korean/Japanese fact & arithmetic QA and instruction-following (translate/summarize) are weak. Coding is mostly broken except one clean palindrome solution.

---

## 4. Pretrain vs SFT

| Metric | pretrain_base_v1 | sft_base_v1 |
|---|---|---|
| Overall (28 items) | **0 / 28** | Non-zero on many no-think QA/open items |
| Coding tests (both modes) | **0 / 34** | no-think **4 / 17** |
| Instruction following | Echo / repetition collapse | Attempts format; content often wrong |
| Takeaway | Expected failure without chat SFT | Real jump after only 5k SFT steps |

SFT clearly moves the model from *“cannot follow instructions at all”* to *“attempts to follow, mostly wrong on substance”* — still far from reliable product quality.

---

## 5. Results by category

Scores below are **no-thinking** only (THINKING is score 0 / empty answer for all).

### 5.1 Korean

| ID | Type | Score | Keywords | Notes |
|---|---|---|---|---|
| `ko_fact_1` | QA | **2** | 서울 ✓ | Mentions Seoul but self-contradictory / circular |
| `ko_math_1` | QA | **1** | 2 ✗ | No subtraction performed |
| `ko_summary_1` | open | **1** | — | Echoes source instead of summarizing |

### 5.2 Japanese

| ID | Type | Score | Keywords | Notes |
|---|---|---|---|---|
| `ja_fact_1` | QA | **0** | 東京 ✗ | Talks about population, not capital |
| `ja_math_1` | QA | **1** | 5 ✗ | Restates the question; no arithmetic |
| `ja_translate_1` | open | **0** | — | Ignores “translate to English”; stays in Japanese |

### 5.3 English

| ID | Type | Score | Keywords | Notes |
|---|---|---|---|---|
| `en_fact_1` | QA | **4** | Paris ✓ | Correct core answer; mild rambling / geo hallucination after |
| `en_math_1` | QA | **1** | 120 ✗ | Wrong factor (×0.5); never gets 120 |
| `en_summary_1` | open | **3** | — | Captures gist but 4 sentences + invented details |

---

## 6. Coding benchmark

Prompts are function-level (`is_prime(n)`, etc.). Extracted code is executed with unit tests.

### no-thinking

| ID | Function | Tests | Score | Notes |
|---|---|---|---|---|
| `code_prime` | `is_prime` | 0/5 | 0 | Logic is effectively `is_even` |
| `code_reverse` | `reverse_string` | 0/3 | 0 | Returns `.lower()`; markdown fence leak → SyntaxError |
| `code_factorial` | `factorial` | 0/3 | 1 | Recursive shape OK; missing `n==0` base case |
| `code_palindrome` | `is_palindrome` | **3/3** | **5** | **Only fully correct coding item** |
| `code_maxlist` | `find_max` | 1/3 | 1 | Shadowing / wrong compare; lucky pass on singleton |

### THINKING mode (coding)

All 5 problems: **0 tests · score 0** — THINKING never closes, so no usable code is extracted.

---

## 7. Interpretation & next steps

### What worked

- Pretrain → SFT produces a **measurable capability jump**
- English fact QA, partial instruction shape, and one clean coding pass are real signals
- Failure modes match the project roadmap’s early-validation stage — not a regression

### What’s blocked

1. **THINKING closure bug** — hard usability blocker (largely addressed in v2; separate report later)
2. **Arithmetic & multilingual facts** — weak KR/JA
3. **Strict instruction following** — echo / ignore on summarize & translate
4. **Coding** — only 1/5 full passes

### Recommended follow-ups (aligned with the roadmap)

- SFT that reinforces closing `</THINKING>` (v2 direction)
- Heavier coding + arithmetic SFT / RLVR
- DPO / corrective SFT for instruction alignment
- Keep the same 14-prompt set for version-to-version comparison

---

## 8. Source data

| File | Contents |
|---|---|
| [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json) | Full SFT v1 items, scores, notes |
| [`llm/ckpt/benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json) | Same items on pretrain |
| [`llm/ckpt/SFT_base_v1/sft_base_v1.json`](llm/ckpt/SFT_base_v1/sft_base_v1.json) | Training meta (steps, lr, data hashes) |

This report is a human-readable view of those JSON files. Numbers and quotes follow the source JSON.

<div align="center">

---

[← README.en.md](README.en.md) &nbsp;|&nbsp; [ARCHITECTURE.en.md](ARCHITECTURE.en.md) &nbsp;|&nbsp; [POST-TRAINING.en.md](POST-TRAINING.en.md)

</div>
