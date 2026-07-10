<div align="center">

# Benchmark Report · Base V1

**How it was trained, what data was used, and what the model actually answered**

<br/>

![ckpt](https://img.shields.io/badge/sft__base__v1-326.7M-3b82f6?style=flat-square)
![pretrain](https://img.shields.io/badge/pretrain-18k%20steps-22c55e?style=flat-square)
![sft](https://img.shields.io/badge/SFT-5k%20steps-22c55e?style=flat-square)
![status](https://img.shields.io/badge/status-early%20stage-f59e0b?style=flat-square)

<br/>

[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.en.md)

</div>

---

### In one glance

| | Result |
|:--|:-------|
| **Pretrain only** | Almost no chat-format ability (~0 / 28) |
| **After SFT** | Tries to follow format; content often wrong |
| **Thinking mode** | Empty answers (0 / 14) → grade normal chat instead |

Not production quality yet.  
This is the first snapshot that shows the **pipeline works** and what SFT changes.

---

## Contents

1. [Training](#1-training)  
2. [Datasets](#2-datasets)  
3. [Scoring](#3-scoring)  
4. [Score summary](#4-score-summary)  
5. [Pretrain vs SFT](#5-pretrain-vs-sft)  
6. [Questions and answers](#6-questions-and-answers)  
7. [Coding items](#7-coding-items)  
8. [Takeaways](#8-takeaways)

---

## 1. Training

### Model card

| Item | Value |
|:-----|:------|
| Family | Decoder-only Transformer (LLaMA-style parts) |
| Params | ~**326.7M** (`base`) |
| Depth · width | 24 layers · d_model 1024 |
| Attention | GQA (Q 16 · KV 4) + RoPE |
| FFN | SwiGLU |
| Norm | RMSNorm (Pre-Norm) |
| Other | weight tying, no bias |

```text
tokens
   │
embedding ───────────────────────────┐
   │                                 │ (tied)
[ Block × 24 ]                       │
  RMSNorm → Attention → +            │
  RMSNorm → SwiGLU    → +            │
   │                                 │
RMSNorm                              │
   │                                 │
LM Head ◄────────────────────────────┘
   │
next-token logits
```

### Order of work

```text
  ① BPE tokenizer (vocab 64k)
           │
           ▼
  ② Pretrain · 18,000 steps · lr 6e-4
           │
           ▼  pretrain_base_v1
  ③ SFT · 5,000 steps · lr 3e-5
           │
           ▼  sft_base_v1
  ④ This benchmark
```

| Stage | Checkpoint | steps | Init | Data fingerprint |
|:------|:-----------|------:|:-----|:-----------------|
| Pretrain | `pretrain_base_v1` | 18,000 | scratch | `train.bin` · `53f815d47abc4887` |
| SFT | `sft_base_v1` | 5,000 | from pretrain | `sft.pt` · `bbc2211091309d3c` |

Shared: AdamW (β 0.9 / 0.95, wd 0.1), grad clip 1.0, warmup + cosine, CUDA.

### Loss

**Pretrain**

| step | train | val |
|----:|------:|----:|
| 0 | 11.29 | |
| 500 | 3.42 | 4.35 |
| 1,000 | 2.76 | 4.08 |
| 1,500 | 3.00 | 3.25 |
| 16,500 | | 2.72 |
| 17,500 | | 2.81 |
| late | ~2.1–2.5 | |

**SFT**

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

Lower loss means “adapting to chat format,” not automatic benchmark wins. See Q&A below.

---

## 2. Datasets

Raw multi-GB files are **not** in the GitHub repo.  
Public HF sources were downloaded and prepared with `data.py`.

### Tokenizer

| | |
|:--|:--|
| Type | Byte-level BPE |
| Size | **64,000** vocab |
| Sample | fineweb, wiki (ko/ja), tinystories (~900MB) |
| Specials | role tokens, thinking span tokens |

### Pretrain · ~1.91B tokens

`train 1,893,646,391` · `val 19,127,741`

<details>
<summary><b>Natural language (EN / KO / JA)</b></summary>

<br/>

| Lang | File | Source | Size |
|:----:|:-----|:-------|:-----|
| EN | fineweb_edu | HuggingFaceFW/fineweb-edu | ~1.3GB |
| EN | tinystories | roneneldan/TinyStories | ~2.1GB |
| KO | fineweb2_ko | HuggingFaceFW/fineweb-2 | ~1.1GB |
| KO | wikipedia_ko | wikimedia 20231101.ko | ~1.3GB |
| JA | fineweb2_ja | HuggingFaceFW/fineweb-2 | ~1.1GB |
| JA | wikipedia_ja | wikimedia 20231101.ja | ~1.6GB |

</details>

<details>
<summary><b>Code</b></summary>

<br/>

| Content | Source | Note |
|:--------|:-------|:-----|
| 11 languages | codeparrot/github-code-clean | ~1,200 files each |
| Functions | Fsoft-AIC/the-vault-function | ~40k rows |
| Commits + diffs | bigcode/commitpack | ~8k rows |

</details>

<details>
<summary><b>SFT · 321,367 examples</b></summary>

<br/>

| Area | Source | rows |
|:-----|:-------|----:|
| Code instruct | Magicoder-OSS-Instruct | 75,197 |
| EN general | OpenHermes-2.5 | 60,000 |
| EN multi-turn | UltraChat 200k | 35,000 |
| Japanese | shisa / llm-jp / dolly-ja mix | 65,015 |
| Korean | KoAlpaca v1.1a | 21,155 |
| Math CoT | OpenR1-Math | 25,000 |
| Thinking CoT | OpenThoughts3 | 40,000 |

Example line:

```json
{
  "user": "2 + 3 * 4 = ?",
  "thinking": "Multiply first. 3*4=12, then 2+12=14.",
  "assistant": "The answer is 14."
}
```

User / system tokens are masked out of the loss.

</details>

### Mix (rough)

```text
Pretrain tokens
  EN web · stories   ████████
  KO/JA wiki · web   ████████
  code               ██

SFT examples
  EN chat            ██████████  ~30%
  code instruct      ████████    ~23%
  Japanese           ██████      ~20%
  thinking · math    ██████      ~20%
  Korean             ██          ~7%
```

---

## 3. Scoring

| Item | Detail |
|:-----|:-------|
| Items | 14 prompts × 2 modes = 28 generations |
| Modes | thinking on / off |
| QA · open | 0–5 (full transcript) |
| Coding | unit tests |
| Dates | generated 2026-07-09 · scored 2026-07-10 |

---

## 4. Score summary

### sft_base_v1

| Mode | Result |
|:-----|:-------|
| Thinking **on** | answers **0 / 14** · avg 0.0 |
| Chat · Korean | avg **1.33 / 5** |
| Chat · Japanese | avg **0.33 / 5** |
| Chat · English | avg **2.67 / 5** |
| Chat · coding | tests **4 / 17** · full pass **1 / 5** |

Thinking mode ends before the close tag, so the runtime returns an empty answer.  
**Use normal chat mode with this checkpoint.**

---

## 5. Pretrain vs SFT

| | Pretrain | After SFT |
|:--|:---------|:----------|
| Overall | ~all zeros | scores appear in chat mode |
| Coding tests | 0 / 34 | 4 / 17 (chat) |
| Behavior | echo / loops | tries answer shape (often wrong) |

Five thousand SFT steps move the model from  
**“cannot follow” → “tries to follow.”**  
Quality beyond that is still far away.

---

## 6. Questions and answers

Scores are **sft_base_v1 · normal chat** unless noted.

---

### Korean

#### Fact · capital

| | |
|:--|:--|
| **Score** | 2 / 5 |
| **Prompt** | 대한민국의 수도는 어디인가요? |
| **Expect** | Seoul |
| **Answer** | Mentions Seoul but with self-contradictory, circular wording |
| **Note** | Keyword hit, logic broken |

Thinking mode: empty answer field.

---

#### Math · apples

| | |
|:--|:--|
| **Score** | 1 / 5 |
| **Prompt** | 철수는 사과를 5개… 3개를 먹었습니다. 남은 사과는? |
| **Expect** | 2 |
| **Answer** | Claims there are 3 apples |
| **Note** | No subtraction |

---

#### Summary

| | |
|:--|:--|
| **Score** | 1 / 5 |
| **Prompt** | Summarize in one sentence (Korean park / weather text) |
| **Answer** | Echoes the source |
| **Note** | Instruction ignored |

---

### Japanese

#### Fact · capital

| | |
|:--|:--|
| **Score** | 0 / 5 |
| **Prompt** | 日本の首都はどこですか? |
| **Expect** | 東京 |
| **Answer** | Talks about population, not capital |
| **Note** | Off-topic |

---

#### Math · apples

| | |
|:--|:--|
| **Score** | 1 / 5 |
| **Prompt** | 太郎はりんごを7個…2個食べました。残りは? |
| **Expect** | 5 |
| **Answer** | “Taro ate 2.” |
| **Note** | Restates, no math |

---

#### Translate

| | |
|:--|:--|
| **Score** | 0 / 5 |
| **Prompt** | Translate the Japanese sentence into English |
| **Answer** | Stays in Japanese |
| **Note** | Ignores translation request |

---

### English

#### Fact · capital

| | |
|:--|:--|
| **Score** | **4 / 5** |
| **Prompt** | What is the capital of France? |
| **Expect** | Paris |
| **Answer** | “The capital of France is Paris…” then mild geo rambling |
| **Note** | Best factual signal in this run |

Pretrain-only: rewrites as a Germany capital question → 0.

---

#### Math · train

| | |
|:--|:--|
| **Score** | 1 / 5 |
| **Prompt** | 60 km/h for 2 hours → how far? |
| **Expect** | 120 |
| **Answer** | Uses 0.5 hours → 60 km |
| **Note** | Broken arithmetic |

---

#### Summary

| | |
|:--|:--|
| **Score** | 3 / 5 |
| **Prompt** | Summarize the park walk sentence in **one** sentence |
| **Answer** | Four sentences + invented swings/flowers |
| **Note** | Gist ok, instruction fail |

---

## 7. Coding items

Function-level prompts, unit-tested.  
**Normal chat only** (thinking mode: all fail).

| Task | Tests | Score | Note |
|:-----|:-----:|:-----:|:-----|
| is_prime | 0 / 5 | 0 | becomes is_even |
| reverse_string | 0 / 3 | 0 | only `.lower()` |
| factorial | 0 / 3 | 1 | missing zero base |
| **is_palindrome** | **3 / 3** | **5** | **only clean pass** |
| find_max | 1 / 3 | 1 | shadowing, lucky case |

### Pass

```python
def is_palindrome(s):
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]
```

### Fail (prime → even)

```python
def is_prime(n):
    if n <= 1:
        return False
    return n % 2 == 0
```

### Fail (reverse)

```python
def reverse_string(s):
    return s.lower()
```

---

## 8. Takeaways

### What works

- Pretrain loss 11 → ~2, SFT 2.67 → ~1.05 → **training loop is real**
- SFT creates a measurable “tries to answer” jump
- One English fact and one coding pass are clear signals

### What’s stuck

1. Thinking close-tag bug  
2. Weak KO/JA facts and arithmetic  
3. Summary / translate instruction ignores  
4. Coding full pass 1 / 5  

### Next

- SFT that closes thinking tags (v2 direction)  
- More coding · math, optional RLVR  
- DPO / corrective SFT  
- Keep the **same 14 prompts** across versions  

---

### Local sources

| Path | Content |
|:-----|:--------|
| `llm/ckpt/benchmark_sft_base_v1.json` | Full SFT items |
| `llm/ckpt/benchmark_pretrain_base_v1.json` | Pretrain compare |
| `llm/DATA_SOURCES.md` | Dataset manifest |

Weights and raw corpora are not published on GitHub.

---

<div align="center">

[README](README.en.md) · [Architecture](ARCHITECTURE.en.md) · [Post-training](POST-TRAINING.en.md)

</div>
