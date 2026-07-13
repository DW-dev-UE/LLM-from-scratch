[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.en.md)

---

# 📊 BENCHMARK · base ~327M

> Closer to a **training journal** than a leaderboard.  
> Goal: record *how we trained · what data we used · what the model answered*.

| | |
|:--|:--|
| 🆕 **Latest** | `sft_base_v3` · 2026-07-13 |
| 📦 **Model** | Decoder-only · ~**326.7M** (`base`) |
| 🧪 **Set** | Same **14 prompts × THINKING on/off** (for version comparison) |
| 📁 **Raw** | [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) |

> ⚠️ **Not production-ready.**  
> These snapshots show whether the pipeline runs and what repeated SFT changes.

### 📑 Contents

1. [At a glance](#1-at-a-glance)
2. [Training process](#2-training-process)
3. [Datasets](#3-datasets)
4. [Scoring](#4-scoring)
5. [sft_base_v3 · latest](#5-sft_base_v3--latest)
6. [Earlier versions](#6-earlier-versions)
7. [Takeaways · next](#7-takeaways--next)

---

## 1. At a glance

Bold column = **current latest**.

| | Pretrain | SFT v1 | SFT v2 | ⭐ **SFT v3** |
|:--|:--------:|:------:|:------:|:------------:|
| Normal-chat feel | ~0 | low | low | **2.57 / 5** |
| Coding full pass | 0/5 | 1/5 | 1/5 | **4/5** |
| THINKING answers | — | **0/14** | **13/14** | **12/14** |
| Read | no chat | tries format | close-tag fixed | coding↑ · KO↓ |

```text
pretrain ──► sft_v1 ──► sft_v2 ──► sft_v3 ★
  chat ~0      format try   THINKING close   coding 4/5
```

| Ver | What worked | What broke |
|:---:|:------------|:-----------|
| v1 | Starts following instruction format | THINKING answers all empty |
| v2 | Most `</THINKING>` closures restored | Coding / quality still weak |
| **v3** | **Coding 4/5**, EN fact/math recovered | **Korean collapse**, THINKING coding 0/5 |

---

## 2. Training process

### 🧩 Model card

| Item | Value |
|:-----|:------|
| Family | Decoder-only Transformer (LLaMA-style) |
| Params | ~**326.7M** |
| Depth · width | 24 layers · `d_model` 1024 |
| Attention | GQA (Q 16 · KV 4) + RoPE |
| FFN | SwiGLU |
| Norm | RMSNorm (Pre-Norm) |
| Other | weight tying · no bias |

```text
input tokens
   │
embedding ───────────────────────────┐
   │                                 │ weight tying
[ Block × 24 ]
  RMSNorm → Attention → +
  RMSNorm → SwiGLU    → +
   │
RMSNorm
   │
LM Head ◄────────────────────────────┘
   │
next-token probs
```

### 🛤️ Training order

```text
① Tokenizer (BPE · vocab 64k)
        ▼
② Pretrain · 18k steps · lr 6e-4
        ▼  pretrain_base_v1
③ SFT v1 · 5k steps · lr 3e-5
        ▼  sft_base_v1
④ SFT v2 · reinforce THINKING close patterns
        ▼  sft_base_v2
⑤ SFT v3 · coding data augmentation
        ▼  sft_base_v3  ★ this doc
```

| Stage | Checkpoint | steps | Note |
|:------|:-----------|------:|:-----|
| Pretrain | `pretrain_base_v1` | 18,000 | from scratch · `53f815d47abc4887` |
| SFT v1 | `sft_base_v1` | 5,000 | from pretrain · `bbc2211091309d3c` |
| SFT v2 | `sft_base_v2` | ~5.6k–5.7k | goal: THINKING closure |
| ⭐ SFT v3 | `sft_base_v3` | augmented SFT | coding full pass **4/5** |

Shared: **AdamW** (β 0.9 / 0.95, wd 0.1) · grad clip 1.0 · warmup + cosine · CUDA

### 📉 Loss (base · v1 era)

<details>
<summary><b>Pretrain · SFT v1 loss tables</b></summary>

<br/>

**Pretrain**

| step | train | val |
|----:|------:|----:|
| 0 | 11.29 | — |
| 500 | 3.42 | 4.35 |
| 1,000 | 2.76 | 4.08 |
| 1,500 | 3.00 | 3.25 |
| 16,500 | | 2.72 |
| 17,500 | | 2.81 |
| late | ~2.1–2.5 | |

**SFT v1**

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

</details>

> 💡 Falling loss = *adapting to chat format*.  
> It does **not** mean benchmark scores instantly rise.

---

## 3. Datasets

Raw multi-GB corpora are not in the repo.  
Downloaded from public HuggingFace etc., prepared with `data.py`.

### 🔤 Tokenizer

| | |
|:--|:--|
| Type | Byte-level BPE |
| Size | **64,000** vocab |
| Sample | fineweb, wiki(ko/ja), tinystories, … ~900MB |
| Specials | role tokens · `THINKING` span tokens |

### 📚 Pretrain · ~1.91B tokens

`train 1,893,646,391` · `val 19,127,741`

<details>
<summary><b>🌍 Natural language (EN / KO / JA)</b></summary>

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
<summary><b>💻 Code</b></summary>

<br/>

| Content | Source | Note |
|:--------|:-------|:-----|
| python, c, cpp, java, js, ts, go, rust, … | codeparrot/github-code-clean | ~1,200 files / lang |
| Function-level | Fsoft-AIC/the-vault-function | ~40k rows |
| Commits + diffs | bigcode/commitpack | ~8k rows |

</details>

<details>
<summary><b>🗣️ SFT · 321,367 examples (v1 mix)</b></summary>

<br/>

| Area | Source | rows |
|:-----|:-------|----:|
| Code instruct | Magicoder-OSS-Instruct | 75,197 |
| EN general | OpenHermes-2.5 | 60,000 |
| EN multi-turn | UltraChat 200k | 35,000 |
| Japanese | shisa / llm-jp / dolly-ja | 65,015 |
| Korean | KoAlpaca v1.1a | 21,155 |
| Math CoT | OpenR1-Math | 25,000 |
| Thinking CoT | OpenThoughts3 | 40,000 |

```json
{
  "user": "2 + 3 * 4 = ?",
  "thinking": "Multiply first. 3*4=12, then 2+12=14.",
  "assistant": "The answer is 14."
}
```

User / system spans are masked from loss — learn the **answer side**, not the question echo.

v2/v3 continue this mix:  
→ v2 reinforces THINKING close patterns · v3 reinforces coding quality.

</details>

### Mix feel (v1)

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

## 4. Scoring

| Item | Detail |
|:-----|:-------|
| Items | **14 prompts × 2 modes = 28 gens** (same every version) |
| Modes | 🧠 THINKING on / 💬 normal chat |
| QA · open | 0–5 (full transcript graded) |
| Coding | unit tests executed |
| v3 gen | temp `0.7` · top_p `0.9` · max_new `256` · seed `0` · 1 sample |
| Dates | v1 `07-09/10` · v2 `07-10` · **v3 `07-13`** (2026) |

> 📌 Coding denominators: v1/v2 total **17** · v3 total **25** (5 cases / task).  
> Prefer **full pass N/5** for cross-version comparison.

---

## 5. sft_base_v3 · latest

| | |
|:--|:--|
| 🏁 Checkpoint | `ckpt/sft_base_v3.pt` |
| 📁 JSON | [`benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) |

### 5.1 Score cards

#### 💬 Normal chat (THINKING off) — overall avg **2.57 / 5**

| Area | Score | Status |
|:-----|:-----:|:------:|
| 🇰🇷 Korean | **0.00 / 5** | 🔴 |
| 🇯🇵 Japanese | **1.33 / 5** | 🟡 |
| 🇺🇸 English | **4.00 / 5** | 🟢 |
| 💻 Coding | **20/25** tests · **4/5** full pass | 🟢 |

#### 🧠 THINKING on

| Metric | Value | Status |
|:-------|:-----:|:------:|
| Avg score | **0.21 / 5** | 🔴 |
| Non-empty answers | **12 / 14** | 🟢 (closure mostly OK) |
| Coding full pass | **0 / 5** | 🔴 prose only |

### 5.2 v2 → v3 highlights

| | Change |
|:--|:-------|
| ✅ | Coding full pass **0/5 → 4/5** (`is_prime`, `factorial`, `is_palindrome`, `find_max`) |
| ✅ | EN Paris **5/5**, 120 km **5/5** (recovered from v2 Burgundy) |
| ⚠️ | `reverse_string` body `s[::-1]` correct · top-level `print` → NameError → **0/5** |
| ❌ | All 6 Korean items (thinking + normal) **score 0** — category collapse |
| ❌ | THINKING coding still **no code 0/5** |
| 🔍 | Correct mid-chain arithmetic then wrong finals more often (e.g. 7−2=5 → ×7=35) |

---

### 5.3 Q&A · normal chat

Scores are **sft_base_v3 · THINKING off**.

#### 🇰🇷 Korean — avg 0.00

| Item | Score | Expect | What the model did | Note |
|:-----|:-----:|:------:|:-------------------|:-----|
| Capital | **0** | Seoul | GDP/area hallucination, no “서울” | worse than v1 (2) |
| Apples 5−3 | **0** | 2 | restates “ate 3” | no subtraction |
| Summary | **0** | one sentence | infinite phrase loop | degenerate |

> 🧠 THINKING capital: reaches “서울” inside thinking, but **empty answer** after the close tag.

#### 🇯🇵 Japanese — avg 1.33

| Item | Score | Expect | What the model did | Note |
|:-----|:-----:|:------:|:-------------------|:-----|
| Capital | **4** | 東京 | 「首都は東京です」+ landmark spam | correct, slight deduct |
| Apples 7−2 | **0** | 5 | `7−8=2` collapse | no 5 |
| Translate | **0** | English | “Hopper's Park is a park for kids.” | unrelated |

> 🧠 THINKING: capital has 東京 in thinking, empty answer (regressed from v2 thinking 5).  
> Math finds 7−2=5 then derails to 35.

#### 🇺🇸 English — avg 4.00

| Item | Score | Expect | What the model did | Note |
|:-----|:-----:|:------:|:-------------------|:-----|
| Capital | **5** | Paris | “The capital of France is Paris.” | recovered 🟢 |
| Train 60×2 | **5** | 120 | Distance = Speed × Time → 120 km | clean 🟢 |
| Summary | **2** | one sentence | “The weather was nice today.” | drops detail |

---

### 5.4 Coding · normal chat

Function signatures only; extracted code **executed** on unit tests.  
🧠 THINKING-mode coding (5 tasks) → prose only → **0/5**.

| Task | Tests | Score | One-line |
|:-----|:-----:|:-----:|:---------|
| `is_prime` | **5/5** | **5** | ✅ √n trial division |
| `reverse_string` | 0/5 | 0 | ⚠️ body OK · `print` NameError |
| `factorial` | **5/5** | **5** | ✅ recursive base case (fixes v2) |
| `is_palindrome` | **5/5** | **5** | ✅ lower + reverse (fixes always-True) |
| `find_max` | **5/5** | **5** | ✅ hand-rolled loop (vs builtin `max`) |

<details>
<summary><b>✅ Passing code</b></summary>

<br/>

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]

def find_max(lst):
    max_num = lst[0]
    for num in lst:
        if num > max_num:
            max_num = num
    return max_num
```

</details>

<details>
<summary><b>⚠️ Near-miss · reverse_string</b></summary>

<br/>

```python
def reverse_string(s):
    return s[::-1]

print(reverse_string(s))  # NameError: s is not defined → executed 0/5
```

The function is correct; a stray top-level `print` breaks the harness.

</details>

---

## 6. Earlier versions

Archive for comparison — expand only when needed.

<details>
<summary><b>📘 sft_base_v2 · 2026-07-10</b></summary>

<br/>

📁 [`benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json)

#### Scores

| Mode | Result |
|:-----|:-------|
| 🧠 THINKING | answers **13/14** · avg **0.71/5** |
| 🇰🇷 normal | **0.33/5** |
| 🇯🇵 normal | **2.00/5** |
| 🇺🇸 normal | **1.00/5** |
| 💻 normal | tests **9/17** · full **1/5** (`find_max` via builtin `max()`) |

#### v1 → v2

| | |
|:--|:--|
| ✅ | THINKING close mostly fixed (0/14 → 13/14) |
| ✅ | First thinking-mode perfect: `ja_fact` **5/5** |
| ⚠️ | EN fact = Burgundy hallucination (temp 0.7 · single sample) |
| ❌ | Coding still weak · no thinking-mode code |

#### Normal chat at a glance

| Item | Score | Note |
|:-----|:-----:|:-----|
| ko_fact / math / summary | 0 / 0 / 1 | Korean weak |
| ja_fact / math / translate | **5** / 1 / 0 | 東京 correct |
| en_fact / math / summary | 0 / 2 / 1 | Burgundy |
| coding full | 1/5 | `find_max` only |

#### Coding (normal)

| Task | Tests | Note |
|:-----|:-----:|:-----|
| is_prime | 2/5 | partial |
| reverse_string | 2/3 | partial |
| factorial | 0/3 | infinite recursion etc. |
| is_palindrome | 2/3 | always-True class |
| **find_max** | **3/3** | delegates to `max()` |

</details>

<details>
<summary><b>📗 sft_base_v1 · 2026-07-09/10</b></summary>

<br/>

📁 [`benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json)

#### Scores

| Mode | Result |
|:-----|:-------|
| 🧠 THINKING | answers **0/14** · avg 0.0 🔴 |
| 🇰🇷 normal | **1.33/5** |
| 🇯🇵 normal | **0.33/5** |
| 🇺🇸 normal | **2.67/5** |
| 💻 normal | tests **4/17** · full **1/5** |

> 🧠 Generation ends before the close tag → inference leaves answer empty.  
> Prefer **normal chat mode** for this checkpoint.

#### Pretrain vs SFT v1

| | Pretrain | SFT v1 |
|:--|:---------|:-------|
| Scores | essentially all zero | scores appear in normal mode |
| Coding | 0/34 | 4/17 |
| Behavior | echo · loops | tries format (content often wrong) |

With 5,000 SFT steps: **“cannot follow → tries the format.”**

#### Q&A · normal chat

| Lang | Items | Scores | Note |
|:----:|:------|:------:|:-----|
| 🇰🇷 | capital / apples / summary | 2 / 1 / 1 | “서울” present, sentences contradict |
| 🇯🇵 | capital / apples / translate | 0 / 1 / 0 | population talk · ignores instruction |
| 🇺🇸 | capital / train / summary | **4** / 1 / 3 | Paris was the strongest signal then |

Pretrain only (en_fact): rewrites as Germany capital question → 0.

#### Coding (normal)

| Task | Tests | Score | Note |
|:-----|:-----:|:-----:|:-----|
| is_prime | 0/5 | 0 | becomes even-check |
| reverse_string | 0/3 | 0 | only `.lower()` |
| factorial | 0/3 | 1 | recursive skeleton, no zero base |
| **is_palindrome** | **3/3** | **5** | ✅ only clean pass |
| find_max | 1/3 | 1 | one lucky case |

```python
# ✅ v1 pass
def is_palindrome(s):
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]

# ❌ v1 is_prime → effectively is_even
def is_prime(n):
    if n <= 1:
        return False
    return n % 2 == 0
```

</details>

<details>
<summary><b>📙 pretrain_base_v1 · chat ~0</b></summary>

<br/>

📁 [`benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json)

| | Result |
|:--|:-------|
| Chat reaction | almost none · ~0 / 28 |
| Coding | 0 / 34 |
| Behavior | echo · loops · rewrite the question |

Before SFT, **instruction format itself is untrained**.  
After v1 SFT, “tries the format” first appears on the bench.

</details>

---

## 7. Takeaways · next

### ✅ What worked across versions

- Pretrain loss 11 → ~2, SFT v1 2.67 → ~1.05 → **training loop is real**
- v1: “tries the format” shows up on the bench
- v2: THINKING **closure** mostly restored (0/14 → 13/14)
- v3: normal-chat **coding 4/5** — data-augmentation goal hit
- EN fact/math stable again in v3

### 🚧 Still stuck

1. 🧠 THINKING-mode coding → still prose only (0/5)
2. 🇰🇷 Korean → category collapse in v3
3. Unstable arithmetic / instruction following (right mid-chain, wrong final)
4. Failures like `reverse_string`: **correct body + unsafe top-level**
5. Far from usable product quality

### 🧭 Next

| Pri | Action |
|:---:|:-------|
| 1 | SFT / constrained decoding so THINKING **emits code and closes tags** |
| 2 | Rebalance Korean fact · math · summary (repair collapse) |
| 3 | Safer coding output format (no harness-breaking side effects) |
| 4 | DPO / correction SFT / RLVR |
| 5 | Keep the **same 14 prompts** for version comparison |

---

### 📁 Source files

| Path | Content |
|:-----|:--------|
| [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) | ⭐ **latest** SFT v3 |
| [`ckpt/benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json) | SFT v2 |
| [`ckpt/benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json) | SFT v1 |
| [`ckpt/benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json) | Pretrain |

> Weights (`.pt`) and raw corpora are not in the public repo.

---

<div align="center">

[README](README.en.md) · [ARCHITECTURE](ARCHITECTURE.en.md) · [POST-TRAINING](POST-TRAINING.en.md)

</div>
