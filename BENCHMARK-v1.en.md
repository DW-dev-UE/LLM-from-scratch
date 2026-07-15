[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.en.md)

---

# 📊 BENCHMARK · base ~327M

> Closer to a **training journal** than a leaderboard.  
> Goal: record *how we trained · what data we used · what the model answered*.

| | |
|:--|:--|
| 🆕 **Latest** | `sft_base_v6` · 2026-07-15 |
| 📦 **Model** | Decoder-only · ~**326.7M** (`base`) |
| 🧪 **Set** | Same **14 prompts × THINKING on/off** (for version comparison) |
| 📁 **Raw** | [`ckpt/benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) |

> ⚠️ **Not production-ready.**  
> These snapshots show whether the pipeline runs and what changes when checkpoints or protocols change.

### 📑 Contents

1. [At a glance](#1-at-a-glance)
2. [Training process](#2-training-process)
3. [Datasets](#3-datasets)
4. [Scoring](#4-scoring)
5. [sft_base_v6 · latest](#5-sft_base_v6--latest)
6. [Earlier versions](#6-earlier-versions)
7. [Takeaways · next](#7-takeaways--next)

---

## 1. At a glance

Bold column = **current latest**.

> ⚠️ **How to read the table**  
> · **Protocol**: v1–v4 use temp `0.7` single-sample · **v5·v6 are greedy (temp `0.0`)**  
> · **Base fork**: v1–v4 sit on `pretrain_base_v1` · **v5·v6 are SFT on `pretrain_base_v2`**  
> · **v5→v6**: same base, same SFT source mix — **only prep_sft preprocessing + a decode budget split** changed. The cleanest single-variable comparison in the project so far.  
> · So **do not** read v6 > v5 > v4 > v3 as a pure SFT ladder.  
> · For coding, compare **full-pass N/5** (test denominators differ: 17 vs 25).

| | Pretrain v1 | SFT v1 | SFT v2 | SFT v3 | SFT v4 | SFT v5 | ⭐ **SFT v6** |
|:--|:-----------:|:------:|:------:|:------:|:------:|:------:|:------------:|
| No-think avg | ~0 | low† | low† | 2.57 | 2.07 | 3.21 | **3.93 / 5** |
| Coding full pass (NT) | 0/5 | 1/5 | 1/5 | 4/5 | 3/5 | 5/5 | **5/5** |
| Coding full pass (T) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **4/5** (+prime 3/5) |
| THINKING nonempty ans | — | **0/14** | **13/14** | 12/14 | 11/14 | 10/14 | **14/14** |
| THINKING avg | ~0 | 0.0 | 0.71 | 0.21 | 0.50 | 0.50 | **2.93** |
| Korean total (T+NT) | 0/30 | — | 2/30 | 0/30 | 1/30 | 8/30 | **10/30** |
| Base | v1 | v1 | v1 | v1 | v1 | v2 | **v2** |
| Sampling | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.0 greedy | **0.0 greedy** |
| Read | no chat | format try | close-tag fix | coding↑ · KO↓ | lateral · noise | coding perfect · KO↑ | handoff fixed · thinking coding unlocked |

† v1/v2 no-think only logged per-language avgs:  
v1 KO **1.33** / JA **0.33** / EN **2.67** · v2 KO **0.33** / JA **2.0** / EN **1.0**

```text
                    ┌─► sft_v1 ─► sft_v2 ─► sft_v3 ─► sft_v4
pretrain_v1 ────────┤   format    close-tag   coding 4/5   KO mix · lateral
   18k · lr 6e-4    │
                    └─► pretrain_v2 (25k · lr 1e-4 · corpus v2)
                              │
                              ├─► sft_v5    (same sft.pt as v4 · greedy bench)
                              │               coding 5/5 · NT 3.21 · KO 8/30
                              │
                              └─► sft_v6 ★  (same mix · prep_sft fix + split decode budget)
                                              thinking avg 0.50→2.93 · thinking coding 0/5→4/5
```

| Ver | What worked | What broke |
|:---:|:------------|:-----------|
| v1 | Starts following instruction format | THINKING answers all empty |
| v2 | Most `</THINKING>` closures restored | Coding / quality still weak |
| v3 | Coding **4/5**, EN fact/math recovered | **Korean collapse**, THINKING coding 0/5 |
| v4 | Higher KO SFT share · Seoul appears inside thinking | Bench KO barely recovers · item flips |
| v5 | **Coding 5/5**, NT **3.21**, KO **8/30** | **thinking→answer handoff** · T coding 0/5 |
| **v6** | **Handoff fixed** (14/14 nonempty) · thinking avg **2.93** · thinking coding **4/5** | Korean arithmetic still broken · repetition/degeneration · JA→EN translation |

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
② Pretrain v1 · 18k steps · lr 6e-4
        ▼  pretrain_base_v1
        ├─► ③ SFT v1 · 5k · lr 3e-5 ──► sft_base_v1
        ├─► ④ SFT v2 · ~5.7k · reinforce THINKING close ──► sft_base_v2
        ├─► ⑤ SFT v3 · 8k · coding data aug ──► sft_base_v3
        ├─► ⑥ SFT v4 · 9.2k · Korean SFT aug ──► sft_base_v4
        │
        └─► ②′ Pretrain v2 · 25k · lr 1e-4 · corpus v2
                ▼  pretrain_base_v2
                ├─► ⑦ SFT v5 · 9.2k · (same sft.pt as v4) ──► sft_base_v5
                └─► ⑧ SFT v6 · 9.5k · prep_sft fix + split decode budget ──► sft_base_v6 ★
```

| Stage | Checkpoint | steps | init | Data hash · note |
|:------|:-----------|------:|:-----|:-----------------|
| Pretrain v1 | `pretrain_base_v1` | 18,000 | — | `53f815d47abc4887` · lr 6e-4 |
| SFT v1 | `sft_base_v1` | 5,000 | pretrain_v1 | `bbc2211091309d3c` |
| SFT v2 | `sft_base_v2` | 5,700 | pretrain_v1 | `fc177a36965df49a` · THINKING close |
| SFT v3 | `sft_base_v3` | 8,000 | pretrain_v1 | `db63d09ddfa41388` · coding aug |
| SFT v4 | `sft_base_v4` | 9,200 | pretrain_v1 | `975c1771bfff1919` · KO +70k |
| Pretrain v2 | `pretrain_base_v2` | 25,000 | pretrain_v1 | `4ad58fc7307962c0` · lr 1e-4 |
| SFT v5 | `sft_base_v5` | 9,200 | pretrain_v2 | `975c1771bfff1919` (same as v4) |
| ⭐ SFT v6 | `sft_base_v6` | 9,500 | **pretrain_v2** | same 14-file mix · **`prep_sft` fix** (thinking tail-preservation: trimmed 47,273 / demoted 17,540) |

Shared: **AdamW** (β 0.9 / 0.95, wd 0.1) · grad clip 1.0 · warmup + cosine · CUDA · SFT lr `3e-5` · v6 on A100 40GB · batch 8 × accum 16

> 🔑 **v5’s main variable is the base (pretrain_v2), not a new SFT mix.**  
> v4 and v5 share the same `sft.pt` (`975c…`) and 9,200 steps — only init weights differ.

> 🔑 **v6’s main variable is neither the base nor the SFT source mix — it's data prep and inference code.**  
> `data.py`'s `prep_sft` changed how it handles long THINKING spans (trim/demote toward tail-preservation), and  
> `infer.py` fixed a bug where thinking and answer shared one `max_new` budget — they now get **separate budgets**.  
> Base and SFT source files are identical to v5, so the v5→v6 delta isolates the effect of these two fixes alone.

### 📉 Loss

<details>
<summary><b>Pretrain · SFT loss tables</b></summary>

<br/>

**Pretrain v1**

| step | train | val |
|----:|------:|----:|
| 0 | 11.29 | — |
| 500 | 3.42 | 4.35 |
| 1,000 | 2.76 | 4.08 |
| 1,500 | 3.00 | 3.25 |
| 16,500 | | 2.72 |
| 17,500 | | 2.81 |
| late | ~2.1–2.5 | |

**SFT v1** (init = pretrain_v1)

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

**SFT v5** (init = pretrain_v2 · same sft mix)

| step | train |
|----:|------:|
| 0 | **2.35** (lower than ~2.8-class init on v1-lineage) |
| 500 | 1.32 |
| 1,000 | 1.40 |
| late | ~1.1–1.5 band |

</details>

> 💡 Falling loss = *adapting to chat format*.  
> It does **not** mean benchmark scores instantly rise.  
> v5’s **lower SFT init loss** is a hint that corpus-v2 continued pretrain helps the SFT stage.

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

### 📚 Pretrain

**v1** · ~1.91B tokens · `train 1,893,646,391` · `val 19,127,741` · hash `53f815…`

<details>
<summary><b>🌍 Natural language (EN / KO / JA) · v1 scale</b></summary>

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
<summary><b>💻 Code · v1</b></summary>

<br/>

| Content | Source | Note |
|:--------|:-------|:-----|
| python, c, cpp, java, js, ts, go, rust, … | codeparrot/github-code-clean | ~1,200 files / lang |
| Function-level | Fsoft-AIC/the-vault-function | ~40k rows |
| Commits + diffs | bigcode/commitpack | ~8k rows |

</details>

**v2** · continued pretrain from v1 · **25,000 steps · lr 1e-4** · data hash `4ad58f…`  
Expanded/retokenized corpus v2 (larger fineweb_edu, finemath added, etc.).  
Some sources (vault/commitpack) failed during the download pipeline.

> Note: ko-wiki val loss slightly worsened vs v1 (reported 2.504 → 2.605),  
> but SFT init loss and downstream bench favored the **v2 base**.

### 🗣️ SFT mix evolution

<details>
<summary><b>v1 mix · 321,367 examples</b></summary>

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
  "thinking": "Multiplication first. 3*4=12, 2+12=14.",
  "assistant": "The answer is 14."
}
```

user / system spans are masked from loss — learn the **answer side** only.

</details>

<details>
<summary><b>v3 mix · 446,771 examples (coding/math aug)</b></summary>

<br/>

Added on top of v1 (with multipliers):

| Add | rows × mult |
|:----|------------:|
| gsm8k | 7,473 ×4 |
| mbpp | 474 ×8 |
| codealpaca_20k | 20,016 ×2 |
| orca_math_ko | 25,000 ×1 |
| jamard_gsm8k_ja | 6,672 ×4 |

→ coding full pass **1/5 → 4/5** (no-think)

</details>

<details>
<summary><b>v4 / v5 mix · 516,771 examples (Korean aug) · hash `975c…`</b></summary>

<br/>

Added on top of v3:

| Add | rows |
|:----|----:|
| kullm_v2 | 40,000 |
| ko_wikidata_qa | 30,000 |

Korean share ~**10.3% → 22.5%**.  
**v4 and v5 share this exact `sft.pt`.** Only the init base differs.

</details>

### Mix sketch (v1)

```text
Pretrain tokens
  EN web·stories    ████████
  KO/JA wiki·web    ████████
  Code              ██

SFT examples (v1)
  EN chat           ██████████  ~30%
  Code instruct     ████████    ~23%
  Japanese          ██████      ~20%
  Thinking·math     ██████      ~20%
  Korean            ██          ~7%
```

---

## 4. Scoring

| Item | Detail |
|:-----|:-------|
| Items | **14 prompts × 2 modes = 28 gens** (identical across versions) |
| Modes | 🧠 THINKING on / 💬 normal chat |
| QA · open | 0–5 (full transcript graded · Claude) |
| Coding | Unit tests executed on extracted code |
| v1–v4 gen | temp **`0.7`** · top_p `0.9` · max_new `256` · seed `0` · **1 sample** |
| v5 gen | temp **`0.0` greedy** · top_p `0.9` · max_new `256` · seed `0` (thinking/answer shared budget) |
| **v6 gen** | temp **`0.0` greedy** · top_p `0.9` · max_new `256` · seed `0` · **separate thinking/answer budgets** |
| Dates | v1 `07-09/10` · v2 `07-10` · v3 `07-13` · v4 `07-13` · v5 `07-14` · **v6 `07-15`** (2026) |

> 📌 Coding test totals: v1/v2 sum **17** · v3+ sum **25** (5 cases / problem).  
> Prefer **full-pass N/5** for cross-version coding.

> 🎲 **Sampling noise (v4 lesson)**  
> Under temp 0.7 single-sample, v3↔v4 item flips were large  
> (e.g. en_fact 5→0, code_prime 5→0, code_maxlist 5→0, code_reverse 0→5).  
> **v5 switches to greedy** to kill within-version noise.  
> Cross-version deltas still need the **protocol caveat**.

> 🐛 **Budget-collision bug (v6 lesson)**  
> The first v6 bench run had `max_think` and `max_new` sharing one budget (256),  
> so a long thinking span could leave zero budget for the answer.  
> `generate()` now splits the two budgets; the v6 numbers below are the re-run after the fix.

> 🇰🇷 Korean total: 3 items × 2 modes × 0–5 = **max 30**.

---

## 5. sft_base_v6 · latest

| | |
|:--|:--|
| 🏁 Checkpoint | `ckpt/sft_base_v6.pt` |
| 🧱 Base | `pretrain_base_v2` (same as v5) |
| 📦 SFT data | same 14-file mix as v4/v5 · `prep_sft` fix (thinking tail-preservation: trimmed 47,273 / demoted 17,540) · 9,500 steps |
| 🔧 Inference fix | `infer.py` EOS guard + **separate** thinking/answer decode budgets |
| 🎲 Decode | greedy · temp 0.0 |
| 📁 JSON | [`benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) |

> ⚠️ The first v6 bench run had a bug where `max_think` and `max_new` shared one budget, leaving answers empty.  
> `generate()` now splits the budgets; the results below are the re-run after the fix (see §4).

### 5.1 Scorecards

#### 💬 Normal chat (THINKING off) — overall avg **3.93 / 5**

| Area | Score | Status |
|:-----|:-----:|:------:|
| 🇰🇷 Korean | **2.67 / 5** (5 / 0 / 3) | 🟢 first perfect fact |
| 🇯🇵 Japanese | **3.00 / 5** (5 / 4 / 0) | 🟢 fact+math |
| 🇺🇸 English | **4.33 / 5** (5 / 5 / 3) | 🟢 |
| 💻 Coding | **25/25** tests · **5/5** full pass | 🟢 perfect maintained |

#### 🧠 THINKING on — avg **2.93 / 5** (~6x over v5)

| Metric | Value | Status |
|:-------|:-----:|:------:|
| Avg score | **2.93 / 5** | 🟢 (v5 0.50) |
| Nonempty answers | **14 / 14** | 🟢 all of them (v5 10/14) |
| Coding full pass | **4 / 5** (+prime partial 3/5) | 🟢 first ever (v5 0/5) |
| Korean sum (think) | 2 / 15 | 🔴 still weak |

### 5.2 v5 → v6 highlights

| | Change |
|:--|:-------|
| ✅ | **THINKING handoff bug fixed** — empty answers 4/14 → **0/14** (every prompt gets an answer) |
| ✅ | THINKING avg **0.50 → 2.93** (~6x) |
| ✅ | THINKING coding **0/5 → 4/5** (+prime partial 3/5) — **first working code output in six versions** |
| ✅ | No-think avg rose alongside it: **3.21 → 3.93** |
| ✅ | Korean fact scores **its first-ever perfect 5**: “대한민국의 수도는 서울특별시입니다.” |
| ✅ | Japanese math gets its **first correct answer in both modes** (7 − 2 = 5) |
| ✅ | Korean total **8/30 → 10/30** |
| 🔬 | Same base · same 14-file SFT source mix · **only prep_sft preprocessing + split decode budget changed** — the cleanest single-variable comparison in the project |
| ❌ | Korean arithmetic still collapses (thinking: 5−3=3, no-think: 3+3=9) |
| ❌ | Repetition/degeneration persists late in long thinking and answers (e.g. en_fact thinking repeats “Wait, the capital of France is Paris.”) — carried forward as RL/DPO territory |
| ❌ | Japanese translation still fails in both modes |

---

### 5.3 Q&A · normal chat (THINKING off)

#### 🇰🇷 Korean — avg 2.67

| Item | Score | Expected | What the model did | Note |
|:-----|:-----:|:--------:|:-------------------|:-----|
| Capital | **5** | Seoul | “대한민국의 수도는 서울특별시입니다.” | **first perfect Korean fact score** 🟢 |
| Apples 5−3 | **0** | 2 | “3(사과) + 3(사과) = 9개” | subtraction turned into addition, arithmetic collapse |
| Summary | **3** | one sentence | “오늘 날씨가 매우 좋아서 공원에 산책을 나갔다.” | holds at v5's level |

#### 🇯🇵 Japanese — avg 3.00

| Item | Score | Expected | What the model did | Note |
|:-----|:-----:|:--------:|:-------------------|:-----|
| Capital | **5** | 東京 | 「日本の首都は東京です。」 | perfect |
| Apples 7−2 | **4** | 5 | 「7 - 2 = 5個です」 | **first correct Japanese math** · slight penalty for verb error |
| Translate | **0** | English | just repeats the source sentence | no English |

#### 🇺🇸 English — avg 4.33

| Item | Score | Expected | What the model did | Note |
|:-----|:-----:|:--------:|:-------------------|:-----|
| Capital | **5** | Paris | “The capital of France is Paris.” | perfect |
| Train 60×2 | **5** | 120 | plugs into formula → “120 km” | perfect (v5 scored a partial 3) |
| Summary | **3** | one sentence | “The weather was nice, and the children were having a great time.” | walk/park detail dropped |

---

### 5.4 Q&A · THINKING mode

Through v5 this slot was a list of handoff failures. In v6 the **answer field is filled in all 14/14 cases** — though the reasoning itself is still less accurate than the answers it produces.

| Item | Score | Inside thinking | Answer field | Note |
|:-----|:-----:|:-----------------|:--------------|:-----|
| ko_fact | **1** | Opens with the correct “서울특별시입니다” then derails into repeating “거리 1km” | Distance-themed rambling (only the Seoul entity survives) | retrieved but derailed |
| ko_math | **0** | “5개 − 3개 = 3개” (wrong) | endless repeats of “사과를 3개” | both the math and the answer collapse |
| ko_summary | **1** | restates the source, then repeats an imaginative phrase | “아이들이 뛰어놀고 있는 모습을 상상해 보세요.” | an imperative sentence, not a summary |
| ja_fact | **5** | confused reasoning | 「答えは東京です。」 | **clean correct answer — the textbook handoff success** |
| ja_math | **4** | arithmetic collapses (1.7, 5.7, etc.) | 「答えは5です。」 | reasoning is wrong but the answer is right |
| ja_translate | **0** | repeats English sentences (mistranslated) | just a string of quotation marks | fails |
| en_fact | **4** | loops on “Wait, the capital of France is Paris.” | “The capital of France is **Paris**.” + repetitive explanation | correct answer stated, penalized for verbosity |
| en_math | **0** | “60/2=30”, “30/2=15” | “The answer is 15.” | arithmetic collapse, wrong answer |
| en_summary | **3** | short, reasonable summary | “The answer is that the park was nice and the children were playing.” | valid summary, awkward framing |

> 🔑 **The key change**: in v5, ko_fact / ja_fact / en_summary had correct content inside thinking but an empty answer field.  
> In v6 the same item types now **fill in the answer field** — accuracy still varies, but the “doesn't transfer” failure mode itself is resolved.

---

### 5.5 Coding · both modes compared

Normal chat holds its **5/5 perfect score** from v5. THINKING mode is the first version in the project to **actually emit working code**.

| Problem | Normal chat | THINKING | Note |
|:--------|:-----------:|:--------:|:-----|
| `is_prime` | 5/5 | **3/5** | Thinking mode: hardcodes modulo checks up to 17 — not generalized, so only a partial pass |
| `reverse_string` | 5/5 | **5/5** | `s[::-1]` passes in thinking mode too (thinking span is empty) |
| `factorial` | 5/5 | **5/5** | recursive base case, identical in both modes |
| `is_palindrome` | 5/5 | **5/5** | thinking mode uses `s.strip()`, normal uses `s.lower()` — both pass |
| `find_max` | 5/5 | **5/5** | both modes lean on builtin `max()` |

<details>
<summary><b>✅ THINKING-mode passing code</b></summary>

<br/>

```python
def is_prime(n):  # 3/5 partial — hardcoded modulo checks up to 17
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    # ... repeats through n % 17 == 0 (not a generalized √n trial division)

def reverse_string(s):
    return s[::-1]

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def is_palindrome(s):
    s = s.strip()
    return s == s[::-1]

def find_max(lst):
    max_num = max(lst)
    for num in lst:
        if num > max_num:
            max_num = num
    return max_num
```

</details>

---

### 5.6 Remaining bottlenecks

With the thinking→answer handoff resolved, what's left is now clearly about **accuracy, not format**.

| Issue | Example | Nature |
|:------|:--------|:-------|
| 🇰🇷 Korean arithmetic | computes 5−3 as “3+3=9” (no-think) / “5개−3개=3개” (thinking) | knowledge/arithmetic error — needs data augmentation |
| 🔁 Repetition/degeneration | en_fact thinking repeats the same sentence, ko_fact thinking repeats “1km” | decoding quality — RL/DPO territory |
| 🇯🇵→🇬🇧 Translation | repeats the source or strings together unrelated English sentences | translation ability itself never formed |
| Coding generalization | THINKING-mode `is_prime` hardcodes 17 modulo checks instead of trial division | partial understanding — the general algorithm isn't learned yet |

> 📌 v1's "never closes" → v5's "closes but doesn't hand off" → v6 has moved the bottleneck to **"hands off, but the content is sometimes wrong."**  
> Format learning looks essentially finished at this point.

---

## 6. Earlier versions

For comparison. Expand only when needed.

<details>
<summary><b>📔 sft_base_v5 · 2026-07-14 · pretrain_v2 · greedy</b></summary>

<br/>

📁 [`benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json)

First greedy (deterministic) bench. Swapped the base to `pretrain_base_v2` (SFT mix identical to v4).
First perfect no-think coding score, Korean starts recovering. THINKING handoff (content stuck in thinking, not copied to the answer) emerges as the new bottleneck (fixed in v6).

#### Scores

| Mode | Result |
|:-----|:-------|
| No-think avg | **3.21 / 5** |
| 🧠 THINKING avg | **0.50 / 5** · nonempty **10/14** |
| 🇰🇷 no-think | **1.67/5** (2 / 0 / 3) |
| 🇯🇵 no-think | **1.33/5** (4 / 0 / 0) |
| 🇺🇸 no-think | **3.67/5** (5 / 3 / 3) |
| 💻 no-think | full **5/5** · tests 25/25 (first perfect) |
| 🇰🇷 total (T+NT) | **8/30** |

#### Coding (no-think · greedy)

| Problem | Tests | Note |
|:--------|:-----:|:-----|
| is_prime | 5/5 | √n trial division |
| reverse_string | 5/5 | `s[::-1]` |
| factorial | 5/5 | recursive base case |
| is_palindrome | 5/5 | lower + reverse |
| find_max | 5/5 | builtin `max()` |

#### Highlights

| | |
|:--|:--|
| ✅ | Coding full pass 3/5 → 5/5 (first perfect · deterministic greedy) |
| ✅ | Korean total 1/30 → 8/30 (first summary success · correct number in think-math) |
| ⚠️ | Same SFT mix, different base + greedy switch → not "SFT alone got better" |
| ❌ | THINKING coding still 0/5 · correct content inside thinking doesn't transfer to the answer field (handoff bottleneck discovered) |

</details>

<details>
<summary><b>📕 sft_base_v4 · 2026-07-13 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v4.json`](ckpt/benchmark_sft_base_v4.json)

Korean SFT augmentation (+kullm-v2 40k, +ko_wikidata_QA 30k · KO share 10.3%→22.5%).  
**Lateral overall.** Bench Korean barely recovered.

#### Scores

| Mode | Result |
|:-----|:-------|
| No-think avg | **2.07 / 5** |
| 🧠 THINKING avg | **0.50 / 5** · nonempty **11/14** |
| 🇰🇷 no-think | **0.33/5** (0 / 0 / 1) |
| 🇯🇵 no-think | **1.67/5** (5 / 0 / 0) |
| 🇺🇸 no-think | **2.67/5** (0 / 5 / 3) |
| 💻 no-think | full **3/5** · tests 15/25 |
| 🇰🇷 total (T+NT) | **1/30** |

#### Coding (no-think · temp 0.7)

| Problem | Tests | Note |
|:--------|:-----:|:-----|
| is_prime | 0/5 | empty body → IndentationError — possible sampling noise |
| **reverse_string** | **5/5** | `s[::-1]` (fixes v3 print side-effect) |
| factorial | 5/5 | recursive base case |
| is_palindrome | 5/5 | alnum filter + lower · more robust |
| find_max | 0/5 | variable shadowing bug |

#### Highlights

| | |
|:--|:--|
| ✅ | First perfect Korean capital sentence **inside thinking** (`서울특별시`) — answer empty |
| ✅ | First thinking-mode math perfect: en_math **5/5** |
| ⚠️ | Large item flips vs v3 → **temp 0.7 single-sample noise** dominates |
| ❌ | Korean bench 1/30 — SFT share alone insufficient → pretrain lever recommended |

</details>

<details>
<summary><b>📘 sft_base_v3 · 2026-07-13 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json)

Coding-aug goal achieved. Korean category collapsed.

#### Scores

| Mode | Result |
|:-----|:-------|
| No-think avg | **2.57 / 5** |
| 🧠 THINKING avg | **0.21 / 5** · nonempty **12/14** |
| 🇰🇷 no-think | **0.00/5** |
| 🇯🇵 no-think | **1.33/5** |
| 🇺🇸 no-think | **4.00/5** |
| 💻 no-think | tests **20/25** · full **4/5** |
| 🇰🇷 total | **0/30** |

#### Coding (no-think)

| Problem | Tests | Note |
|:--------|:-----:|:-----|
| is_prime | 5/5 | √n trial division |
| reverse_string | 0/5 | body OK · top-level `print` → NameError |
| factorial | 5/5 | recursive base (fixes v2 infinite recursion) |
| is_palindrome | 5/5 | lower + reverse (fixes v2 always-True) |
| find_max | 5/5 | hand-rolled loop |

#### No-think glance

| Item | Score | Note |
|:-----|:-----:|:-----|
| ko_fact / math / summary | 0 / 0 / 0 | Korean collapse |
| ja_fact / math / translate | **4** / 0 / 0 | 東京 correct |
| en_fact / math / summary | **5** / **5** / 2 | Paris·120 recovered |
| coding full pass | 4/5 | reverse only fail |

</details>

<details>
<summary><b>📗 sft_base_v2 · 2026-07-10 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json)

#### Scores

| Mode | Result |
|:-----|:-------|
| 🧠 THINKING | ans **13/14** · avg **0.71/5** |
| 🇰🇷 no-think | **0.33/5** |
| 🇯🇵 no-think | **2.00/5** |
| 🇺🇸 no-think | **1.00/5** |
| 💻 no-think | tests **9/17** · full **1/5** (`find_max` → builtin `max()`) |

#### v1 → v2

| | |
|:--|:--|
| ✅ | THINKING close largely fixed (0/14 → 13/14) |
| ✅ | First thinking-mode perfect: `ja_fact` **5/5** |
| ⚠️ | EN fact = Burgundy hallucination (temp 0.7 · single sample) |
| ❌ | Coding still weak · thinking produces no code |

#### Coding (no-think)

| Problem | Tests | Note |
|:--------|:-----:|:-----|
| is_prime | 2/5 | partial |
| reverse_string | 2/3 | partial |
| factorial | 0/3 | infinite recursion etc. |
| is_palindrome | 2/3 | always-True family |
| **find_max** | **3/3** | delegates to `max()` |

</details>

<details>
<summary><b>📙 sft_base_v1 · 2026-07-09/10 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json)

#### Scores

| Mode | Result |
|:-----|:-------|
| 🧠 THINKING | ans **0/14** · avg 0.0 🔴 |
| 🇰🇷 no-think | **1.33/5** |
| 🇯🇵 no-think | **0.33/5** |
| 🇺🇸 no-think | **2.67/5** |
| 💻 no-think | tests **4/17** · full **1/5** |

> 🧠 Generation ends before the closing tag → runtime empties the answer.  
> Prefer **no-thinking** mode for this checkpoint.

#### Pretrain vs SFT v1

| | Pretrain | SFT v1 |
|:--|:---------|:-------|
| Scores | essentially all 0 | scores appear in no-think |
| Coding | 0/34 | 4/17 |
| Behavior | echo · loops | tries format (content often wrong) |

5,000 SFT steps alone move the model from **cannot follow at all → tries the format**.

#### Coding (no-think)

| Problem | Tests | Score | Note |
|:--------|:-----:|:-----:|:-----|
| is_prime | 0/5 | 0 | degrades to even-check |
| reverse_string | 0/3 | 0 | only `.lower()` |
| factorial | 0/3 | 1 | recursive skeleton, no n==0 |
| **is_palindrome** | **3/3** | **5** | ✅ only clean full pass |
| find_max | 1/3 | 1 | lucky single case |

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
<summary><b>📓 pretrain_base_v1 · chat ~0</b></summary>

<br/>

📁 [`benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json)

| | Result |
|:--|:-------|
| Chat response | almost none · ~0 / 28 |
| Coding | 0 / 34 (both modes) |
| Behavior | echo · repetition loops · rephrased questions |

Before SFT, **chat format itself is untrained**.  
After v1 SFT, “tries the format” first shows up on the bench.

> 📭 **No standalone chat bench for pretrain_base_v2 yet.**  
> v2’s effect is observed only downstream via **sft_base_v5 · sft_base_v6**.

</details>

---

## 7. Takeaways · next

### ✅ What worked across versions

- Pretrain loss 11 → ~2, SFT loop healthy · pipeline validated
- v1: “format attempt” shows up on the bench
- v2: THINKING **close** mostly restored (0/14 → 13/14)
- v3: no-think **coding 4/5** — data-aug goal hit
- v4: higher KO SFT share · first clean Korean capital inside thinking (handoff fails)
- v5: no-think coding 5/5 · avg 3.21 · KO 8/30 · greedy-reproducible
- **v6: THINKING handoff fixed · thinking avg 0.50→2.93 · thinking coding first pass (4/5) · no-think avg rose too, 3.21→3.93** — best so far
- Continued pretrain (v2) lifts downstream **even with the same SFT mix**
- prep_sft preprocessing + a split decode budget alone resolved the handoff problem (base and SFT source untouched)

### 🚧 Still blocked (priority order)

> ✅ The two top priorities through v5 — “force the THINKING handoff” and “train THINKING to emit code” — are **resolved as of v6** (prep_sft fix + split decode budget). What's below is the next layer.

1. 🇰🇷 Korean arithmetic/fact accuracy — format works, but the math is often wrong (thinking: 5−3=3, no-think: 3+3=9)
2. 🔁 Repetition/degeneration in long generations (both thinking and answers) — RL/DPO territory
3. 🇯🇵→🇬🇧 Translation ability never formed — just repeats the source
4. Coding generalization — THINKING-mode `is_prime` hardcodes modulo checks instead of √n trial division
5. Far from production quality

### 🧭 Next

| Pri | Action |
|:---:|:-------|
| 1 | More Korean fact/arithmetic augmentation (pretrain + SFT together) — the one remaining "accuracy" axis |
| 2 | Suppress repetition/degeneration — DPO / correction SFT / RLVR (format works, now it's a quality pass) |
| 3 | Japanese→English translation data augmentation |
| 4 | Coding generalization — teach general logic like √n trial division inside THINKING mode too |
| 5 | Add standalone pretrain_v2 chat bench (isolate base effect, still unmeasured) |
| 6 | Keep the **same 14 prompts** for version comparison |

### 📐 Interpretation guide (once more)

```text
❌  “More SFT ⇒ v6 > v5 > v4 > v3”
✅  “Separate base fork + protocol change + preprocessing change”
    · v3→v4: same base · same temp0.7 · SFT mix change (lateral + noise)
    · v4→v5: same SFT mix · different base · greedy switch (best scores)
    · v5→v6: same base · same SFT mix · prep_sft fix + split decode budget (handoff resolved)
```

---

### 📁 Source files

| Path | Content |
|:-----|:--------|
| [`ckpt/benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) | ⭐ **latest** SFT v6 (handoff fixed · greedy · pretrain_v2) |
| [`ckpt/benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) | SFT v5 (greedy · pretrain_v2) |
| [`ckpt/benchmark_sft_base_v4.json`](ckpt/benchmark_sft_base_v4.json) | SFT v4 |
| [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) | SFT v3 |
| [`ckpt/benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json) | SFT v2 |
| [`ckpt/benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json) | SFT v1 |
| [`ckpt/benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json) | Pretrain v1 |
| — | pretrain_v2 chat bench **not measured** |

> Weights (`.pt`) and raw corpora are not included in the public repo.

---

<div align="center">

[README](README.en.md) · [ARCHITECTURE](ARCHITECTURE.en.md) · [POST-TRAINING](POST-TRAINING.en.md)

</div>
