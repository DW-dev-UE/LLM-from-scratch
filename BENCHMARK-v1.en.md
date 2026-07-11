[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.en.md)

---

This doc is a record of actually training **base ~327M**.
The point is not a pretty score table. It is:

- how training was run
- what data was used
- what the model answered to each question

### In one line first

| | Result |
|:--|:-------|
| **Pretrain only** | Almost no reaction to chat prompts (~0 / 28) |
| **After SFT** | Tries to follow instruction format; content often wrong |
| **Thinking mode** | Empty-answer bug (0 answers / 14) → evaluate normal chat instead |

Not production quality yet.
It is the first snapshot that shows the **pipeline actually runs**, and what SFT changes.

---

## 1. Training process

### Model card

| Item | Value |
|:-----|:------|
| Family | Decoder-only Transformer (LLaMA-style parts) |
| Params | ~**326.7M** (`base` preset) |
| Depth · width | 24 layers · d_model 1024 |
| Attention | GQA (Q 16 · KV 4) + RoPE |
| FFN | SwiGLU |
| Norm | RMSNorm (Pre-Norm) |
| Other | weight tying, no bias |

```text
input tokens
   │
embedding ───────────────────────────┐
   │                                 │ (weight tying)
[ Block × 24 ]                       │
  RMSNorm → Attention → +            │
  RMSNorm → SwiGLU    → +            │
   │                                 │
RMSNorm                              │
   │                                 │
LM Head ◄────────────────────────────┘
   │
next-token probs
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
  ④ Benchmark (this doc)
```

| Stage | Checkpoint | steps | Init | Data fingerprint |
|:------|:-----------|------:|:-----|:-----------------|
| Pretrain | `pretrain_base_v1` | 18,000 | from scratch | `train.bin` · `53f815d47abc4887` |
| SFT | `sft_base_v1` | 5,000 | continued from pretrain | `sft.pt` · `bbc2211091309d3c` |

Shared: AdamW (β 0.9 / 0.95, weight decay 0.1), grad clip 1.0, warmup + cosine, CUDA.

### How loss came down

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

**SFT** (continued from the pretrain checkpoint)

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

Loss going down is a signal of “adapting to chat format.”
It does not mean benchmark scores instantly improve. The Q&A below shows that gap.

---

## 2. Datasets

Raw multi-GB files are not in the GitHub repo.
They were downloaded from public HuggingFace (etc.) and prepared with `data.py`.

### Tokenizer

| | |
|:--|:--|
| Type | Byte-level BPE |
| Size | **64,000** vocab |
| Sample | fineweb, wiki (ko/ja), tinystories, etc. ~900MB |
| Specials | role tokens, `THINKING` span tokens |

### Pretrain · ~1.91B tokens

`train 1,893,646,391` · `val 19,127,741`

<details>
<summary><b>Natural language (EN / KO / JA)</b> · expand</summary>

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
<summary><b>Code</b> · expand</summary>

<br/>

| Content | Source | Note |
|:--------|:-------|:-----|
| python, c, cpp, java, js, ts, go, rust, … | codeparrot/github-code-clean | ~1,200 files per language |
| Function-level | Fsoft-AIC/the-vault-function | ~40k rows |
| Commits + diffs | bigcode/commitpack | ~8k rows |

</details>

<details>
<summary><b>SFT · 321,367 examples</b> · expand</summary>

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

Example training line:

```json
{
  "user": "2 + 3 * 4 = ?",
  "thinking": "Multiply first. 3*4=12, then 2+12=14.",
  "assistant": "The answer is 14."
}
```

User / system spans are masked out of the loss. Goal is learning the **answer side**, not echoing the question.

</details>

### Mix (rough feel)

```text
Pretrain tokens
  EN web · stories   ████████
  KO/JA wiki · web   ████████
  code               ██

SFT examples (approx.)
  EN chat            ██████████  ~30%
  code instruct      ████████    ~23%
  Japanese           ██████      ~20%
  thinking · math    ██████      ~20%
  Korean             ██          ~7%
```

---

## 3. How scoring worked

| Item | Detail |
|:-----|:-------|
| Items | 14 prompts × 2 modes = 28 generations |
| Modes | thinking on / off |
| QA · open | 0–5 (full transcript read) |
| Coding | unit-test pass counts |
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

In thinking mode, generation ends before the close tag is written.
The inference code then leaves the answer empty, so **this checkpoint is better used in normal chat mode**.

---

## 5. Pretrain vs SFT

| | Pretrain | After SFT |
|:--|:---------|:----------|
| Overall scores | essentially all zeros | scores appear in normal mode |
| Coding tests | 0 / 34 | 4 / 17 (normal mode) |
| Behavior | echo the question, loop the same phrase | tries answer format (content often wrong) |

With only 5,000 SFT steps, the model moves one step from  
**“cannot follow at all → tries the format.”**  
Quality above that is still far away.

---

## 6. Questions and answers

Scores are **sft_base_v1 · normal chat** unless noted.  
Thinking mode is empty for most items, so it is only mentioned briefly when needed.

---

### Korean

#### Fact · capital

| | |
|:--|:--|
| **Score** | 2 / 5 |
| **Prompt** | 대한민국의 수도는 어디인가요? |
| **Expect** | Seoul |
| **Answer** | 서울은 수도로서, 수도의 역할을 하는 도시인 수도는 서울이 아닌 다른 지역의 수도입니다… 서울은 수도이며, 수도는 따로 없습니다. |
| **Note** | “Seoul” appears, but the sentences contradict themselves |

Thinking mode: empty answer field. (a fragment of thinking briefly shows “서울”)

---

#### Math · apples

| | |
|:--|:--|
| **Score** | 1 / 5 |
| **Prompt** | 철수는 사과를 5개 가지고 있었는데, 3개를 먹었습니다. 남은 사과는 몇 개인가요? |
| **Expect** | 2 |
| **Answer** | 사과는 3개의 사과를 가지고 있습니다. |
| **Note** | No subtraction |

---

#### Summary

| | |
|:--|:--|
| **Score** | 1 / 5 |
| **Prompt** | 다음 문장을 한 문장으로 요약해 주세요. *「오늘 날씨가 매우 좋아서 공원에 산책을 나갔다. 사람들이 많이 나와 있었고, 아이들은 뛰어놀고 있었다.」* |
| **Answer** | Nearly rewrites the source as-is |
| **Note** | Echo, not summary |

---

### Japanese

#### Fact · capital

| | |
|:--|:--|
| **Score** | 0 / 5 |
| **Prompt** | 日本の首都はどこですか? |
| **Expect** | 東京 |
| **Answer** | 日本は世界でも最も人口の多い国であり、人口が多い国の一つです… |
| **Note** | Talks about population, not capital |

---

#### Math · apples

| | |
|:--|:--|
| **Score** | 1 / 5 |
| **Prompt** | 太郎はりんごを7個持っていて、2個食べました。残りは何個ですか? |
| **Expect** | 5 |
| **Answer** | 太郎は2個食べています。 |
| **Note** | Restates the problem only |

---

#### Translate

| | |
|:--|:--|
| **Score** | 0 / 5 |
| **Prompt** | 「今日は天気が良いので公園に散歩に行った。」を英語に翻訳してください。 |
| **Answer** | 「公園で散歩をするのは素晴らしいね。今日は天気が良いので…」(stays Japanese) |
| **Note** | Ignores the English translation request |

---

### English

#### Fact · capital

| | |
|:--|:--|
| **Score** | **4 / 5** |
| **Prompt** | What is the capital of France? |
| **Expect** | Paris |
| **Answer** | The capital of France is Paris. The capital of France is also located in the heart of France, which is part of the Loire Valley… |
| **Note** | Core is correct; geography after that gets a bit weird. **Best factual signal in this run** |

Pretrain only: rewrites the question as a Germany capital question → 0.

---

#### Math · train

| | |
|:--|:--|
| **Score** | 1 / 5 |
| **Prompt** | If a train travels at 60 km/h for 2 hours, how far does it go? |
| **Expect** | 120 |
| **Answer** | … 60 km/h * 0.5 hours = 60 km … |
| **Note** | Time–distance relation is wrong |

---

#### Summary

| | |
|:--|:--|
| **Score** | 3 / 5 |
| **Prompt** | Summarize in one sentence: *“The weather was nice today so I went for a walk in the park, and there were many people and children playing.”* |
| **Answer** | Stretches into four sentences; invents swings, flowers, etc. |
| **Note** | Mood is similar; “one sentence” instruction fails |

---

## 7. Coding items

Function signatures only; generated code run against unit tests.  
Below is **normal chat mode** (thinking-mode coding all failed).

| Task | Tests | Score | One-line note |
|:-----|:-----:|:-----:|:--------------|
| is_prime | 0 / 5 | 0 | Degrades into even-check |
| reverse_string | 0 / 3 | 0 | Only `.lower()` then stops |
| factorial | 0 / 3 | 1 | Recursive skeleton, no zero base case |
| **is_palindrome** | **3 / 3** | **5** | **Only clean pass** |
| find_max | 1 / 3 | 1 | Variable shadowing; one lucky case |

### Pass (is_palindrome)

```python
def is_palindrome(s):
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]
```

### Fail (is_prime → effectively is_even)

```python
def is_prime(n):
    if n <= 1:
        return False
    return n % 2 == 0
```

### Fail (reverse_string)

```python
def reverse_string(s):
    return s.lower()
```

---

## 8. Takeaways · next steps

### What worked

- Pretrain loss 11 → ~2, SFT 2.67 → ~1.05 → **training loop is real**
- After SFT, “tries the format” shows up in the bench
- One English fact item and one palindrome coding pass are clear signals

### Still stuck

1. Thinking close-tag failure  
2. Weak KO/JA facts and arithmetic  
3. Summary / translate instruction ignored  
4. Coding full pass 1 / 5  

### Next

- SFT with more thinking-close patterns (v2 direction)  
- More coding · arithmetic share; RLVR if needed  
- DPO / correction SFT  
- Keep the **same 14 prompts** for version comparison  

---

### Local source files

| Path | Content |
|:-----|:--------|
| `llm/ckpt/benchmark_sft_base_v1.json` | Full SFT items |
| `llm/ckpt/benchmark_pretrain_base_v1.json` | Pretrain compare |
| `llm/DATA_SOURCES.md` | Dataset source list |

Weights and raw corpora are not in the public repo.

---

[README](README.en.md) · [ARCHITECTURE](ARCHITECTURE.en.md) · [POST-TRAINING](POST-TRAINING.en.md)
