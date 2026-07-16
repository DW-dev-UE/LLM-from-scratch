> **This repo is actively updated.**  
> Training results, benchmarks, and docs change over time. Prefer the latest commit. Issues and PRs welcome.

> Before we start: this writing may not be the most polished for readability.
> You could ask an AI to "make this more readable," but then the record of how I thought and what process I went through can disappear.
> Learning with AI is fine, but rewriting sentences myself to organize my knowledge matters more.

---

Alright, let's begin.

## What this project does

A from-scratch experiment to build a **general-purpose LLM** on NVIDIA CUDA, and AI that can be used for real work like coding and documents.

- Tokenizer training
- Model architecture (Decoder-only Transformer)
- Pretrain · SFT · DPO · GRPO
- KV-cache inference · thinking mode (THINKING)
- Human-feedback UI · promotion gate

Fine-tuning an existing open-source LLM would be much faster. Even so, I rebuilt it from scratch for one reason only.

If you never implement an LLM’s internals yourself, you only ever borrow other people’s code.

> If you never implement an LLM’s internals yourself, you only ever borrow other people’s code.

So this repository does **not** depend on external commercial or open LLM weights.

---

## Docs

| Doc | Contents |
|:----|:---------|
| **README** (this file) | Philosophy, parameter strategy, structure, how to run |
| [GLOSSARY](GLOSSARY.en.md) | Terms like Transformer, RoPE, DPO |
| [ARCHITECTURE](ARCHITECTURE.en.md) | Model · tokenizer · train · infer |
| [POST-TRAINING](POST-TRAINING.en.md) | Post-deploy human feedback loop |
| [BENCHMARK v1](BENCHMARK-v1.en.md) | Base model benchmark (training process · Q&A) |

[한국어](README.md) · [English](README.en.md) · [日本語](README.ja.md)

---

## 1. Architecture

### Transformer architecture

> This section describes how an LLM is trained.

To train an LLM you need a dataset. Datasets are also called a “corpus.” In this walkthrough the corpus is “I like cats,” shown with five demo tokens aligned to the original Korean example: I · cat · OBJ · like · do (OBJ stands in for an object marker; English word order is SVO, but we keep the same five-slot layout so the numbers match).

---

#### 1) Tokenizer

The tool that turns the corpus into a form a computer can work with is the **tokenizer**.

"I like cats" (demo split) → <kbd>I</kbd> <kbd>cat</kbd> <kbd>OBJ</kbd> <kbd>like</kbd> <kbd>do</kbd>

That split produces a dictionary called **vocab**.

```text
vocab = {"I": 1, "cat": 2, "OBJ": 3, "like": 4, "do": 5}
```

token ids: `[1, 2, 3, 4, 5]`

---

#### 2) Embedding table

For each token we fill a row with random real numbers.

| # | token | d1 | d2 | d3 | d4 |
|--:|-------|---:|---:|---:|---:|
| 1 | I | 0.12 | -0.53 | 0.33 | 0.90 |
| 2 | cat | -0.51 | 0.30 | -2.10 | 0.87 |
| 3 | OBJ | 0.05 | -0.44 | 1.32 | -0.06 |
| 4 | like | 0.71 | 0.18 | -0.29 | 0.55 |
| 5 | do | -0.33 | 0.92 | 0.14 | -0.78 |

The width of each row is the **embedding dimension** (d_model). More dimensions give more room to describe a token, so expressiveness goes up.

How many bytes store each real number is a matter of **precision** (FP32, FP16, and so on).

---

#### 3) Building the training problem

Shift the token sequence by one position to make input / target pairs.

Input: `[1, 2, 3, 4]` = "I cat OBJ like"<br>
Target: `[2, 3, 4, 5]` = "cat OBJ like do"

<kbd>I</kbd> → ? (answer: cat)<br>
<kbd>I</kbd> <kbd>cat</kbd> → ? (answer: OBJ)

---

#### 4) Normalization

x = embedding values. For the problem [I, cat, OBJ, like], x is the embedding of “like”.

x = <kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd>

After normalization, N:

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>

---

#### 5) Attention

This is the step that mixes context — the core of the Transformer architecture from here on.<br>
A plain bigram model cannot see context. The same “like” looks the same whether the previous context was “cat OBJ” or something else.

> 📌 **Three random matrices project N into three views**

Problem 4 — tokens `[1, 2, 3, 4]` = "I cat OBJ like", targets `[2, 3, 4, 5]` = "cat OBJ like do".<br>
The position we are predicting from is 「like」.

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>

Query `Q = N · W_Q`. W_Q, W_K, W_V start random at the beginning of training and are adjusted as training proceeds.

K(I) · Q ÷ √4 = 0.1<br>
K(cat) · Q ÷ √4 = 2.0<br>
K(OBJ) · Q ÷ √4 = −0.6<br>
K(like) · Q ÷ √4 = 0.5

softmax(<kbd>0.1</kbd> <kbd>2.0</kbd> <kbd>-0.6</kbd> <kbd>0.5</kbd>) = <kbd>0.10</kbd> <kbd>0.69</kbd> <kbd>0.05</kbd> <kbd>0.15</kbd>

「like」 attended to 「cat」 at 69% — that is why the same word becomes a different vector depending on prior context.

Weighted sum of V: `0.10·V(I) + 0.69·V(cat) + 0.05·V(OBJ) + 0.15·V(like)`

Attention output = <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd> — a new vector with context mixed in.

---

#### 6) Residual connection

Add x and the attention output to get new v. This keeps the original meaning while stacking context on top.

<kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd> + <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd><br>
= <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd>

Still just numbers, but we are ready to capture context and learn.

---

### Block

From here we are in the block architecture.

#### ① Normalization

Normalize new v.

N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd>

---

#### ② FFN step 1 — expand with W₁

Expand N₂ by a fixed multiple. GPT-2 expands by 4×, so this example does the same.<br>
Attention output is only a weighted sum of V values — mixing and averaging — but the expanded tensor acts more like a set of “pattern detectors.”

Expand: `U = N·W₁ + b₁` (4×4 matrix · 4×16 matrix = 4×16; b₁ is a 16-dim bias, initialized to 0 so omitted)

> **Legend**: <span style="color:#1d9e75">■</span> will pass (positive) · <span style="color:#d85a30">■</span> will be blocked (negative) · <span style="color:#888780">■</span> near zero

**Table A — before GELU (U)**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-1.53</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#1d9e75">1.96</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#d85a30">-2.58</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.29</span> | <span style="color:#d85a30">-1.37</span> | <span style="color:#d85a30">-0.28</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-0.26</span> | <span style="color:#d85a30">-0.65</span> | <span style="color:#1d9e75">2.15</span> | <span style="color:#1d9e75">1.48</span> |
| cat | <span style="color:#d85a30">-1.96</span> | <span style="color:#d85a30">-0.90</span> | <span style="color:#1d9e75">1.93</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-1.33</span> | <span style="color:#1d9e75">0.89</span> | <span style="color:#d85a30">-2.79</span> | <span style="color:#1d9e75">0.39</span> | <span style="color:#1d9e75">2.57</span> | <span style="color:#d85a30">-1.80</span> | <span style="color:#d85a30">-0.58</span> | <span style="color:#1d9e75">0.77</span> | <span style="color:#d85a30">-1.60</span> | <span style="color:#1d9e75">1.57</span> | <span style="color:#d85a30">-1.06</span> | <span style="color:#d85a30">-2.67</span> |
| OBJ | <span style="color:#1d9e75">2.35</span> | <span style="color:#1d9e75">0.64</span> | <span style="color:#d85a30">-2.43</span> | <span style="color:#d85a30">-2.82</span> | <span style="color:#1d9e75">2.28</span> | <span style="color:#d85a30">-1.91</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">0.50</span> | <span style="color:#d85a30">-2.27</span> | <span style="color:#1d9e75">0.94</span> | <span style="color:#1d9e75">0.31</span> | <span style="color:#d85a30">-0.76</span> | <span style="color:#1d9e75">0.90</span> | <span style="color:#d85a30">-1.31</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.10</span> |
| like | <span style="color:#d85a30">-0.88</span> | <span style="color:#d85a30">-2.45</span> | <span style="color:#1d9e75">1.63</span> | <span style="color:#1d9e75">3.40</span> | <span style="color:#d85a30">-1.20</span> | <span style="color:#1d9e75">2.67</span> | <span style="color:#d85a30">-3.24</span> | <span style="color:#1d9e75">2.07</span> | <span style="color:#1d9e75">0.79</span> | <span style="color:#d85a30">-0.62</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.21</span> | <span style="color:#1d9e75">0.82</span> | <span style="color:#d85a30">-0.63</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-1.77</span> |

---

#### ③ FFN step 2 — GELU

Without GELU, `W₂(W₁·N) = (W₂W₁)·N`, which is mathematically the same as multiplying by a single matrix, so the 16-wide expansion is wasted.<br>
Nonlinearity is what enables conditional behavior: “turn this feature on, turn that one off.”

`GELU(x) = x × Φ(x)` — passes the input only by the fraction Φ(x).<br>
Φ(x) is the probability that a standard normal draw is ≤ x (between 0 and 1), so large x passes near fully and small x is driven toward 0.

If x is -2.6, pass rate 0.5% = <span style="color:#d85a30">-0.01</span> (effectively blocked)<br>
If x is 3.10, pass rate 99.9% = <span style="color:#1d9e75">3.10</span> (effectively unchanged)

Φ(x) is **not** “the probability of the next token.” It is only a fixed mathematical gate on how strongly activation x is allowed through. Next-token probabilities first appear at softmax.

**Table B — after GELU**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | <span style="color:#1d9e75">1.37</span> | <span style="color:#d85a30">-0.10</span> | <span style="color:#d85a30">-0.13</span> | <span style="color:#1d9e75">1.91</span> | <span style="color:#1d9e75">2.24</span> | <span style="color:#d85a30">-0.13</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.11</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#d85a30">-0.11</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#d85a30">-0.10</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">2.12</span> | <span style="color:#1d9e75">1.37</span> |
| cat | <span style="color:#d85a30">-0.05</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#1d9e75">0.72</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">0.25</span> | <span style="color:#1d9e75">2.55</span> | <span style="color:#d85a30">-0.06</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#1d9e75">0.60</span> | <span style="color:#d85a30">-0.09</span> | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-0.15</span> | <span style="color:#d85a30">-0.01</span> |
| OBJ | <span style="color:#1d9e75">2.32</span> | <span style="color:#1d9e75">0.47</span> | <span style="color:#d85a30">-0.02</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-0.05</span> | <span style="color:#1d9e75">1.77</span> | <span style="color:#1d9e75">0.34</span> | <span style="color:#d85a30">-0.03</span> | <span style="color:#1d9e75">0.78</span> | <span style="color:#1d9e75">0.19</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">0.73</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">3.10</span> |
| like | <span style="color:#d85a30">-0.17</span> | <span style="color:#d85a30">-0.02</span> | <span style="color:#1d9e75">1.55</span> | <span style="color:#1d9e75">3.39</span> | <span style="color:#d85a30">-0.14</span> | <span style="color:#1d9e75">2.66</span> | <span style="color:#888780">0.00</span> | <span style="color:#1d9e75">2.03</span> | <span style="color:#1d9e75">0.62</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.09</span> | <span style="color:#1d9e75">0.65</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#d85a30">-0.07</span> |

Table A 「like」 d7 = <span style="color:#d85a30">-3.24</span> → Table B <span style="color:#888780">0.00</span> — almost wiped out; you can see GELU “blocking” a strong negative.<br>
Conversely d4 = <span style="color:#1d9e75">3.40</span> → <span style="color:#1d9e75">3.39</span> almost fully “passed.”

📌 From here we keep tracking only the prediction position 「like」.<br>
N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd> → after GELU (Table B row 「like」):<br>
`-0.17 -0.02 1.55 3.39 -0.14 2.66 0.00 2.03 0.62 -0.17 0.00 -0.09 0.65 -0.17 -0.16 -0.07`

---

#### ④ Project down

Matrix multiply by W₂ (16×4). “Summarize 16 signals by a weighted sum back into 4 dims.”<br>
Result: <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd>

---

#### ⑤ FFN output

This is the end of the block.<br>
new v <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd> + FFN output <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd><br>
= <kbd>7.86</kbd> <kbd>0.78</kbd> <kbd>-3.57</kbd> <kbd>6.56</kbd>

Stacking many such blocks is a multi-layer Transformer; GPT-2 uses 12 layers.<br>
It runs as `x → block1 → h₁ → block2 → h₂ → … → blockN → h_N → (exit matmul → logit → softmax → loss)`.

That is the principle behind an LLM that can use context. Next the model must pick the next word and turn that score into a probability.

---

### 📌 logit: exit matrix multiply

logit = raw score that each word is the “next token.” Dot product of h with each word embedding.

| token | logit | note |
|---|---:|---|
| like | 10.36 | ← largest (model’s top guess) |
| cat | 9.43 | |
| I | 5.26 | |
| OBJ | -5.06 | |
| do | -7.49 | ← smallest (but this is the correct answer) |

Higher means the model believes that word more, but the key answer 「do」 was correct. (Weights are random, so this outcome is expected.)<br>
We need softmax → probabilities before we can compute loss.

softmax = map logits (any range) → all positive probabilities that sum to 1.00

| token | logit | gap(=logit-10.36) | e^gap | prob |
|---|---:|---:|---:|---:|
| like | 10.36 | 0.00 | 1.0000 | 71.4% |
| cat | 9.43 | -0.93 | 0.3946 | 28.2% |
| I | 5.26 | -5.10 | 0.0061 | 0.4% |
| OBJ | -5.06 | -15.42 | 0.0000002 | ≈0% |
| do | -7.49 | -17.85 | 0.0000000177 | ≈0% |

sum = 1.4007

---

That finishes the forward pass. Loss is `−ln(prob of do) = −ln(0.0000000126) ≈ 18.2`. That is quite high; next we improve the weights with backpropagation.

### First plan

The first plan was simple.

| Step | Idea |
|:----:|:-----|
| **1** | Build an LLM first with natural-language data only |
| **2** | Stack coding-specialized data on top |
| **3** | Refine with human reinforcement learning |

Even for a coding AI, questions come in natural language, so I thought **language understanding had to be finished first**.

### What the papers show

[Llama 3](https://ar5iv.labs.arxiv.org/html/2407.21783) does not use that sequence.  
Pretraining already mixes domains in one mix.

| Share (approx.) | Data |
|:---------------:|:-----|
| 50% | Natural language |
| 25% | Math / reasoning |
| 17% | Code |
| 8% | Multilingual |

Post-training only strengthens instruction, preference, coding, tool-use.  
There is **no** “finish language, then bolt on code” stage.

[DeepSeek-Coder](https://arxiv.org/html/2401.14196v1) goes the same way.  
It trained from the start on ~87% code + ~10% code-related English + ~3% other,  
and still ranked near the top among open code LLMs at the time despite only ~13% natural language.

### Pipeline we use now

| | Order |
|:--|:------|
| Old idea | language only → code only → instruction |
| **Current** | **mixed pretrain** → code-heavy continue → instruction → preference |

```text
  language + code + math + docs
            │
            ▼
   base pretraining
            │
            ▼
  code-heavy continued pretrain
            │
            ▼
    instruction tuning (SFT)
            │
            ▼
   preference (DPO / GRPO …)
```

### Mix guide (300M ~ 1B)

| Kind | Share | Examples |
|:-----|:-----:|:---------|
| Raw source code | 35–45% | source files |
| Code-adjacent | 15–20% | README, issue, PR, commit |
| Technical language | 20–30% | EN / KO tech docs |
| Math / reasoning | 5–10% | math, logic |
| Logs / tests / diffs | 5–10% | logs, tests, diffs |
| Korean / bilingual | 5–10% | instructions, bilingual |

---

## 2. Scale parameters slowly

Parameters are model size. (Terms: [GLOSSARY](GLOSSARY.en.md))  
Bigger can help, but this project grows in stages like **300M → 1B → 3B**.

[Chinchilla](https://arxiv.org/abs/2203.15556) in short:

> Under a fixed compute budget, grow **model size and token count together**.

Large models of that era often had too little data for their size,  
and the smaller Chinchilla (70B, 1.4T tokens) sometimes scored better than larger Gopher.

So the rule is:

1. On a small model, verify **tokenizer · dataloader · loss · eval · checkpoint** work  
2. Then scale up  

[StarCoder2](https://arxiv.org/abs/2402.19173) also trained 3B / 7B / 15B under different token budgets.

### Stages this project follows

| Size | Goal |
|:-----|:-----|
| 10M ~ 50M | training loop, tokenizer, loss drop |
| 100M ~ 300M | completion, FIM, basic instruction |
| 1B | small coding assistant experiments |
| 3B | in-house tool integration candidate |
| 7B+ | minimum size worth external exposure |

The latest published benchmark checkpoint is **`sft_base_v6`** (base ~327M). → [§5 Benchmark snapshot](#5-benchmark-snapshot) · per-version write-up [BENCHMARK v1](BENCHMARK-v1.en.md)

---

## 3. Model vs RAG · Tool

Trying to put every company fact into weights does not scale.  
Code, docs, and build logs keep changing.

| Owner | Responsibility |
|:------|:---------------|
| **Model (weights)** | how to think, coding patterns, context, answer tone, **how** to use tools |
| **RAG / Tools** | live facts, internal docs, repos, test runs, build results |

> Model = **how to think and use tools**  
> RAG / Tool = **what is true right now**

(Runtime RAG · Tool integration is on the roadmap; the public focus for now is train · infer · align.)

---

## 4. Project layout

```text
AI/
├── README.md                 philosophy · how to run
├── GLOSSARY.md               terms
├── ARCHITECTURE.md           model · train · infer
├── POST-TRAINING.md          feedback post-training
├── BENCHMARK-v1.md           bench report
└── llm/                      implementation
    ├── model.py              RMSNorm + RoPE + SwiGLU + GQA
    ├── tokenizer.py
    ├── data.py
    ├── train.py              pretrain / sft / dpo
    ├── infer.py              chat · logs
    ├── feedback.py           feedback web UI
    ├── reward.py · rlhf.py   RM + GRPO / RLVR
    └── eval_gate.py          promotion gate
```

> Large weights (`.pt`), raw corpora, and some source artifacts are not published.  
> Design and benchmarks are documented first.

---

## 5. Benchmark snapshot

**Training GPU**: H100, A100

Same 14 prompts × THINKING on/off, `base` (~327M).  
Latest snapshot: **`sft_base_v6`** (`ckpt/benchmark_sft_base_v6.json`, 2026-07-15).

| Checkpoint | Takeaway |
|:-----------|:---------|
| pretrain v1 | effectively 0 on chat format (no instruction training) |
| sft v1 | tries to follow instructions; accuracy still low. THINKING answers 0/14 (close-tag bug) |
| sft v2 | THINKING closure mostly fixed (non-empty answers 13/14). Coding still weak |
| sft v3 | coding (normal chat) 4/5 fully pass. English strong; Korean 0 on all items. THINKING coding still 0/5 |
| sft v4 | higher Korean SFT share (10.3%→22.5%). Bench is mostly lateral, Korean barely recovers |
| sft v5 | swapped base to pretrain_v2 + greedy decoding. First perfect coding (5/5), Korean starts recovering (8/30) |
| **sft v6** | **THINKING handoff fixed.** A prep_sft preprocessing change + split decode budget alone moved thinking avg 0.50→2.93, first thinking-mode coding pass (4/5) |

**sft_base_v6 · normal chat (THINKING off)** · avg score 3.93 / 5

| Area | Result |
|:-----|:-------|
| Korean | avg **2.67 / 5** (first perfect fact score) |
| Japanese | avg 3.00 / 5 |
| English | avg **4.33 / 5** |
| Coding | **25 / 25** tests (**5 / 5** fully pass) |
| THINKING on | avg **2.93 / 5** · non-empty answers **14/14** · coding fully pass **4/5** (+prime partial 3/5) |

v6 highlights (vs v5):

- **THINKING handoff bug fixed**: empty answers 4/14 → 0/14 (every prompt gets an answer), THINKING avg 0.50 → 2.93 (~6x)
- THINKING coding **0/5 → 4/5** — first working code output in six versions (`+prime` partial pass 3/5)
- No-think avg rose alongside it: **3.21 → 3.93**, coding still perfect at 5/5
- First-ever perfect Korean fact score · first correct Japanese math in both modes (Korean total 8/30 → 10/30)
- Base and SFT source mix are identical to v5 — the only change is data preprocessing (`prep_sft`) and a split decode budget
- Remaining issues: Korean arithmetic still collapses, repetition/degeneration in long generations, JA→EN translation never formed

Training process, datasets, and per-item Q&A (per-version write-up):

- [BENCHMARK-v1.md](BENCHMARK-v1.md) (Korean)  
- [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md)  
- [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md)  

Still early-stage. Later versions keep the same prompt set for comparison.

---

## 6. References

| Paper | Link |
|:------|:-----|
| Llama 3 Herd of Models | https://ar5iv.labs.arxiv.org/html/2407.21783 |
| DeepSeek-Coder | https://arxiv.org/html/2401.14196v1 |
| Chinchilla (compute-optimal) | https://arxiv.org/abs/2203.15556 |
| StarCoder 2 | https://arxiv.org/abs/2402.19173 |

---

<div align="center">

**Next**

[Glossary](GLOSSARY.en.md) · [Architecture](ARCHITECTURE.en.md) · [Post-training](POST-TRAINING.en.md) · [Benchmark](BENCHMARK-v1.en.md)

</div>
