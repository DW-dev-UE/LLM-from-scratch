# LLM From Scratch

[![KO](https://img.shields.io/badge/KO-lightgrey)](README.md) [![EN](https://img.shields.io/badge/EN-0969da)](README.en.md) [![JA](https://img.shields.io/badge/JA-lightgrey)](README.ja.md) ![Last commit](https://img.shields.io/github/last-commit/DW-dev-UE/LLM-from-scratch)

> [!NOTE]
> **This repo is actively updated.**  
> Training results, benchmarks, and docs change over time. Prefer the latest commit. Issues and PRs welcome.

> Before we start: this writing may not be the most polished for readability.
> You could ask an AI to "make this more readable," but then the record of how I thought and what process I went through can disappear.
> Learning with AI is fine, but rewriting sentences myself to organize my knowledge matters more.

---

Alright, let's begin.

## What this project does

A from-scratch experiment to build a **general-purpose LLM** on NVIDIA CUDA — an AI that can be used for real work like coding and documents.

- Tokenizer training
- Model architecture (Decoder-only Transformer)
- Pretrain · SFT · DPO · GRPO
- KV-cache inference · thinking mode (THINKING)
- Human-feedback UI · promotion gate

Fine-tuning an existing open-source LLM would be much faster. Even so, I rebuilt it from scratch for one reason only.

If you never implement an LLM’s internals yourself, you only ever borrow other people’s code.

> If you never implement an LLM’s internals yourself, you only ever borrow other people’s code.

So this repository does **not** depend on external commercial or open LLM weights.

```ini
[Lessons learned building the 326.7M-scale model]

1. Multilingual inefficiency

Supporting multiple languages under a small-model budget is fatal below ~7B. As v7 showed, a small multilingual model converges to a mediocre middle ground across all three languages.

Above ~70B, cross-lingual transfer starts to pay off, but at small scale the English corpus is overwhelmingly larger and higher quality, so small models should focus on English.

2. Corpus quality matters a lot

Performance = parameters x token count x data quality. Even with an already-public corpus, finding a high-quality one is the top priority.

3. Noise in benchmarking

After training, I ran the benchmark by sampling each item once at temperature 0.7. The same model gives a different answer every run, so I couldn't tell whether a score gap between versions was real skill or just a dice roll.

Fix: from v5 on, switched to a greedy protocol so the same model always gives the same answer.

But I also got fooled by val loss.

Val was the last 1% of the corpus file order — a single ko-wiki tail domain, not representative of overall ability. (In v2, ko-wiki val got worse while downstream benchmarks hit an all-time high.)

The measurement was a single batch of 16 random sequences, with ±0.25 noise, and I misread that noise as real improvement/regression twice — fooled twice, in other words.

Fix: judge only by multi-batch averages, plus baked a 1%-per-source mixed val directly into the v3-en downloader.

4. Thinking collapse (v1–v5, the biggest mistake)

The correct answer would get written inside `<THINKING>` while the actual answer field stayed blank. Zero thinking-mode coding passes (0/5) for six versions straight.
There were three root causes.

> 99% of the thinking training data got truncated mid-thought at max_len=1024.
> THINK_SUFFIX was inconsistent between training and inference.
> infer.py didn't separate the thinking/answer token budget, so once thinking ate the whole budget the answer came out empty.

Fix: in data.py, preserve the thinking tail (if too long, cut the middle of the reasoning but keep the closing tag + answer: 47,273 examples; if still too long, downgrade to no-thinking: 17,540 examples) + attach THINK_SUFFIX conditionally + split the budget in infer.py and add an EOS guard.

After the fix, thinking-mode answers went from 10/14 to 14/14, the average from 0.50 to 2.93, and the first code output scored 4/5.

The common thread across all three: none of it was the model being dumb — the measurement or the plumbing was broken. It happened because I kept focusing on the training model itself; from now on, when scores look off, I check sampling, val design, and preprocessing first, in that order.
```

---

## Docs

| Doc | Contents |
|:----|:---------|
| **README** (this file) | Philosophy, parameter strategy, structure, how to run |
| [GLOSSARY](GLOSSARY.en.md) | Terms like Transformer, RoPE, DPO |
| [ARCHITECTURE](ARCHITECTURE.en.md) | Transformer walkthrough · model design · tokenizer · train · infer |
| [POST-TRAINING](POST-TRAINING.en.md) | Post-deploy human feedback loop |
| [BENCHMARK v1](BENCHMARK-v1.en.md) | Base model benchmark (training process · Q&A) |

[![KO](https://img.shields.io/badge/KO-lightgrey)](README.md) [![EN](https://img.shields.io/badge/EN-0969da)](README.en.md) [![JA](https://img.shields.io/badge/JA-lightgrey)](README.ja.md)

---

## 1. Architecture

The **numeric walkthrough** from tokens → embeddings → attention → FFN → logits has moved to [ARCHITECTURE — How a Transformer works](ARCHITECTURE.en.md#2-how-a-transformer-works-pedagogical-walkthrough). (That document continues with how the pedagogical example differs from the LLaMA-style production design.)

### First plan

The first plan was simple.

1. Build an LLM first with natural-language data only
2. Stack coding-specialized data on top
3. Refine with human reinforcement learning

Even for a coding AI, questions come in natural language, so I thought **language understanding had to be finished first**.

### What the papers show

[Llama 3](https://ar5iv.labs.arxiv.org/html/2407.21783) does not use that sequence.  
Pretraining already blends every domain together from the start.

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
| **Current** | **mixed pretrain** → code-heavy continued pretrain → instruction → preference |

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

> [!TIP]
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

> [!IMPORTANT]
> Model = **how to think and use tools**  
> RAG / Tool = **what is true right now**

(Runtime RAG · Tool integration is on the roadmap; the public focus for now is the train · infer · align pipeline.)

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
| sft v1 | tries to follow instructions; accuracy still low. THINKING answers 0/14 (never emits the closing tag) |
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
