[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](README.md) [![English](https://img.shields.io/badge/English-0969DA?style=flat-square)](README.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](README.ja.md)

# LLM From Scratch

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)

> [!NOTE]
> **This repo is actively updated.**  
> Training results, benchmarks, and docs change over time. Prefer the latest commit. Issues and PRs welcome.

> Before we start: this writing may not be the most polished for readability.
> You could ask an AI to "make this more readable," but then the record of how I thought and what process I went through can disappear.
> Learning with AI is fine, but rewriting sentences myself to organize my knowledge matters more.

> [!IMPORTANT]
> **APEX-1 (1B scale) — Pretrain · SFT · DPO done**
>
> ~**1,119.5M** measured parameters. ( 24 layers · d_model 2048 · GQA (16Q/4KV) + RoPE + SwiGLU + RMSNorm )
>
> - 🤗 **Released**: [huggingface.co/YOON1v/Apex-1-DPO](https://huggingface.co/YOON1v/Apex-1-DPO) — loads directly with `transformers` (mapped onto the Qwen3ForCausalLM architecture)
> - 📏 **Context**: max 4096 (trained at 2048)
> - 🔥 **Pretrain**: 51K steps · 20B tokens · v3-en 80.8GB · English-only
>
> | Model | Training tokens | HellaSwag | ARC (avg) | PIQA | GSM8K | HumanEval |
> |:---|---:|---:|---:|---:|---:|---:|
> | **Apex-1 DPO (1.1B)** | 0.02T | 46.9 | 41.6 | 68.6 | 1.9 | 8.5 |
> | Pythia-1.0B | 0.3T | 47.2 | 38.0 | 69.2 | — | 1.8 |
> | OPT-1.3B | 0.18T | 53.7 | 40.1 | 72.4 | — | — |
> | TinyLlama-1.1B | 3T | 59.2 | 42.7 | 73.3 | — | 9.2 |
> | OLMo-1B | 2T | 62.5 | 46.3 | 73.7 | — | — |
> | Llama-3.2-1B | 9T | 61.2 | 49.2 | 74.8 | 7.6 | 18.9 |
> | Qwen2.5-1.5B | 18T | 66.4 | 58.5 | 76.1 | 61.7 🏆 | 37.2 🏆 |
> | SmolLM2-1.7B | 11T | 68.7 🏆 | 60.5 🏆 | 77.6 🏆 | 31.1 | 22.6 |
>
> **DPO ≥ SFT, no alignment tax** · Apex-1 matches the 15×-more-trained Pythia on commonsense average and beats it on ARC (most token-efficient here) · the gap to Llama-3.2/Qwen2.5 is a 450–900× data-scale difference, not architecture → [BENCHMARK v2 §5.4](BENCHMARK-v2.en.md#54-standard-benchmarks-lm-evaluation-harness)
>
> <sub>**Benchmark sources** (canonical, heavily-cited papers in each area): HellaSwag [Zellers+19](https://arxiv.org/abs/1905.07830) · ARC [Clark+18](https://arxiv.org/abs/1803.05457) · PIQA [Bisk+19](https://arxiv.org/abs/1911.11641) · GSM8K [Cobbe+21](https://arxiv.org/abs/2110.14168) · HumanEval [Chen+21](https://arxiv.org/abs/2107.03374)</sub>  
> <sub>**Comparison model sources**: Pythia [Biderman+23](https://arxiv.org/abs/2304.01373) · OPT [Zhang+22](https://arxiv.org/abs/2205.01068) · TinyLlama [Zhang+24](https://arxiv.org/abs/2401.02385) · OLMo [Groeneveld+24](https://arxiv.org/abs/2402.00838) · Qwen2.5 [Qwen+24](https://arxiv.org/abs/2412.15115) · SmolLM2 [Ben Allal+25](https://arxiv.org/abs/2502.02737) · Llama-3.2 (no arXiv report, Meta blog only)</sub>
>
> - ⏸️ **Deferred**: MoE · YaRN · FP8 · multi-GPU (next scale after 1B is validated)

---
> [!NOTE]
> 🚀 **Next goal — APEX-2 (7B scale)**: architecture design in progress

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
| [BENCHMARK v2](BENCHMARK-v2.en.md) | APEX-1 (1B) benchmark · RLVR → DPO decision log |
| [BENCHMARK v1](BENCHMARK-v1.en.md) | Base model (327M) benchmark (training process · Q&A) |
| [ThinkingLab](ThinkingLab/ThinkingLab.en.md) | Hypothesis / brainstorming log (not-yet-validated ideas) |

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

Pretrain·SFT·DPO are done through 1B (APEX-1); 3B is being skipped in favor of going straight to **APEX-2 (7B scale)**, now in development.

Two lines run in parallel: the 1B `Apex-1` line's latest is **`dpo_Apex-1_v1`** (pretrain+SFT+DPO done, all 11 standard benchmarks measured), the 327M `base` line's latest is **`sft_base_v6`**. Next goal: **APEX-2 (7B scale)**. → [§5 Benchmark snapshot](#5-benchmark-snapshot) · per-version write-up [BENCHMARK v2](BENCHMARK-v2.en.md)

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
├── BENCHMARK-v2.md           APEX-1 (1B) bench report
├── BENCHMARK-v1.md           327M bench report
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

Two lines run in parallel: **1B `Apex-1`** (English-only) and **327M `base`** (KO/JA/EN multilingual). The prompt sets differ too.

### 5.1 1B `Apex-1` line ⭐ primary

English-only 15 prompts × THINKING on/off (the 5 coding prompts are the same items as the 327M set).  
Latest snapshot: **`sft_Apex-1_v1`** (`ckpt/benchmark_sft_Apex-1_v1_raw.json`, 2026-07-22). Pretrain 51K steps (20B tokens) + SFT 8.4K steps + RLVR abandoned then DPO — all done (§6 in [BENCHMARK v2](BENCHMARK-v2.en.md)).

**sft_Apex-1_v1**

| | 💬 normal chat | 🧠 THINKING on |
|:--|:--:|:--:|
| QA correct (keyword) | 5/8 | 4/8 |
| Coding fully passed | **5/5** (tests 25/25) | 3/5 (tests 15/25) |
| Empty answers | 0 | 0 |
| Over-length | 8/10 | 5/10 |
| Mean repetition rate | 0.032 | 0.042 |

The 327M line's biggest failure — THINKING handoff (a blank answer field) — **never happened at all on this line.** The new bottleneck instead is verbosity and repetition, and turning THINKING on actually drops both coding and QA (thinking eats the answer budget first).

Capability (coding 5/5) looked sufficient, so the first choice was **RLVR** — but running it showed GSM8K accuracy was ~0%, so GRPO groups had zero reward variance and no learning signal at all. Since the real problem (verbosity/non-compliance) doesn't require correctness, the direction switched to **DPO**, which finished training on 60K preference pairs.

Full write-up: [BENCHMARK-v2.en.md](BENCHMARK-v2.en.md)

### 5.2 Standard benchmarks (lm-evaluation-harness)

Converted `sft_Apex-1_v1` and `dpo_Apex-1_v1` to HF format (logit-match verified, max diff 2.3e-5) and measured them on the standard harness **side by side with other models that have published scores**. 7 commonsense tasks 0-shot, MMLU 5-shot, GSM8K 5-shot, HumanEval/MBPP 0-shot pass@1 — matching the TinyLlama paper's (Table 2·3) protocol.

| Metric | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B |
|:---|---:|---:|---:|---:|
| Commonsense avg, 7 tasks (0-shot) | 49.54 | 49.65 | 52.99 | 48.30 |
| MMLU (5-shot) | 24.83 | 24.90 | 25.34 | 25.70 |
| GSM8K (5-shot, strict) | 1.44 | 1.90 | — | — |
| HumanEval (pass@1) | 8.54 | 8.54 | 9.15 | 1.83 |
| MBPP (pass@1) | 4.80 | 5.20 | — | — |

> [!IMPORTANT]
> **DPO ≥ SFT, no alignment tax** — DPO matches or beats SFT on every metric here. **BoolQ 62.20 ranks 1st among the 4 comparison models** (including TinyLlama-1.1B, Pythia-1.0B, OPT-1.3B), and trained on only ~20B tokens, Apex-1 beats the 300B-token Pythia-1.0B on HumanEval by a wide margin (8.54 vs 1.83).

**Comparison models**: TinyLlama-1.1B (1.1B params · 3T tokens, 2024) · Pythia-1.0B (1.0B params · 300B tokens, EleutherAI) · OPT-1.3B (1.3B params · 180B tokens, Meta) — vs. Apex-1's 1.12B params · ~20B tokens, that's 150×, 15×, and 9× more tokens respectively.

For what each of these benchmarks (commonsense 7-task suite, MMLU, GSM8K, HumanEval, MBPP) actually measures, how authoritative each one is, and the full per-task breakdown (including OPT): [BENCHMARK-v2.en.md §5.4](BENCHMARK-v2.en.md#54-standard-benchmarks-lm-evaluation-harness)

**A wider comparison** — including 2024–2025 small open models trained on far more tokens (🏆 = column winner):

| Model | Training tokens | HellaSwag | ARC (avg) | PIQA | GSM8K | HumanEval |
|:---|---:|---:|---:|---:|---:|---:|
| **Apex-1 DPO (1.1B)** | 0.02T | 46.9 | 41.6 | 68.6 | 1.9 | 8.5 |
| Pythia-1.0B | 0.3T | 47.2 | 38.0 | 69.2 | — | 1.8 |
| OPT-1.3B | 0.18T | 53.7 | 40.1 | 72.4 | — | — |
| TinyLlama-1.1B | 3T | 59.2 | 42.7 | 73.3 | — | 9.2 |
| OLMo-1B | 2T | 62.5 | 46.3 | 73.7 | — | — |
| Llama-3.2-1B | 9T | 61.2 | 49.2 | 74.8 | 7.6 | 18.9 |
| Qwen2.5-1.5B | 18T | 66.4 | 58.5 | 76.1 | 61.7 🏆 | 37.2 🏆 |
| SmolLM2-1.7B | 11T | 68.7 🏆 | 60.5 🏆 | 77.6 🏆 | 31.1 | 22.6 |

> [!IMPORTANT]
> **Reading it by token efficiency** — Apex-1 (0.02T) matches the 15×-more-trained Pythia-1.0B (0.3T) on commonsense average and beats it on ARC. HumanEval 8.5 is **4.7×** Pythia's 1.83, and close to TinyLlama's 9.15 (3T tokens) — unusual for a ~20B-token model. The gap to Llama-3.2-1B (9T) and Qwen2.5-1.5B (18T) is not architecture — it's a **450–900× data-scale difference**.

SmolLM2 and Qwen2.5 were trained on 550–900× more tokens than Apex-1 — falling behind here is expected, and **comparing models with a similar token budget (table above)** gives a fairer read of where this project actually stands. Details: [BENCHMARK-v2.en.md §5.4](BENCHMARK-v2.en.md#54-standard-benchmarks-lm-evaluation-harness)

<details>
<summary><b>5.3 327M `base` line (expand to view)</b></summary>

<br/>

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

</details>

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

[Glossary](GLOSSARY.en.md) · [Architecture](ARCHITECTURE.en.md) · [Post-training](POST-TRAINING.en.md) · [Benchmark v1](BENCHMARK-v1.en.md) · [Benchmark v2](BENCHMARK-v2.en.md) · [ThinkingLab](ThinkingLab/ThinkingLab.en.md)

</div>
