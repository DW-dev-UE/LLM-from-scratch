---

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

The published V1 benchmark checkpoint is **base ~327M**. → [BENCHMARK v1](BENCHMARK-v1.en.md)

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

## 5. How to run

Full commands: [llm/명령어-정리.md](llm/명령어-정리.md).  
One demo loop:

```bash
cd llm
pip install torch tokenizers numpy

python data.py demo
python tokenizer.py --input data/corpus.txt data/sft.jsonl
python data.py pretrain
python train.py --mode pretrain --preset nano --steps 2000
python data.py sft
python train.py --mode sft --init ckpt/pretrain_nano.pt --steps 300
python infer.py --ckpt ckpt/sft_nano.pt
```

### Feedback cycle (short)

```bash
python infer.py --ckpt ckpt/sft_nano.pt
python feedback.py

# A) correction SFT
python data.py sft --input data/sft.jsonl data/feedback.jsonl --weight 1 3
python train.py --mode sft --init ckpt/sft_nano.pt --steps 100 --tag v1

# B) DPO
python train.py --mode dpo --init ckpt/sft_nano.pt --steps 100 --tag v1

# C) RM + GRPO / RLVR
python reward.py --init ckpt/sft_nano.pt --steps 200
python rlhf.py --mode rm   --init ckpt/sft_nano.pt --rm ckpt/rm_nano.pt --data data/sft.jsonl
python rlhf.py --mode rlvr --init ckpt/sft_nano.pt --data data/sft.jsonl

python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano_v1.pt
```

For real runs, swap in public corpora / SFT jsonl and scale with `small` / `base`.

---

## 6. Benchmark snapshot

Same 14 prompts × THINKING on/off, `base` (~327M).

| Checkpoint | Takeaway |
|:-----------|:---------|
| pretrain v1 | effectively 0 on chat format (no instruction training) |
| **sft v1** | tries to follow instructions; accuracy still low |

**sft_base_v1 · normal chat (THINKING off)**

| Area | Result |
|:-----|:-------|
| Korean | avg 1.33 / 5 |
| Japanese | avg 0.33 / 5 |
| English | avg 2.67 / 5 |
| Coding | 4 / 17 tests (1 / 5 fully pass) |
| THINKING on | 0 / 14 answers (close-tag issue) |

Training process, datasets, and per-item Q&A:

- [BENCHMARK-v1.md](BENCHMARK-v1.md) (Korean)  
- [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md)  
- [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md)  

Still early-stage. Later versions keep the same prompt set for comparison.

---

## 7. References

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
