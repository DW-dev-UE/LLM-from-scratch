<div align="center">

# LLM-from-scratch

**Tokenizer → train → infer → feedback loop, built from scratch (no external LLM weights)**

GPT-2-class decoder-only Transformer · general-purpose and in-house AI experiments

<br/>

![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?style=flat-square&logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![From scratch](https://img.shields.io/badge/from%20scratch-no%20external%20LLM-7c3aed?style=flat-square)
![Status](https://img.shields.io/badge/status-actively%20updated-22c55e?style=flat-square)

<br/>

[한국어](README.md) · [English](README.en.md) · [日本語](README.ja.md)

**Repo:** [github.com/DW-dev-UE/LLM-from-scratch](https://github.com/DW-dev-UE/LLM-from-scratch)

</div>

---

> **This repo is actively updated.**  
> Training results, benchmarks, and docs change over time. Prefer the latest commit. Issues and PRs welcome.

---

## What this is

A from-scratch experiment to build a **general-purpose LLM** on NVIDIA CUDA, aimed at real work (coding, docs, etc.).

- Tokenizer training  
- Decoder-only Transformer architecture  
- Pretrain · SFT · DPO · GRPO  
- KV-cache inference · thinking mode  
- Human-feedback UI · promotion gate  

Fine-tuning an existing open model would be faster.  
I still rebuilt it for one reason:

> If you never implement an LLM’s internals yourself, you only ever borrow other people’s code.

This repository does **not** depend on external commercial or open LLM weights.

---

## Docs

| Doc | Contents |
|:----|:---------|
| **README** (this file) | Philosophy, scaling, structure, how to run |
| [GLOSSARY](GLOSSARY.en.md) | Terms (Transformer, RoPE, DPO, …) |
| [ARCHITECTURE](ARCHITECTURE.en.md) | Model · tokenizer · train · infer |
| [POST-TRAINING](POST-TRAINING.en.md) | Post-deploy feedback loop |
| [BENCHMARK v1](BENCHMARK-v1.en.md) | Base checkpoint report (training, data, Q&A) |

Code lives under [`llm/`](llm/).  
Commands: [`llm/명령어-정리.md`](llm/명령어-정리.md).

---

## Contents

1. [How the design changed](#1-how-the-design-changed)  
2. [Scale parameters slowly](#2-scale-parameters-slowly)  
3. [Model vs RAG / tools](#3-model-vs-rag--tools)  
4. [Layout](#4-layout)  
5. [How to run](#5-how-to-run)  
6. [Benchmark snapshot](#6-benchmark-snapshot)  
7. [References](#7-references)

---

## 1. How the design changed

### First plan

| Step | Idea |
|:----:|:-----|
| **1** | Train language-only first |
| **2** | Add code data later |
| **3** | Human RL at the end |

Even coding AIs get **natural-language** questions, so language understanding felt like the first milestone.

### What the papers actually do

[Llama 3](https://ar5iv.labs.arxiv.org/html/2407.21783) does not use that sequence.  
Pretraining already mixes domains:

| Share (approx.) | Data |
|:---------------:|:-----|
| 50% | Natural language |
| 25% | Math / reasoning |
| 17% | Code |
| 8% | Multilingual |

Post-training strengthens instruction, preference, coding, tools.  
There is **no** “finish language, then bolt on code” stage.

[DeepSeek-Coder](https://arxiv.org/html/2401.14196v1) is similar from the other side:  
~87% code from the start, strong results despite only ~13% natural language.

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

### Mix guide (300M–1B)

| Kind | Share | Examples |
|:-----|:-----:|:---------|
| Raw source code | 35–45% | source files |
| Code-adjacent | 15–20% | README, issues, PRs, commits |
| Technical language | 20–30% | EN / KO tech writing |
| Math / reasoning | 5–10% | math, logic |
| Logs / tests / diffs | 5–10% | logs, tests, diffs |
| Korean / bilingual | 5–10% | instructions, bilingual |

---

## 2. Scale parameters slowly

Parameters = model size ([GLOSSARY](GLOSSARY.en.md)).  
Bigger can help, but this project grows **300M → 1B → 3B** in stages.

[Chinchilla](https://arxiv.org/abs/2203.15556) in one line:

> Under a fixed compute budget, grow **model size and tokens together**.

So:

1. Make tokenizer, dataloader, loss, eval, and checkpoints work on a **small** model  
2. Then scale up  

[StarCoder2](https://arxiv.org/abs/2402.19173) trained 3B / 7B / 15B under different token budgets as a family.

### Stages here

| Size | Goal |
|:-----|:-----|
| 10M–50M | loop, tokenizer, loss drop |
| 100M–300M | completion, FIM, basic instruction |
| 1B | small coding assistant |
| 3B | in-house tooling candidate |
| 7B+ | minimum size worth external exposure |

The published V1 benchmark uses **~327M base**. → [BENCHMARK v1](BENCHMARK-v1.en.md)

---

## 3. Model vs RAG / tools

Do not put every company fact into weights.  
Code and docs change constantly.

| Owner | Responsibility |
|:------|:---------------|
| **Model (weights)** | how to think, code patterns, style, **how** to use tools |
| **RAG / tools** | what is true now, internal docs, repos, tests, builds |

> Model = reasoning and tool use  
> RAG / tools = live facts  

(Runtime RAG is on the roadmap; this release focuses on train / infer / align.)

---

## 4. Layout

```text
AI/
├── README*.md
├── GLOSSARY*.md
├── ARCHITECTURE*.md
├── POST-TRAINING*.md
├── BENCHMARK-v1*.md
└── llm/                 implementation
    ├── model.py
    ├── train.py · infer.py
    ├── feedback.py · reward.py · rlhf.py
    └── eval_gate.py
```

> Large weights, raw corpora, and some source artifacts are **not** published.  
> Docs and benchmark write-ups are the public surface for now.

---

## 5. How to run

Full command list: [llm/명령어-정리.md](llm/명령어-정리.md).

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

python data.py sft --input data/sft.jsonl data/feedback.jsonl --weight 1 3
python train.py --mode sft --init ckpt/sft_nano.pt --steps 100 --tag v1
python train.py --mode dpo --init ckpt/sft_nano.pt --steps 100 --tag v1

python reward.py --init ckpt/sft_nano.pt --steps 200
python rlhf.py --mode rm   --init ckpt/sft_nano.pt --rm ckpt/rm_nano.pt --data data/sft.jsonl
python rlhf.py --mode rlvr --init ckpt/sft_nano.pt --data data/sft.jsonl

python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano_v1.pt
```

For real runs, swap in public corpora / SFT sets and use `small` or `base`.

---

## 6. Benchmark snapshot

Same 14 prompts × thinking on/off, `base` ~327M.

| Checkpoint | Takeaway |
|:-----------|:---------|
| pretrain v1 | ~0 on chat-format items |
| **sft v1** | tries to follow instructions; accuracy still low |

**sft_base_v1 · normal chat (thinking off)**

| Area | Result |
|:-----|:-------|
| Korean | avg 1.33 / 5 |
| Japanese | avg 0.33 / 5 |
| English | avg 2.67 / 5 |
| Coding | 4 / 17 tests (1 / 5 fully pass) |
| Thinking on | 0 / 14 answers (close-tag bug) |

Full write-up (training, datasets, every Q&A):

- [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md)  
- [BENCHMARK-v1.md](BENCHMARK-v1.md) · [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md)  

Still early-stage. Later versions keep the same prompt set for comparison.

---

## 7. References

| Paper | Link |
|:------|:-----|
| Llama 3 | https://ar5iv.labs.arxiv.org/html/2407.21783 |
| DeepSeek-Coder | https://arxiv.org/html/2401.14196v1 |
| Chinchilla | https://arxiv.org/abs/2203.15556 |
| StarCoder 2 | https://arxiv.org/abs/2402.19173 |

---

<div align="center">

**Next**

[Glossary](GLOSSARY.en.md) · [Architecture](ARCHITECTURE.en.md) · [Post-training](POST-TRAINING.en.md) · [Benchmark](BENCHMARK-v1.en.md)

</div>
