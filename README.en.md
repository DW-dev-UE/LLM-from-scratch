<div align="center">

# General-Purpose LLM and In-House AI Architecture

<img src="https://img.shields.io/badge/CUDA-enabled-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/from--scratch-no%20external%20LLM-blueviolet?style=flat" alt="From scratch" />
<img src="https://img.shields.io/badge/status-actively%20updated-brightgreen?style=flat" alt="Actively updated" />

[한국어](README.md) &nbsp;·&nbsp; **[English](README.en.md)** &nbsp;·&nbsp; [日本語](README.ja.md)

</div>

---

> **🚧 This repository is actively updated.**  
> Training runs, benchmarks, docs, and the pipeline keep evolving. Checkpoint versions (v1 → v2 …),
> benchmark reports, and command notes may change — read against the latest commit.
> Issues and PRs are welcome.

This is a project to build, from scratch, a general-purpose LLM that runs on NVIDIA CUDA, plus AI that's
actually useful for real company work — coding, design, that kind of thing. I wrote every piece of it myself:
the tokenizer, the training and inference engines, and the whole pipeline that keeps patching the model with
human feedback after it ships.

I'm well aware that grabbing an existing open-source LLM and fine-tuning it would be far faster and far more
practical. I'm rebuilding it from zero anyway, for one reason — if you never build an LLM's internals with
your own hands, you'll always just be someone who borrows other people's code. So this repo doesn't lean on
any external LLM.

The docs are split as below. Read them in order, or just jump straight to whichever one you need.

| Document | What's inside |
|---|---|
| **README.en.md** (this one) | Design philosophy, parameter strategy, RAG/Tool role split, project structure, how to run it |
| [GLOSSARY.en.md](GLOSSARY.en.md) | If terms like Transformer, weight, RoPE, or DPO sound unfamiliar, start here |
| [ARCHITECTURE.en.md](ARCHITECTURE.en.md) | Model architecture, tokenizer, training/inference engine — with the actual code |
| [POST-TRAINING.en.md](POST-TRAINING.en.md) | The feedback loop that keeps fixing the model with human input after deployment |
| [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md) | Base V1 benchmark report (pretrain vs SFT, language & coding detail) |

The actual implementation lives in the [llm/](llm/) folder, and the commands are collected in
[llm/명령어-정리.md](llm/명령어-정리.md).

## Table of Contents

1. [Architecture Design Process](#1-architecture-design-process)
2. [Parameter Count](#2-parameter-count)
3. [RAG and Tools](#3-rag-and-tools)
4. [Project Structure](#4-project-structure)
5. [How to Run It](#5-how-to-run-it)
6. [Benchmarks (V1)](#6-benchmarks-v1)
7. [References](#references)

---

## 1. Architecture Design Process

Here's what I originally planned:

1️⃣ Build the LLM first on a pure natural-language dataset
2️⃣ Layer a coding-specific dataset on top of that
3️⃣ Polish it by hand with human-guided reinforcement learning

The reasoning seemed obvious enough: no matter how good an AI gets at coding, people still ask their
questions in natural language. So I figured "understanding what the user actually wants, in context" had
to come first, before anything else.

Then I read the Llama 3 paper (*The Llama 3 Herd of Models*), and it turns out this ordering doesn't exist
at all.

> https://ar5iv.labs.arxiv.org/html/2407.21783

Llama 3 is a dense Transformer. It picks up language structure and world knowledge through next-token
prediction during pretraining, then strengthens instruction following, preference alignment, coding,
reasoning, and tool use during post-training. But the pretraining itself is already one mixed blend from
day one — roughly 50% natural language, 25% math/reasoning, 17% code, 8% multilingual, all stirred together.
There's no stage where you "finish natural language first, then bolt on code." It just isn't there.

DeepSeek-Coder makes basically the same point from the opposite direction. They trained 1.3B–33B models
from scratch on 2T tokens, using repository-level corpora and fill-in-the-blank (FIM) tasks at a 16K context.
The data mix was 87% source code, 10% code-related English, and 3% Chinese text unrelated to code.

> https://arxiv.org/html/2401.14196v1

With natural language sitting at only about 13% of the mix, it still came out as the best-performing
open-source code LLM at the time. Neither paper has a step where natural language gets "completed" on its
own first — both train natural language and code together from the very beginning.

Bottom line: the order I originally set up was just wrong.

**The order I threw out**

```
Natural-language-only training → code-only training → instruction tuning
```

Natural language ability is obviously still necessary — I just didn't need to carve it out and "finish" it
separately beforehand. So I flipped the order.

**The order I switched to**

```
Mixed base pretraining (natural language + code + math + docs)
  → code-heavy continued pretraining
    → instruction tuning
      → preference optimization
```

### Data Mix Ratios for the 300M-1B Range

| Data type | Ratio | Notes |
|---|---|---|
| Raw source code | 35% ~ 45% | |
| Code-adjacent text | 15% ~ 20% | README, docs, issues, PRs, commits |
| General technical natural language | 20% ~ 30% | English + Korean technical documents |
| Math / reasoning | 5% ~ 10% | |
| Logs / tests / errors / diffs | 5% ~ 10% | |
| Korean instruction / bilingual data | 5% ~ 10% | |

---

## 2. Parameter Count

Parameters are, roughly, the size of the model (if the terminology's throwing you off, check
[GLOSSARY.en.md](GLOSSARY.en.md)). Bigger generally means a better shot at being smart, but this project
went with a slow climb instead: 300M → 1B → 3B.

The Chinchilla paper (*Training Compute-Optimal Large Language Models*) argues that for a fixed compute
budget, model size and training-token count need to scale up together.

> https://arxiv.org/abs/2203.15556

They ran over 400 models spanning 70M to 16B+ parameters and 5B to 500B tokens, and the conclusion was:
double the model size, and you need to double the token count too, or you're not compute-optimal. Models
from that era — GPT-3, Gopher, Jurassic-1, Megatron-Turing NLG — all turned out to be undertrained relative
to their size. Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens), a model four times its size,
on MMLU and other benchmarks, at the same compute cost.

So instead of blindly cranking up parameter count first, it's safer to confirm the tokenizer, dataloader,
loss curve, eval, and checkpoint pipeline all actually work on a small model, and only then scale up.
StarCoder2 is a real example of this playing out — they trained 3B/7B/15B models separately, each on a
different token budget (roughly 3.1T/3.5T/4.1T), and shipped a whole model family at once so people could
pick based on their deployment budget. (The 15B model matched CodeLlama-34B, a model more than twice its
size.)

> https://arxiv.org/abs/2402.19173

Here's the actual scale-up path this project follows:

| Scale | Purpose |
|---|---|
| 10M ~ 50M | Confirm the training loop, tokenizer, and that loss is actually going down |
| 100M ~ 300M | Experiment with code completion, FIM, basic instruction following |
| 1B | Small coding-assistant experiments |
| 3B | Real-use candidate that can hook into in-house tools |
| 7B+ | Minimum scale worth exposing as an external-facing service |

---

## 3. RAG and Tools

Being a company-specific LLM doesn't mean you should try to cram all knowledge into the model weights.
Stuff that keeps changing — the latest code, latest docs, test results, build logs — is better wired up
through RAG and tool calls. Retraining the whole model every time some new piece of tech shows up just
isn't something anyone can keep up with.

Here's how I split the responsibilities:

| Responsible for | Covers |
|---|---|
| **Model (weights)** | Reasoning style, coding patterns, context understanding, answer style, ability to use tools |
| **RAG / Tools (external hookups)** | Up-to-date knowledge, internal docs, real repos, test runs, build results, deployment environment |

The model owns "how to think and how to use tools"; RAG/Tools own "what's actually true right now."

---

## 4. Project Structure

```
AI/
├── README.md              # This doc — design philosophy, parameter strategy, project structure, how to run it
├── README.en.md            # English
├── README.ja.md            # Japanese
├── GLOSSARY.md              # Terminology reference (ko / en / ja)
├── GLOSSARY.en.md
├── GLOSSARY.ja.md
├── ARCHITECTURE.md          # Model architecture, training, inference (ko / en / ja)
├── ARCHITECTURE.en.md
├── ARCHITECTURE.ja.md
├── POST-TRAINING.md         # Post-training (human feedback) (ko / en / ja)
├── POST-TRAINING.en.md
├── POST-TRAINING.ja.md
├── BENCHMARK-v1.md          # Base V1 benchmark report (ko / en / ja)
├── BENCHMARK-v1.en.md
├── BENCHMARK-v1.ja.md
└── llm/                      # Actual implementation
    ├── model.py               # Architecture — RMSNorm+RoPE+SwiGLU+GQA, KV cache, hidden()
    ├── tokenizer.py            # Tokenizer training/loading
    ├── data.py                # Data preparation — demo / pretrain / sft·correction·mixing / ratings / hygiene
    ├── train.py                # Training loop — pretrain / sft / dpo, --tag versioning·snapshots
    ├── infer.py                # Inference engine + CLI chat, auto-saves conversation logs
    ├── feedback.py             # LLM Feedback Studio — web UI for collecting feedback
    ├── reward.py               # Reward model — GPT backbone + scalar head, ranking loss
    ├── rlhf.py                 # GRPO policy optimization (RLVR rule-based reward / RM scalar reward)
    ├── eval_gate.py            # Promotion gate — holdout regression loss + win-rate
    ├── 명령어-정리.md            # Collection of commands for running the full pipeline
    ├── tokenizer.json
    ├── data/                   # corpus, sft/preference/feedback/ratings jsonl, rated_sft.jsonl, train/val bin
    ├── ckpt/                   # pretrain/sft/dpo/rm/grpo checkpoints + benchmark JSON
    └── logs/                   # chat_log.jsonl — inference conversation logs
```

---

## 5. How to Run It

The full command list is laid out step by step in [llm/명령어-정리.md](llm/명령어-정리.md). To run
through the whole thing end-to-end once:

```bash
cd llm
pip install torch tokenizers numpy

python data.py demo                                        # 1. Generate sample data
python tokenizer.py --input data/corpus.txt data/sft.jsonl # 2. Train tokenizer
python data.py pretrain                                    # 3. Prepare pretraining data
python train.py --mode pretrain --preset nano --steps 2000 # 4. Pretraining
python data.py sft                                         # 5. Prepare SFT data
python train.py --mode sft --init ckpt/pretrain_nano.pt --steps 300  # 6. SFT
python infer.py --ckpt ckpt/sft_nano.pt                    # 7. Chat (logs recorded automatically)
```

How to run a full cycle of human feedback after deployment is covered in
[POST-TRAINING.en.md](POST-TRAINING.en.md). The short version:

```bash
python infer.py --ckpt ckpt/sft_nano.pt          # 1. Conversation → collected into logs/chat_log.jsonl
python feedback.py                               # 2. Rate, correct, and label preferences in the web UI

# Track A — Corrective SFT
python data.py sft --input data/sft.jsonl data/feedback.jsonl --weight 1 3
python train.py --mode sft --init ckpt/sft_nano.pt --steps 100 --tag v1

# Track B — DPO
python train.py --mode dpo --init ckpt/sft_nano.pt --steps 100 --tag v1

# Track C — Reward model + GRPO / RLVR
python reward.py --init ckpt/sft_nano.pt --steps 200
python rlhf.py --mode rm   --init ckpt/sft_nano.pt --rm ckpt/rm_nano.pt --data data/sft.jsonl
python rlhf.py --mode rlvr --init ckpt/sft_nano.pt --data data/sft.jsonl

# If the promotion gate passes, repeat from step 1 with the new checkpoint
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano_v1.pt
```

For real use, swap `data/corpus.txt` for an actual corpus (TinyStories, FineWeb-Edu, etc.), swap
`data/sft.jsonl` for public instruction/reasoning data, and scale up with `--preset small|base`.

---

## 6. Benchmarks (V1)

Results for **pretrain_base_v1** and **sft_base_v1** (`base` preset, ~327M) on the same 14 prompts × 2
modes (THINKING on/off). Full tables, interpretation, and per-item notes are in the language-specific reports.

| Document | Language |
|---|---|
| [BENCHMARK-v1.md](BENCHMARK-v1.md) | 한국어 |
| [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md) | English |
| [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md) | 日本語 |

**Headline (sft_base_v1, no-thinking)**

| Area | Result |
|---|---|
| Korean avg | 1.33 / 5 |
| Japanese avg | 0.33 / 5 |
| English avg | 2.67 / 5 |
| Coding tests | 4 / 17 (full pass 1/5 — `is_palindrome`) |
| THINKING mode | **0/14 answers** (`</THINKING>` never closed — use `--no-thinking`) |

Pretrain alone scores 0/28 (textbook failure without chat SFT). After 5,000 SFT steps the model reaches an
“attempts to follow instructions” stage. Still early-stage; later versions and benchmarks will keep landing.

Raw JSON: [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json),
[`llm/ckpt/benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json)

---

## References

- Llama Team, AI @ Meta. *The Llama 3 Herd of Models.* — https://ar5iv.labs.arxiv.org/html/2407.21783
- Guo, D. et al. *DeepSeek-Coder: When the Large Language Model Meets Programming — The Rise of Code Intelligence.* — https://arxiv.org/html/2401.14196v1
- Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* — https://arxiv.org/abs/2203.15556
- Lozhkov, A. et al. *StarCoder 2 and The Stack v2: The Next Generation.* — https://arxiv.org/abs/2402.19173

<div align="center">

---

**Next up**

[📖 GLOSSARY.en.md](GLOSSARY.en.md) — Start with the terminology &nbsp;|&nbsp;
[🏗️ ARCHITECTURE.en.md](ARCHITECTURE.en.md) — Model architecture, training, inference &nbsp;|&nbsp;
[🔁 POST-TRAINING.en.md](POST-TRAINING.en.md) — Post-training with human feedback &nbsp;|&nbsp;
[📊 BENCHMARK-v1.en.md](BENCHMARK-v1.en.md) — Base V1 benchmark

</div>
