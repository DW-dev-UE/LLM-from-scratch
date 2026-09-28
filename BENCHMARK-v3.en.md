[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](BENCHMARK-v3.md) [![English](https://img.shields.io/badge/English-0969DA?style=flat-square)](BENCHMARK-v3.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](BENCHMARK-v3.ja.md)

[← README](README.en.md)

---

# 📊 BENCHMARK v3 · APEX-2 (MoE 3.87B · 1.45B active)

> A different line from APEX-1 (1.1B dense). This model was pretrained from scratch with a **MoE (Mixture-of-Experts)** architecture; its code lives in a separate repository (GeneralLLM).

| | |
|:--|:--|
| 🆕 **Final** | `sft2_chat` (SFT stage 2, step 536) · 2026-09-28 |
| 📦 **Model** | APEX-2 · Decoder-only MoE · measured **3,869.1M** (**1,453.2M** active per token) |
| 🤗 **Released** | [huggingface.co/YOON1v/Apex-2](https://huggingface.co/YOON1v/Apex-2) — loads directly with `transformers` and vLLM (Qwen3MoeForCausalLM mapping) |
| 🧪 **Evaluation** | 5 code · 2 math · IFEval · MMLU · 6 commonsense (vLLM greedy, chat template) |
| 📁 **Raw** | [`ckpt/benchmark_sft_Apex-2_v1.json`](ckpt/benchmark_sft_Apex-2_v1.json) · [`ckpt/benchmark_dpo_Apex-2_v1.json`](ckpt/benchmark_dpo_Apex-2_v1.json) |
| ✅ **Outcome** | Pretrain → SFT done. DPO lowered scores and was dropped (§7) |

### Contents

1. [At a glance](#1-at-a-glance)
2. [Model card](#2-model-card)
3. [Training](#3-training)
4. [Datasets](#4-datasets)
5. [Scoring](#5-scoring)
6. [Comparison with similar-size models](#6-comparison-with-similar-size-models)
7. [DPO · RLVR log](#7-dpo--rlvr-log)
8. [Limitations](#8-limitations)

---

## 1. At a glance

| Area | Benchmark | **SFT (final)** | base | DPO (dropped) |
|:--|:--|--:|--:|--:|
| Code | HumanEval | **43.9** | 36.6 | 36.0 |
| Code | HumanEval+ | **41.5** | 32.9 | 32.3 |
| Code | MBPP | **56.3** | 54.8 | 45.2 |
| Code | MBPP+ | **48.9** | 46.3 | 37.3 |
| Code | MultiPL-E HumanEval C++ | **36.0** | — | 8.1 |
| Code | MultiPL-E MBPP C++ | **41.6** | — | 3.8 |
| Code | LiveCodeBench v5–v6 | 3.2 | — | 3.5 |
| Code | └ LCB easy (84) | 11.9 | — | 10.7 |
| Code | └ LCB medium (104) | 1.0 | — | 2.9 |
| Code | └ LCB hard (154) | 0.0 | — | 0.0 |
| Code | CRUXEval-O | 8.5 | — | 12.1 |
| Code | CRUXEval-I | 3.2 | — | 8.9 |
| Math | GSM8K | **32.4** (0-shot CoT) | 15.1 (8-shot) | 14.3 |
| Math | MATH-500 (0-shot CoT) | **21.0** | — | 14.8 |
| Instruction following | IFEval prompt strict | **44.7** | — | 35.7 |
| Instruction following | IFEval instruction strict | **56.6** | — | 49.2 |
| Knowledge | MMLU (5-shot) | 28.6 | 28.2 | 28.3 |
| Commonsense | HellaSwag (acc_norm) | 62.2 | 60.8 | 63.1 |
| Commonsense | ARC-e (acc_norm) | 64.1 | 66.7 | 64.6 |
| Commonsense | ARC-c (acc_norm) | 38.4 | 39.7 | 39.2 |
| Commonsense | PIQA (acc_norm) | 74.8 | 73.9 | 74.4 |
| Commonsense | WinoGrande (acc) | 61.8 | 60.1 | 63.3 |
| Commonsense | LAMBADA (acc) | 52.8 | 54.5 | 51.2 |

- SFT raised code, math and instruction following; knowledge and commonsense stayed at the pretraining level (within ±2).
- base GSM8K is 8-shot, so it is not the same protocol as the SFT 0-shot CoT score.

## 2. Model card

| Item | Value |
|:--|:--|
| Architecture | Decoder-only Transformer, pre-norm, **every layer is MoE** (no dense layers) |
| Layers · d_model | 32 · 2,048 |
| Attention | GQA 16Q / 4KV, head_dim 128, QK-RMSNorm, no bias |
| Position | RoPE θ = 10,000 · context 4,096 (document-boundary masking) |
| FFN | SwiGLU MoE: 16 experts, **top-4** per token, expert width 1,024 (FFN width per token 4,096) |
| Router | Linear 2048→16, fp32 softmax → top-4 → renormalized probabilities |
| Load balancing | aux loss 0.01 (global-batch statistics) + router z-loss 1e-3 |
| Vocabulary | 151,936 (Qwen3 tokenizer), tied embedding–lm_head |
| HF format | 1:1 mapping onto `Qwen3MoeForCausalLM` (export check: logits relative diff 1.0e-6, next-token agreement 100%) |

| Component | Parameters |
|:--|--:|
| Attention | 335.6M |
| Experts (32 × 16 × 3 × 2048 × 1024) | 3,221.2M |
| Router · norms | 1.2M |
| Embedding (= lm_head) | 311.2M |
| **Total** | **3,869,124,608** |
| **Active per token** | **1,453,205,504** (non-embedding 1,142,040,576) |

## 3. Training

| Stage | Tokens (cumulative) | Setup |
|:--|:--|:--|
| Pretrain 1 | 20.0B (20.0B) | 1× GH200, LR 4e-4 (1,000 warmup steps) |
| Pretrain 2 (CPT) | 46.1B (66.1B) | 1× GH200 → **2× GH200 DiLoCo** (from step 26,000, weight averaging every 250 steps), LR 4e-4 held |
| Pretrain 3 (decay) | 20.4B (**86.5B**) | 2× GH200 DiLoCo, LR 4e-4 → 0 linear, 9,750 steps |
| SFT stage 1 | 2.21B (1 epoch) | 1× H100, LR 2e-5 → 0, 2,130 steps |
| SFT stage 2 | 0.28B × 2 epochs | LR 1e-5 → 0, 536 steps |

- Global batch 1,048,576 tokens (256 × 4,096), 8-bit AdamW (β 0.9/0.95), weight decay 0.1, grad clip 1.0, bf16 compute + fp32 master weights.
- Pretraining total: 54,250 steps · 86,507,520,000 tokens.
- SFT validation loss: stage 1 0.627 → 0.577, stage 2 0.642 → 0.635 (no overfitting in the 2nd epoch).

## 4. Datasets

**Pretraining** (40 deduplicated sources)

| Span | Mix |
|:--|:--|
| Pretrain 1 | web 62 / code 22 / math 8 / curated 8 (FineWeb-Edu, DCLM, StarCoder-family, FineMath, Wikipedia, StackExchange, …) |
| Pretrain 2 | one full pass over the corpus (web · code in 11 languages · math · papers · StackExchange · synthetic) |
| Decay | web 30 / code 35 / math 15 / curated 20 |

**SFT** (ChatML, loss on assistant turns only, 13-gram decontamination against the eval sets)

| Stage | Composition |
|:--|:--|
| Stage 1, 2.21B (4.14M conversations) | general 29% (Dolci-Instruct) / code 44% (OpenCoder SFT, OpenCodeInstruct pass rate ≥ 0.9, self-oss, Code-Feedback) / math 27% (OpenMathInstruct-2, NuminaMath) |
| Stage 2, 0.28B | execution-verified code 59% / NuminaMath 21% / precise instruction following 20% |

## 5. Scoring

- Generation: vLLM greedy, chat template, prompt + answer ≤ 4,096 tokens.
- Model-written code runs in a bwrap sandbox (no network, `/` read-only). Every scorer was first checked with gold answers.

| Benchmark | Method |
|:--|:--|
| HumanEval(+) · MBPP(+) | EvalPlus 0.3.1, chat prompt |
| LiveCodeBench | code_generation_lite release v5·v6 additions (342 problems after 2024-08), official prompt, pass@1 |
| CRUXEval-O / -I | CoT then `[ANSWER]assert …[/ANSWER]`, executed |
| MultiPL-E C++ | full function combined with the tests, `g++ -std=c++17` |
| GSM8K · MATH-500 | 0-shot CoT, `\boxed{}` (MATH via math-verify) |
| IFEval | lm-eval 0.4.13, chat template |
| MMLU · 6 commonsense | lm-eval loglikelihood (no chat template), same as base |

## 6. Comparison with similar-size models

Other models' numbers come from official model cards and technical reports. Shots, CoT, IFEval metric and LiveCodeBench window differ between models, so treat this as a **rough comparison**.

| Model | Pretraining tokens | HE+ | MBPP+ | GSM8K | IFEval | MMLU |
|:--|--:|--:|--:|--:|--:|--:|
| **Apex-2 SFT (3.87B · 1.45B active)** | **0.087T** | **41.5** | **48.9** | **32.4** | **44.7** | **28.6** |
| Qwen2.5-1.5B-Instruct | 18T | (HE 61.6) | (MBPP 63.2) | 73.2 | 42.5 | 50.7 |
| Qwen2.5-Coder-1.5B-Instruct | 5.5T | 66.5 | 59.4 | — | — | — |
| Qwen3-1.7B (non-thinking) | 36T | — | — | — | 68.2 | 64.4 |
| Llama-3.2-1B-Instruct | 9T | — | — | 44.4 | 59.5* | 49.3 |
| Gemma-3-1B-it | 2T | (HE 41.5) | (MBPP 35.2) | 62.8 | 80.2* | 38.8 |
| OLMoE-1B-7B (1.3B active) | 5.1T | 54.4 | — | 72.4 | 66.4* | 55.1 |
| DeepSeek-Coder-1.3B | 2T | 60.4 | 54.8 | — | — | — |

<sub>Parenthesized values are HumanEval/MBPP. \* Different IFEval metric (Apex-2 · Qwen: prompt-level strict). Qwen2.5 · Qwen3 MMLU is MMLU-Redux; OLMoE MMLU uses CoT.</sub>

**Base models** (right after pretraining)

| Model | Pretraining tokens | HE / HE+ | MBPP / MBPP+ | MMLU | HellaSwag | PIQA | WinoGrande |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Apex-2 base** | **0.087T** | 36.6 / 32.9 | 54.8 / 46.3 | 28.2 | 60.8 | 73.9 | 60.1 |
| Qwen2.5-1.5B | 18T | 37.2 / 32.9 | 60.2 / 49.6 | 60.9 | 67.9 | — | 65.0 |
| Qwen2.5-Coder-1.5B | 5.5T | 43.9 / 36.6 | 69.2 / 58.6 | 53.6 | 61.8 | — | 60.7 |
| Gemma-2-2B | 2T | 17.7 / — | 29.6 / — | 51.3 | 73.0 | 77.8 | 70.9 |
| OLMoE-1B-7B | 5.1T | — | — | 56.3 | 81.7 | 78.7 | 70.6 |

- With 1/23 to 1/400 of the pretraining data, the base model's code score matches Qwen2.5-1.5B (18T): HE+ 32.9.
- The large gaps are knowledge (MMLU) and math, both of which track pretraining tokens.

<sub>Sources: Qwen2.5 [blog](https://qwenlm.github.io/blog/qwen2.5-llm/) · [report](https://arxiv.org/abs/2412.15115) · Qwen2.5-Coder [report](https://arxiv.org/abs/2409.12186) · Qwen3 [report](https://arxiv.org/abs/2505.09388) · [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) · Gemma 3 [report](https://arxiv.org/abs/2503.19786) · [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b-it) · [OLMoE Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) · [OLMoE base](https://huggingface.co/allenai/OLMoE-1B-7B-0125) · DeepSeek-Coder [paper](https://arxiv.org/abs/2401.14196)</sub>

## 7. DPO · RLVR log

**DPO — dropped**
- Data: 220K pairs from allenai/Dolci-Instruct-DPO (off-policy), length-normalized DPO (β 5), LR 1e-6.
- Validation accuracy was nearly saturated at step 100 (81%), while chosen NLL kept rising 1.023 → 1.161 (likelihood displacement: the chosen answers lose probability too). Stopped at step 300.
- In evaluation the mean answer length grew 320 → 733 tokens (2.3×), many more answers hit the length limit, and code, math and instruction following all dropped (§1).

**RLVR (GSPO) — stopped**
- Corpus: OpenCodeInstruct (unit tests) · GSM8K · MATH · DeepMath · IFBench train set (all permissively licensed); rewards from sandboxed execution, math-verify and IFEval constraints.
- Started from the DPO model and was stopped at step 33 after the DPO results came in.

## 8. Limitations

- English-centric. Korean and other languages barely work (multilingual data was excluded from SFT).
- Limited knowledge, so factual errors (hallucinations) are common (MMLU 28.6).
- Weak at competitive programming (LiveCodeBench medium 1%) and code-execution reasoning (CRUXEval).
- 4,096-token context.
