[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](GLOSSARY.md) [![English](https://img.shields.io/badge/English-0969DA?style=flat-square)](GLOSSARY.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](GLOSSARY.ja.md)

# GLOSSARY

[← README](README.en.md)

---

> [!TIP]
> This doc is a dictionary.  
> If you get stuck while reading [ARCHITECTURE](ARCHITECTURE.en.md) or [POST-TRAINING](POST-TRAINING.en.md), come here.  
> You don't need to memorize everything up front. Just look up the words you don't know.

---

## Model architecture

| Term | Explanation |
|---|---|
| **Transformer** | A neural network built around self-attention as its core operation. Almost every LLM since 2017 (GPT, Llama, DeepSeek, …) runs on this. |
| **Decoder-only** | Only decoder blocks that “look at tokens so far and predict the next.” GPT family. This project too. |
| **Parameter** | Count of numbers updated by training. A “300M model” has 300 million parameters. Usually called model “size.” |
| **Weight** | The actual values those parameters hold. e.g. `nn.Linear(768, 768)` has 768×768 weights. Training nudges these numbers. |
| **Weight tying** | Input embedding and output layer (LM Head) share the same weight matrix. Fewer parameters, more stable training. |
| **Tokenizer** | Turns text into integer token IDs. “Python is” → 4123, that kind of thing. Output is integers, not vectors; Embedding does the vector step next. |
| **Embedding** | Layer that turns one token into a fixed-length numeric vector. Language is handled as vector math, not raw characters. |
| **Vision Encoder** | Separate structure that turns images into vectors the model can use. “Pixels are numbers” does not mean a text LLM understands images out of the box. This project is text/code-only, so none here. |
| **RMSNorm** | A layer-normalization method. Simpler compute than LayerNorm, similar performance. |
| **Pre-Norm** | Apply normalization *before* attention/FFN. Training holds up better when stacks get deep. |
| **RoPE (Rotary Position Embedding)** | Puts token position into attention via a rotation. Computed from a formula, so length extension (NTK-aware, YaRN, …) is relatively easier. Still degrades if you go far past trained length with no fix. |
| **Self-Attention** | Tokens in a sequence look at each other and score relatedness. Core of the Transformer. |
| **Causal masking** | Blocks peeking at future tokens. Required for next-token prediction. |
| **MHA (Multi-Head Attention)** | Split self-attention into several heads in parallel. Different heads can learn different relations (syntax, meaning, …). |
| **GQA (Grouped-Query Attention)** | Fewer KV heads than Query heads, so inference KV cache is smaller. Saves memory vs MHA. |
| **KV cache (Key-Value Cache)** | Store already-computed Key/Value and reuse on the next token. Avoids recomputing the whole sequence every step. |
| **Flash Attention / SDPA** | Kernels that run self-attention with less GPU memory and more speed. PyTorch `F.scaled_dot_product_attention`. |
| **FFN (Feed-Forward Network)** | Fully connected net after attention, applied per token independently. |
| **SwiGLU** | FFN activation combo. Said to beat GELU; common in recent LLMs. |
| **Residual connection** | Add the layer input back onto its output. Information survives deep stacks; training stays stable. |
| **LM Head** | Final linear layer from last hidden state to vocab-sized logits. |
| **Logits** | Raw “how plausible is this next token” scores before softmax. |

---

## Training

| Term | Explanation |
|---|---|
| **Pretraining** | First stage: next-token prediction on large raw text/code. Language structure, knowledge, coding patterns soak in here. |
| **Next-token prediction** | “Given tokens so far, predict the next.” e.g. “Python is” → “a programming” → … until a sentence forms. |
| **Fine-tuning** | Train a pretrained model further on smaller, goal-focused data. |
| **SFT (Supervised Fine-Tuning)** | Supervised fine-tuning on (question, ideal answer) pairs. |
| **Instruction tuning** | Fine-tune in chat form so the model follows instructions. A form of SFT. |
| **Loss** | How far the prediction is from the target. Training moves parameters to lower this. |
| **Cross-entropy** | The usual language-model loss. Lower probability on the correct token → larger loss. |
| **Loss masking** | Exclude some tokens (e.g. user question) from loss (`-100`). Model learns the answer side, not to “generate” the question. |
| **Perplexity** | Exp of validation loss. Lower means better next-token prediction. |
| **Epoch / Step / Batch** | Full pass over data (epoch), one parameter update (step), chunk processed together (batch). |
| **Gradient accumulation** | When a big batch won't fit, stack gradients over several steps then update once. |
| **Optimizer / AdamW** | Actually updates parameters to reduce loss. AdamW is the de facto default. |
| **Learning rate / Warmup / Cosine decay** | Step size (lr): ramp up first (warmup), then ease down on a cosine (decay). |
| **Gradient clipping** | Cap gradient size so training doesn't explode. |
| **Mixed precision / bfloat16** | Heavy ops in 16-bit (bfloat16); important weights/optimizer state in 32-bit. Speed and memory. |
| **Checkpoint** | Snapshot file of weights. Something like `ckpt/sft_nano.pt`. |
| **Chinchilla scaling law** | How to grow model size vs token count under fixed compute. [README §2](README.en.md#2-scale-parameters-slowly). |
| **FIM (Fill-In-the-Middle)** | Given prefix and suffix, fill the middle. Important for code autocomplete. |

---

## Inference · generation

| Term | Explanation |
|---|---|
| **Context window** | Max tokens the model can attend to at once. |
| **Sampling** | logits → distribution → random next token. So answers aren't identical every time. |
| **Temperature** | Flatten (more variety) or sharpen (more confident) the distribution. |
| **Top-p (nucleus sampling)** | Keep only top candidates whose cumulative mass exceeds p, then sample. Cuts weird rare tokens. |
| **THINKING mode (Chain-of-Thought)** | Write reasoning in `<THINKING>...</THINKING>` before the final answer. [ARCHITECTURE](ARCHITECTURE.en.md). |
| **Prefill** | Run the whole prompt in one pass to fill the KV cache first. |

---

## RAG / Tool

| Term | Explanation |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Retrieve related docs externally, put them in the prompt, then generate. Fresh knowledge without retraining. |
| **Tool use / Function calling** | Model ability to call external tools (calculator, code runner, …). |

---

## Post-training · alignment

| Term | Explanation |
|---|---|
| **RLHF** | Align a model with RL using human preference feedback as reward. Umbrella term. |
| **Preference pair / Chosen / Rejected** | Two answers to the same question: better (chosen) and worse (rejected). |
| **DPO (Direct Preference Optimization)** | Align from preference pairs only, no reward model. Simpler and more stable than full RLHF. |
| **Reward Model, RM** | Model trained to score answers. Often last layer swapped for a scalar head. |
| **GRPO** | Sample several answers per prompt; update from relative standing (advantage) inside the group. No value model unlike PPO. |
| **RLVR** | On auto-gradable math/coding, use rule rewards (exact match, tests pass) instead of an RM. |
| **KL divergence / KL penalty** | Term that keeps the new model from drifting too far from a reference model. |
| **Rejection sampling** | Generate many candidates; keep only those that clear a bar for training data. |
| **Reward hacking** | Learning to game the reward score without actually improving answer quality. |

---

## Evaluation

| Term | Explanation |
|---|---|
| **Benchmark** | Standard problem set to measure ability. e.g. HellaSwag/MMLU (language), HumanEval/MBPP (coding), GSM8K (math). |
| **pass@1** | Fraction of times the first generated code attempt passes tests. |

---

[README](README.en.md) · [ARCHITECTURE](ARCHITECTURE.en.md) · [POST-TRAINING](POST-TRAINING.en.md)
