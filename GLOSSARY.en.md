<div align="center">

# Glossary

[한국어](GLOSSARY.md) &nbsp;·&nbsp; **[English](GLOSSARY.en.md)** &nbsp;·&nbsp; [日本語](GLOSSARY.ja.md)

[← Back to README.en.md](README.en.md)

</div>

---

So that anyone brand new to LLMs can still get through [ARCHITECTURE.en.md](ARCHITECTURE.en.md) and
[POST-TRAINING.en.md](POST-TRAINING.en.md), I've laid out the terms those docs use right here first.
Skip whatever you already know, and if you hit a word that trips you up while reading, just come back
and look it up.

## Model Architecture Terms

| Term | Explanation |
|---|---|
| **Transformer** | A neural network architecture built around self-attention as its core operation. Pretty much every LLM since 2017 — GPT, Llama, DeepSeek, take your pick — runs on top of this. |
| **Decoder-only** | The flavor of Transformer that only stacks decoder blocks: the kind that predicts the next token by looking at everything generated so far. The GPT family lives here, and so does this project. |
| **Parameter** | The total count of numbers that get updated during training. Say "a 300M model" and you mean 300 million parameters — it's the standard shorthand for a model's "size." |
| **Weight** | The actual values a parameter holds. A single `nn.Linear(768, 768)` layer, for example, has 768×768 weights, and training is just the process of nudging these numbers little by little. |
| **Weight tying** | A trick where the input embedding layer and the output layer (LM Head) share the same weight matrix. Cuts parameter count and helps stabilize training. |
| **Tokenizer** | The tool that converts text into integer token IDs the model can actually work with. If "Python is" exists in the vocab, it gets turned into a single integer like 4123. The output here is an integer, not a vector — turning that integer into an actual numeric vector is the Embedding layer's job, right after this. |
| **Embedding** | The layer that turns each token into a fixed-length vector of numbers. It's what lets you handle language as vector math instead of literal characters. |
| **Vision Encoder** | A separate architecture component that turns images into vectors the model can understand. "Images are ultimately just pixels — i.e., numbers" isn't, on its own, enough to make a text LLM understand images; you need this dedicated structure for that. This project is text/code-only for now, so it has no Vision Encoder. |
| **RMSNorm** | One flavor of layer normalization — the trick of keeping value magnitudes in a consistent range so training doesn't go haywire. Simpler to compute than the older LayerNorm, with roughly the same performance. |
| **Pre-Norm** | Placing normalization *before* the attention/FFN operations rather than after. Keeps training from falling apart as you stack more layers. |
| **RoPE (Rotary Position Embedding)** | A positional encoding scheme that bakes "which position is this token at" into the attention computation via a rotation. Because position is computed from a formula rather than learned as a fixed parameter, extending context length later (NTK-aware scaling, YaRN, etc.) is relatively easier than with learned absolute position embeddings. That said, RoPE isn't magic — push it well past its trained length with no correction and performance still degrades, same as anything else. |
| **Self-Attention** | The operation where every token in a sequence looks at every other token and computes "how related are you and me." This is the heart of the Transformer. |
| **Causal masking** | The mechanism that blocks a token from peeking at future tokens. Non-negotiable if you want to keep the "predict the next token" objective honest. |
| **MHA (Multi-Head Attention)** | Splitting self-attention into several independent "heads" that run in parallel. Each head can learn a different kind of relationship — grammar, meaning, whatever it lands on. |
| **GQA (Grouped-Query Attention)** | An attention variant that uses fewer Key/Value heads than Query heads, shrinking the KV cache you need to keep around at inference time. Saves memory compared to plain MHA. |
| **KV Cache (Key-Value Cache)** | Caching the Key/Value values already computed during inference and reusing them for the next token, instead of recomputing the whole sequence from scratch every single step. |
| **Flash Attention / SDPA** | A kernel implementation that computes self-attention fast while using way less GPU memory. PyTorch's `F.scaled_dot_product_attention` gives you this out of the box. |
| **FFN (Feed-Forward Network)** | The fully-connected network that comes after attention and gets applied to each token independently. |
| **SwiGLU** | The activation combo used in the FFN. Reportedly outperforms the older GELU, which is why most recent LLMs have adopted it. |
| **Residual connection** | A connection that adds a layer's input straight back onto its output. Lets you stack layers deep without information vanishing, and keeps training stable. |
| **LM Head** | The final linear layer that converts the model's last hidden state into vocab-sized scores (logits). |
| **Logits** | The raw score vector — before softmax — representing how plausible each candidate next token is. |

## Training Terms

| Term | Explanation |
|---|---|
| **Pretraining** | The first training stage: feed the model a mountain of raw text and code, train it on next-token prediction, and let it soak up language structure, world knowledge, and coding patterns. |
| **Next-token prediction** | "Look at everything so far and guess what comes next" — the most basic objective in all of LLM training. Example: "Python is" → predict "a" → feed back "Python is a" → predict "programming" → feed back "Python is a programming" → predict "language." → repeat until you land on "Python is a programming language." |
| **Fine-tuning** | Taking an already-pretrained model and training it further on smaller, more targeted data. |
| **SFT (Supervised Fine-Tuning)** | Fine-tuning in a supervised fashion on (question, ideal answer) pairs that a human wrote or vetted. |
| **Instruction tuning** | Fine-tuning on conversational-format data so the model actually follows what users ask it to do. A specific flavor of SFT. |
| **Loss** | A number that measures how far off the model's prediction is from the correct answer. Training is just nudging the parameters, step by step, to push this number down. |
| **Cross-entropy** | The loss function almost everyone uses for language model training. The lower the probability it assigned to the correct token, the bigger the loss. |
| **Loss masking** | Excluding certain tokens — say, the user's question — from the loss calculation (marking them `-100`), so the model isn't trained to generate that part, only the answer part. |
| **Perplexity** | The exponential of validation loss. Shows how well the model predicts the next token; lower is better. |
| **Epoch / Step / Batch** | One full pass over the data (epoch), one parameter update (step), and a chunk of data processed together at once (batch). |
| **Gradient accumulation** | When GPU memory won't let you fit a big batch all at once, you accumulate gradients over several steps and then apply a single update — same effect as a bigger batch, just spread out. |
| **Optimizer / AdamW** | The algorithm that actually updates the parameters in the direction that reduces loss. AdamW is basically the default for LLM training at this point. |
| **Learning rate / Warmup / Cosine decay** | The learning rate is how big a step you take when updating parameters. The usual schedule ramps it up gradually at first (warmup), then eases it back down along a cosine curve (decay). |
| **Gradient clipping** | Capping how large gradients are allowed to get so training doesn't blow up and diverge. |
| **Mixed precision / bfloat16** | Run the expensive parts — matrix multiplies, activations — in 16-bit (bfloat16) to save time and memory, while keeping the master weights and optimizer state that actually need precision in 32-bit. Mixing two precisions like that is exactly why it's called "mixed precision." |
| **Checkpoint** | A saved snapshot of the model's weight state, taken during or after training. A file like `ckpt/sft_nano.pt` is one of these. |
| **Chinchilla scaling law / Compute-optimal** | A rule of thumb for how to split a fixed compute budget between model size and training-token count to squeeze the most out of it. Covered in more depth in [README.en.md §2](README.en.md#2-parameter-count). |
| **FIM (Fill-In-the-Middle)** | A task where the model is given the start and end of some text (or code) and has to fill in the middle. Essential for code autocomplete. |

## Inference and Generation Terms

| Term | Explanation |
|---|---|
| **Context window** | The maximum number of tokens the model can look at in one shot. |
| **Sampling** | Turning logits into a probability distribution, then randomly drawing the next token from it. This is what keeps output from being identical every time and gives you variety. |
| **Temperature** | A knob that controls how flat (more varied) or peaked (more confident) that sampling distribution gets. |
| **Top-p (nucleus sampling)** | Only sample from the smallest set of top candidates whose cumulative probability exceeds p. Keeps weird, low-probability tokens from sneaking through. |
| **THINKING mode (Chain-of-Thought)** | Have the model write out its reasoning inside a `<THINKING>...</THINKING>` block before the final answer, which boosts accuracy on harder problems. Covered in depth in [ARCHITECTURE.en.md](ARCHITECTURE.en.md). |
| **Prefill** | The inference stage where the entire prompt is run through in one pass to pre-populate the KV cache. |

## RAG and Tool Terms

| Term | Explanation |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Retrieve documents relevant to the question from some external store, stuff them into the prompt, then generate the answer from there. Lets you bring in fresh knowledge without retraining the model. |
| **Tool use / Function calling** | The model's ability to decide for itself — "I need a calculator right now," "I need to run some code right now" — and call out to an external tool. |

## Post-Training and Alignment Terms

| Term | Explanation |
|---|---|
| **RLHF (Reinforcement Learning from Human Feedback)** | The umbrella term for using human preference feedback as a reward signal to align a model — i.e., steer it toward what people actually want — via reinforcement learning. |
| **Preference pair / Chosen / Rejected** | Two answers to the same question, labeled by a human as the better one (chosen) and the worse one (rejected). |
| **DPO (Direct Preference Optimization)** | Aligning a model directly from preference-pair data, no separate reward model needed. Simpler and more stable to implement than RLHF. |
| **Reward Model (RM)** | A model trained to score answers. Usually built by swapping an LLM's final layer for a scalar (single-number) output and training that. |
| **GRPO (Group Relative Policy Optimization)** | A reinforcement-learning technique that samples several answers to the same question at once and updates the model based on each answer's relative standing (advantage) within that group. Unlike PPO, it needs no separate value model, which keeps it simpler. |
| **RLVR (Reinforcement Learning from Verifiable Rewards)** | For problems you can grade automatically — math, coding — this swaps out the reward model for a rule-based reward, like "does the answer match exactly." |
| **KL divergence / KL penalty** | A penalty term that keeps the model being trained from drifting too far from the original (reference) model. Keeps alignment training from quietly breaking the model. |
| **Rejection sampling** | Generate a bunch of candidate answers, keep only the ones that clear some bar (a score threshold, etc.), and reuse those as training data. |
| **Reward hacking** | When the model learns to game the reward score instead of actually improving answer quality. |

## Evaluation Terms

| Term | Explanation |
|---|---|
| **Benchmark** | A standardized problem set for measuring a specific model capability. E.g., HellaSwag/ARC/MMLU (language understanding), HumanEval/MBPP (coding), GSM8K (math). |
| **pass@1** | In code-generation evaluation, the fraction of the time the model's first generated attempt passes the tests. |

<div align="center">

---

[← README.en.md](README.en.md) &nbsp;|&nbsp; [ARCHITECTURE.en.md →](ARCHITECTURE.en.md)

</div>
