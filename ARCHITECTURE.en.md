# LLM From Scratch

[![KO](https://img.shields.io/badge/KO-lightgrey)](ARCHITECTURE.md) [![EN](https://img.shields.io/badge/EN-0969da)](ARCHITECTURE.en.md) [![JA](https://img.shields.io/badge/JA-lightgrey)](ARCHITECTURE.ja.md)

[← README](README.en.md) · Stuck on a term? [GLOSSARY](GLOSSARY.en.md)

---

This doc is about how the model is built, trained, and run for inference.
§2 walks through a Transformer forward pass with numeric examples; from §3 onward it covers this repo’s actual design (tokenizer · training · inference · thinking mode).

---

## 1. Overall roadmap

I didn't start with a perfect pipeline. Roughly, work went in this order.

| Stage | What it covers | Output |
|---|---|---|
| 0 | Environment setup (PyTorch + CUDA) | Dev environment |
| 1 | Tokenizer (BPE) | `tokenizer.json` |
| 2 | Model architecture (decoder-only Transformer) | `model.py` |
| 3 | Pretraining | base checkpoint |
| 4 | Supervised fine-tuning (SFT, chat format) | chat checkpoint |
| 5 | Thinking-mode SFT (`<THINKING>` data) | thinking checkpoint |
| 6 | Inference engine (KV cache + sampling + thinking-mode parsing) | `inference.py` |
| 7 | (post-deployment) Follow-up training from human feedback | aligned checkpoint — [POST-TRAINING.en.md](POST-TRAINING.en.md) |

---

## 2. How a Transformer works (pedagogical walkthrough)

Pedagogical numeric example (GPT-2-style GELU, etc.). The production design in this repo is the LLaMA-style stack in the next section (§3): RMSNorm / RoPE / SwiGLU / GQA.

### Transformer architecture

> This section describes how an LLM is trained.

To train an LLM you need a dataset. Datasets are also called a “corpus.” In this walkthrough the corpus is “I like cats,” shown with five demo tokens aligned to the original Korean example: I · cat · OBJ · like · do (OBJ stands in for an object marker; English word order is SVO, but we keep the same five-slot layout so the numbers match).


#### 1) Tokenizer

The tool that turns the corpus into a form a computer can work with is the **tokenizer**.

"I like cats" (demo split) → <kbd>I</kbd> <kbd>cat</kbd> <kbd>OBJ</kbd> <kbd>like</kbd> <kbd>do</kbd>

That split produces a dictionary called **vocab**.

```text
vocab = {"I": 1, "cat": 2, "OBJ": 3, "like": 4, "do": 5}
```

token ids: `[1, 2, 3, 4, 5]`


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


#### 3) Building the training problem

Shift the token sequence by one position to make input / target pairs.

Input: `[1, 2, 3, 4]` = "I cat OBJ like"<br>
Target: `[2, 3, 4, 5]` = "cat OBJ like do"

<kbd>I</kbd> → ? (answer: cat)<br>
<kbd>I</kbd> <kbd>cat</kbd> → ? (answer: OBJ)


#### 4) Normalization

x = embedding values. For the problem [I, cat, OBJ, like], x is the embedding of “like”.

x = <kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd>

After normalization, N:

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>


#### 5) Attention

This is the step that mixes context — the core of the Transformer architecture from here on.<br>
A plain bigram model cannot see context. The same “like” looks the same whether the previous context was “cat OBJ” or something else.

> [!TIP]
> 📌 **Three random matrices project N into three views**

Problem 4 — tokens `[1, 2, 3, 4]` = "I cat OBJ like", targets `[2, 3, 4, 5]` = "cat OBJ like do".<br>
The position we are predicting from is "like".

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>

Query `Q = N · W_Q`. W_Q, W_K, W_V start random at the beginning of training and are adjusted as training proceeds.

K(I) · Q ÷ √4 = 0.1<br>
K(cat) · Q ÷ √4 = 2.0<br>
K(OBJ) · Q ÷ √4 = −0.6<br>
K(like) · Q ÷ √4 = 0.5

softmax(<kbd>0.1</kbd> <kbd>2.0</kbd> <kbd>-0.6</kbd> <kbd>0.5</kbd>) = <kbd>0.10</kbd> <kbd>0.69</kbd> <kbd>0.05</kbd> <kbd>0.15</kbd>

"like" attended to "cat" at 69% — that is why the same word becomes a different vector depending on prior context.

Weighted sum of V: `0.10·V(I) + 0.69·V(cat) + 0.05·V(OBJ) + 0.15·V(like)`

Attention output = <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd> — a new vector with context mixed in.


#### 6) Residual connection

Add x and the attention output to get new v. This keeps the original meaning while stacking context on top.

<kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd> + <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd><br>
= <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd>

Still just numbers, but we are ready to capture context and learn.

---

### Block

From here we are in the block architecture.

#### 1) Normalization

Normalize new v.

N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd>


#### 2) FFN step 1 — expand with W₁

Expand N₂ by a fixed multiple. GPT-2 expands by 4×, so this example does the same.<br>
Attention output is only a weighted sum of V values — mixing and averaging — but the expanded tensor acts more like a set of “pattern detectors.”

Expand: `U = N·W₁ + b₁` (4×4 matrix · 4×16 matrix = 4×16; b₁ is a 16-dim bias, initialized to 0 so omitted)

> [!NOTE]
> **Legend**: <span style="color:#1d9e75">■</span> will pass (positive) · <span style="color:#d85a30">■</span> will be blocked (negative) · <span style="color:#888780">■</span> near zero

**Table A — before GELU (U)**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-1.53</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#1d9e75">1.96</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#d85a30">-2.58</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.29</span> | <span style="color:#d85a30">-1.37</span> | <span style="color:#d85a30">-0.28</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-0.26</span> | <span style="color:#d85a30">-0.65</span> | <span style="color:#1d9e75">2.15</span> | <span style="color:#1d9e75">1.48</span> |
| cat | <span style="color:#d85a30">-1.96</span> | <span style="color:#d85a30">-0.90</span> | <span style="color:#1d9e75">1.93</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-1.33</span> | <span style="color:#1d9e75">0.89</span> | <span style="color:#d85a30">-2.79</span> | <span style="color:#1d9e75">0.39</span> | <span style="color:#1d9e75">2.57</span> | <span style="color:#d85a30">-1.80</span> | <span style="color:#d85a30">-0.58</span> | <span style="color:#1d9e75">0.77</span> | <span style="color:#d85a30">-1.60</span> | <span style="color:#1d9e75">1.57</span> | <span style="color:#d85a30">-1.06</span> | <span style="color:#d85a30">-2.67</span> |
| OBJ | <span style="color:#1d9e75">2.35</span> | <span style="color:#1d9e75">0.64</span> | <span style="color:#d85a30">-2.43</span> | <span style="color:#d85a30">-2.82</span> | <span style="color:#1d9e75">2.28</span> | <span style="color:#d85a30">-1.91</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">0.50</span> | <span style="color:#d85a30">-2.27</span> | <span style="color:#1d9e75">0.94</span> | <span style="color:#1d9e75">0.31</span> | <span style="color:#d85a30">-0.76</span> | <span style="color:#1d9e75">0.90</span> | <span style="color:#d85a30">-1.31</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.10</span> |
| like | <span style="color:#d85a30">-0.88</span> | <span style="color:#d85a30">-2.45</span> | <span style="color:#1d9e75">1.63</span> | <span style="color:#1d9e75">3.40</span> | <span style="color:#d85a30">-1.20</span> | <span style="color:#1d9e75">2.67</span> | <span style="color:#d85a30">-3.24</span> | <span style="color:#1d9e75">2.07</span> | <span style="color:#1d9e75">0.79</span> | <span style="color:#d85a30">-0.62</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.21</span> | <span style="color:#1d9e75">0.82</span> | <span style="color:#d85a30">-0.63</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-1.77</span> |


#### 3) FFN step 2 — GELU

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

Table A “like” d7 = <span style=”color:#d85a30”>-3.24</span> → Table B <span style=”color:#888780”>0.00</span> — almost wiped out; you can see GELU “blocking” a strong negative.<br>
Conversely d4 = <span style="color:#1d9e75">3.40</span> → <span style="color:#1d9e75">3.39</span> almost fully “passed.”

📌 From here we keep tracking only the prediction position "like".<br>
N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd> → after GELU (Table B row "like"):<br>
`-0.17 -0.02 1.55 3.39 -0.14 2.66 0.00 2.03 0.62 -0.17 0.00 -0.09 0.65 -0.17 -0.16 -0.07`


#### 4) Project down

Matrix multiply by W₂ (16×4). “Summarize 16 signals by a weighted sum back into 4 dims.”<br>
Result: <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd>


#### 5) FFN output

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

Higher means the model believes that word more, but the key answer "do" was correct. (Weights are random, so this outcome is expected.)<br>
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

That finishes the forward pass. Loss is `−ln(prob of do) = −ln(0.0000000126) ≈ 18.2`. That is quite high; next we improve the weights with backpropagation.

For the actual implementation stack, see [§3 Model architecture design](#3-model-architecture-design) below.

---

## 3. Model architecture design

### 3.1 Basic structure — Decoder-only Transformer

At first I wondered if plain GPT-2 would be enough.
Looking at recent recipes (the LLaMA side of things), there is quite a bit worth changing.
If a term is unfamiliar, see [GLOSSARY](GLOSSARY.en.md#model-architecture).

```text
Input Tokens
   │
Token Embedding (shared with output layer via weight tying)
   │
[ Transformer Block ] × N
   ├─ RMSNorm (Pre-Norm)
   ├─ Self-Attention (Causal, GQA + RoPE + Flash Attention)
   ├─ Residual connection
   ├─ RMSNorm
   ├─ FFN (SwiGLU)
   └─ Residual connection
   │
RMSNorm (final)
   │
LM Head (Linear → vocab logits)
```

| Component | GPT-2 (2019) | This design | Why |
|---|---|---|---|
| Normalization | LayerNorm | **RMSNorm** | Simpler, more stable training |
| Positional info | Learned absolute position | **RoPE** | Context extension is relatively easier later |
| Activation | GELU | **SwiGLU** | Better performance |
| Attention | MHA | **GQA** (Grouped-Query) | Saves KV cache memory |
| Bias | Present | **Removed** | Saves parameters, more stable |

In one line: I picked these so **training breaks less, inference uses less memory, and length extension is at least somewhat easier**.

### 3.2 Model sizes (by GPU budget)

| Preset | Parameters | layers | d_model | heads (Q/KV) | ctx | VRAM needed (training) |
|---|---|---|---|---|---|---|
| **nano** (tutorial) | ~30M | 6 | 384 | 6 / 2 | 1024 | ~4GB |
| **small** (GPT-2 tier) | ~124M | 12 | 768 | 12 / 4 | 2048 | ~12GB |
| **base** (beyond GPT-2) | ~350M | 24 | 1024 | 16 / 4 | 4096 | ~24GB (or an A100 in the cloud) |

- vocab: **32,000** (BPE) — bump to 48k–64k if Korean is in the mix
- FFN hidden: `d_model × 8/3` (for SwiGLU, e.g. 768 → 2048)

The actual implementation ([llm/](llm/)) has been run end-to-end on **nano** — training, inference, and the feedback loop.
Scaling to small/base follows the same path as [README §2](README.en.md#2-scale-parameters-slowly).

### 3.3 Core code skeleton (PyTorch)

<details>
<summary><b>View the full model.py code</b> (click to expand)</summary>

```python
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12          # Query heads
    n_kv_head: int = 4        # KV heads (GQA)
    d_model: int = 768
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def apply_rope(x, cos, sin):          # x: (B, H, T, Dh)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.dh = cfg.d_model // cfg.n_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.dh, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.dh, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.dh, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:                      # KV cache for inference
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[:] = [k, v]
        k = k.repeat_interleave(self.n_head // self.n_kv, dim=1)   # expand for GQA
        v = v.repeat_interleave(self.n_head // self.n_kv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.d_model * 8 / 3 / 64) * 64   # align to a multiple of 64
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)  # down
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.ffn  = RMSNorm(cfg.d_model), SwiGLU(cfg)
    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, kv_cache)
        return x + self.ffn(self.norm2(x))

class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying
        # RoPE cos/sin buffer precompute omitted — use register_buffer

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        cos, sin = self.rope_cache(idx.shape[1])            # implementation omitted
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-100)
        return logits, loss
```

</details>

---

## 4. Tokenizer

- **BPE (byte-level)** — HuggingFace `tokenizers`
- Special tokens (important because of thinking mode):

```text
<|bos|>  <|eos|>  <|pad|>
<|user|>  <|assistant|>  <|system|>       ← conversation roles
<THINKING>  </THINKING>                    ← thinking-mode span
```

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=[
    "<|bos|>","<|eos|>","<|pad|>","<|user|>","<|assistant|>","<|system|>",
    "<THINKING>","</THINKING>"])
tok.train(files=["corpus.txt"], trainer=trainer)
tok.save("tokenizer.json")
```

---

## 5. Dataset strategy

Per the plan below, Korean / English / Japanese natural language plus code in 11 languages
was downloaded from non-gated public sources and organized under [llm/datasets/](llm/datasets/).
About 11GB total: pretraining text plus ~320k SFT/thinking rows.
Inventory and sources: [llm/datasets/README.md](llm/datasets/README.md).

### 5.1 Pretraining

| Preset | Dataset | Token count | Notes |
|---|---|---|---|
| nano | **TinyStories** | ~500M | Finish in a day; check grammatical generation |
| small | **FineWeb-Edu (10B sample)** | 5–10B | Roughly GPT-2 reproduction level |
| base | FineWeb-Edu + **The Stack (smol)** | 10–30B | **Coding** hinges on a 15–25% code ratio |
| Korean | + AI Hub / Modu Corpus / ko-wikipedia | +α | When Korean capability is needed |

Chinchilla in short ([README §2](README.en.md#2-scale-parameters-slowly)): optimal tokens ≈ parameters × 20  
(so a 124M model needs at least 2.5B tokens).

### 5.2 SFT (chat format)

- Public instruction data such as **OpenHermes-2.5**, **UltraChat**, **KoAlpaca**
- Format:

```text
<|system|>You are a helpful assistant.<|eos|>
<|user|>Write a quicksort implementation in Python<|eos|>
<|assistant|>...code...<|eos|>
```

### 5.3 Thinking-mode data

- Convert reasoning-trace datasets like **OpenThoughts / OpenR1-Math / Raiden-DeepSeek-R1** into the `<THINKING>` format
- Format:

```text
<|user|>What is the largest 3-digit prime number?<|eos|>
<|assistant|><THINKING>
The largest 3-digit number is 999. 999=3×333, composite. 998 is even. Check 997:
√997≈31.6, dividing by primes from 2 to 31... none divide evenly → prime.
</THINKING>
The largest 3-digit prime is **997**.<|eos|>
```

- **Loss masking**: user tokens get `-100` (excluded from loss).  
  Loss is applied over the whole assistant span, including `<THINKING>`, so the model learns to write the reasoning process itself.

---

## 6. Training pipeline

### 6.1 Hyperparameters (small preset)

| Item | Value |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| LR schedule | Warmup 2k steps → cosine decay |
| Peak LR | 6e-4 (pretrain) / 2e-5 (SFT) |
| Batch | ~0.5M effective tokens (gradient accumulation) |
| Precision | bfloat16 (mixed precision) |
| Grad clip | 1.0 |
| Other | `torch.compile`, Flash Attention (SDPA) |

### 6.2 Training loop skeleton

```python
model = GPT(cfg).cuda()
model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4,
                        betas=(0.9, 0.95), weight_decay=0.1)

for step, (x, y) in enumerate(loader):        # x,y: (B, T), recommend uint16 memmap
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x.cuda(), y.cuda())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    lr_scheduler.step()
    if step % 1000 == 0:
        save_checkpoint(model, opt, step)      # + log val loss
```

### 6.3 Step-by-step order

```text
[1] Pretraining  : raw text + code, next-token prediction        (days~weeks)
[2] SFT          : conversational format, user-turn masking       (hours)
[3] Thinking SFT : <THINKING> data mixed in (general:thinking = 3:7)  (hours)
    ※ Steps 2 and 3 can be merged into one pass. Thinking mode on/off
      can be controlled via the system prompt ("think step by step in <THINKING>")
[4] (optional) DPO/GRPO : RL on math/coding with verifiable answers → better reasoning (POST-TRAINING.en.md)
```

---

## 7. Inference engine

### 7.1 Generation loop (KV cache + thinking mode)

```python
@torch.no_grad()
def generate(model, tok, prompt, thinking=True,
             max_new=2048, temperature=0.7, top_p=0.9):
    sys = "You are a helpful assistant."
    if thinking:
        sys += " Reason step-by-step inside <THINKING> tags before answering."
    ids = tok.encode(f"<|system|>{sys}<|eos|><|user|>{prompt}<|eos|><|assistant|>")
    if thinking:
        ids += tok.encode("<THINKING>")        # force-start thinking mode (prefill)

    kv_caches = [[] for _ in model.blocks]     # per-layer KV cache
    x = torch.tensor([ids]).cuda()
    out = []
    for _ in range(max_new):
        logits = model.forward_with_cache(x, kv_caches)[:, -1]
        logits /= temperature
        probs = top_p_filter(F.softmax(logits, -1), top_p)
        next_id = torch.multinomial(probs, 1)
        if next_id.item() == tok.token_to_id("<|eos|>"):
            break
        out.append(next_id.item())
        x = next_id                             # thanks to the cache, only the new token is forwarded

    text = tok.decode(out)
    # split thinking span → collapse/hide in the UI
    thought, _, answer = text.partition("</THINKING>")
    return {"thinking": thought.strip(), "answer": answer.strip()}
```

### 7.2 How thinking mode works

1. **Training**: SFT on assistant replies that contain `<THINKING>...</THINKING>` → the model picks up “think first, answer second”  
2. **Inference**: prefill `<THINKING>` right after the assistant turn to force reasoning (off = skip prefill + change system prompt)  
3. **Post-processing**: split thinking / answer at `</THINKING>`. Multi-turn history stores **answer only** (saves context)  
4. **Safety net**: if thinking tokens pass a cap (e.g. 1024), force-insert `</THINKING>`  

---

## 8. Evaluation

| Capability | Benchmark | Tool |
|---|---|---|
| Language understanding | HellaSwag, ARC, MMLU (subset) | `lm-evaluation-harness` |
| Coding | **HumanEval, MBPP** | pass@1 scoring script |
| Math / reasoning | GSM8K | compare thinking on/off |
| Basic | validation loss / perplexity | in-house |

---

## 9. Realistic expectations and scaling

- **Pretraining nano/small from scratch**: fluent generation around original GPT-2 level, plus simple instructions, is realistic.  
  Real coding ability is not something to expect at this scale.
- **To actually improve coding**: ① raise the code-data ratio (The Stack), ② scale to 350M–1B+,  
  ③ add thinking mode plus coding SFT (e.g. Magicoder-OSS-Instruct).
- **Shortcut**: grab a public small base (Qwen2.5-0.5B/1.5B, etc.) and only run  
  SFT → Thinking → inference from [POST-TRAINING](POST-TRAINING.en.md).  
  You still learn the same pipeline, and end up with quality far more usable than original GPT-2.
- References: `karpathy/nanoGPT`, `karpathy/build-nanogpt`, HuggingFace `smol-course`.

---

[GLOSSARY](GLOSSARY.en.md) · [README](README.en.md) · [POST-TRAINING](POST-TRAINING.en.md) · [BENCHMARK](BENCHMARK-v1.en.md)
