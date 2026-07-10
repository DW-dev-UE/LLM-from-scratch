<div align="center">

# Model Architecture · Training · Inference

<img src="https://img.shields.io/badge/CUDA-enabled-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />

[한국어](ARCHITECTURE.md) &nbsp;·&nbsp; **[English](ARCHITECTURE.en.md)** &nbsp;·&nbsp; [日本語](ARCHITECTURE.ja.md)

[← README.en.md](README.en.md) &nbsp;|&nbsp; New to the terminology? Start with [GLOSSARY.en.md](GLOSSARY.en.md)

</div>

---

This covers building, training, and running inference on a small general-purpose LLM that's at least
GPT-2 (the original GPT) caliber — including bolting on a `<THINKING>`-tag-based thinking mode along the way.

## Table of Contents

1. [Overall Roadmap](#1-overall-roadmap)
2. [Model Architecture Design](#2-model-architecture-design)
3. [Tokenizer](#3-tokenizer)
4. [Dataset Strategy](#4-dataset-strategy)
5. [Training Pipeline](#5-training-pipeline)
6. [Inference Engine](#6-inference-engine)
7. [Evaluation](#7-evaluation)
8. [Realistic Expectations and Scaling Paths](#8-realistic-expectations-and-scaling-paths)

---

## 1. Overall Roadmap

| Stage | What it covers | Output |
|---|---|---|
| 0 | Set up the environment (PyTorch + CUDA) | Dev environment |
| 1 | Tokenizer (BPE) | `tokenizer.json` |
| 2 | Model architecture (decoder-only Transformer) | `model.py` |
| 3 | Pretraining | base checkpoint |
| 4 | Supervised fine-tuning (SFT, chat format) | chat checkpoint |
| 5 | Thinking-mode SFT (`<THINKING>` data) | thinking checkpoint |
| 6 | Inference engine (KV cache + sampling + thinking-mode parsing) | `inference.py` |
| 7 | (post-deployment) Follow-up training from human feedback | aligned checkpoint — [POST-TRAINING.en.md](POST-TRAINING.en.md) |

---

## 2. Model Architecture Design

### 2.1 Basic Structure — Decoder-only Transformer (Modern Recipe)

This uses more modern components than vanilla GPT-2 — basically the LLaMA-style recipe. If any of
these component names throw you off, look them up one at a time in
[GLOSSARY.en.md](GLOSSARY.en.md#model-architecture-terms).

```
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

| Component | GPT-2 (2019) | This design (recommended) | Why |
|---|---|---|---|
| Normalization | LayerNorm | **RMSNorm** | Simpler, more stable training |
| Positional info | Learned absolute position | **RoPE** | Extending context length later is comparatively easy |
| Activation | GELU | **SwiGLU** | Better performance |
| Attention | MHA | **GQA** (Grouped-Query) | Saves KV cache memory |
| Bias | Present | **Removed** | Saves parameters, more stable |

### 2.2 Model Sizes (Three Tiers by GPU Budget)

| Preset | Parameters | layers | d_model | heads (Q/KV) | ctx | VRAM needed (training) |
|---|---|---|---|---|---|---|
| **nano** (tutorial) | ~30M | 6 | 384 | 6 / 2 | 1024 | ~4GB |
| **small** (GPT-2 tier) | ~124M | 12 | 768 | 12 / 4 | 2048 | ~12GB |
| **base** (beyond GPT-2) | ~350M | 24 | 1024 | 16 / 4 | 4096 | ~24GB (or an A100 in the cloud) |

- vocab: **32,000** (BPE) — bump to 48k–64k if Korean is in the mix
- FFN hidden: `d_model × 8/3` (for SwiGLU, e.g. 768 → 2048)

> The actual implementation in this repo ([llm/](llm/)) has been verified end-to-end on the **nano**
> preset — training, inference, the whole feedback loop, all of it working. Scaling up to small/base
> just follows the same strategy laid out in [README.en.md §2](README.en.md#2-parameter-count).

### 2.3 Core Code Skeleton (PyTorch)

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

## 3. Tokenizer

- **BPE (byte-level)** — uses the HuggingFace `tokenizers` library
- Special-token design (this is the key part for thinking mode):

```
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

## 4. Dataset Strategy

> **Implementation status**: real Korean/English/Japanese natural-language and 11-language code datasets
> have been downloaded from non-gated public sources per the strategy below, and are organized under
> [llm/datasets/](llm/datasets/) — about 11GB total, spanning pretraining text plus 320k+ SFT/thinking-mode
> rows. Full inventory and sources: [llm/datasets/README.md](llm/datasets/README.md) (Korean only).

### 4.1 Pretraining

| Preset | Dataset | Token count | Notes |
|---|---|---|---|
| nano | **TinyStories** | ~500M | Doable in a day, good for confirming grammatical generation |
| small | **FineWeb-Edu (10B sample)** | 5-10B | Roughly GPT-2-reproduction quality |
| base | FineWeb-Edu + **The Stack (smol)** | 10-30B | **Coding performance** hinges on a 15-25% code-data ratio |
| Korean | + AI Hub / Modu Corpus / ko-wikipedia | +α | If Korean capability is needed |

> **Chinchilla scaling law** (see [README.en.md §2](README.en.md#2-parameter-count)): optimal token
> count ≈ parameters × 20 (so a 124M model needs at least 2.5B tokens).

### 4.2 SFT (Conversational Format)

- Public instruction datasets like **OpenHermes-2.5**, **UltraChat**, **KoAlpaca** (Korean)
- Format:

```
<|system|>You are a helpful assistant.<|eos|>
<|user|>Write a quicksort implementation in Python<|eos|>
<|assistant|>...code...<|eos|>
```

### 4.3 Thinking-Mode Data (Core)

- Convert public reasoning-trace datasets like **OpenThoughts / OpenR1-Math / Raiden-DeepSeek-R1**
  into the `<THINKING>` format
- Format:

```
<|user|>What is the largest 3-digit prime number?<|eos|>
<|assistant|><THINKING>
The largest 3-digit number is 999. 999=3×333, composite. 998 is even. Check 997:
√997≈31.6, dividing by primes from 2 to 31... none divide evenly → prime.
</THINKING>
The largest 3-digit prime is **997**.<|eos|>
```

- **Loss masking**: user tokens get marked `-100` (excluded from the loss), while the loss is applied
  across the *entire* assistant span — `<THINKING>` block included — so the model actually learns to
  produce the reasoning process on its own.

---

## 5. Training Pipeline

### 5.1 Hyperparameters (small preset)

| Item | Value |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| LR schedule | Warmup 2k steps → cosine decay |
| Peak LR | 6e-4 (pretrain) / 2e-5 (SFT) |
| Batch | ~0.5M effective tokens (via gradient accumulation) |
| Precision | bfloat16 (mixed precision) |
| Grad clip | 1.0 |
| Other | `torch.compile`, Flash Attention (SDPA) |

### 5.2 Training Loop Skeleton

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

### 5.3 Step-by-Step Order

```
[1] Pretraining  : raw text + code, next-token prediction        (days~weeks)
[2] SFT          : conversational format, user-turn masking       (hours)
[3] Thinking SFT : <THINKING> data mixed in (general:thinking = 3:7)  (hours)
    ※ Steps 2 and 3 can be merged into one pass. Thinking mode on/off can be
      controlled via the system prompt ("think step by step in <THINKING>")
[4] (optional) DPO/GRPO : RL on math/coding problems with verifiable answers → improves reasoning quality (see POST-TRAINING.md)
```

---

## 6. Inference Engine

### 6.1 Generation Loop (KV Cache + Thinking Mode)

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
        x = next_id                             # thanks to the cache, only the single new token needs a forward pass

    text = tok.decode(out)
    # split off the thinking span → handled as collapse/hide in the UI
    thought, _, answer = text.partition("</THINKING>")
    return {"thinking": thought.strip(), "answer": answer.strip()}
```

### 6.2 How Thinking Mode Actually Works

1. **Training**: SFT on assistant responses that include `<THINKING>...</THINKING>` → the model
   internalizes a "think first, answer second" pattern.
2. **Inference**: right as the assistant turn starts, prefill the `<THINKING>` token to force
   reasoning to kick in (thinking-off mode just skips the prefill and swaps the system prompt).
3. **Post-processing**: split thinking from the answer at `</THINKING>`, and store **only the
   answer** in multi-turn history (saves context).
4. **Safety net**: if the thinking-token count blows past a cap (say, 1024), force-insert
   `</THINKING>`.

---

## 7. Evaluation

| Capability | Benchmark | Tool |
|---|---|---|
| Language understanding | HellaSwag, ARC, MMLU (subset) | `lm-evaluation-harness` |
| Coding | **HumanEval, MBPP** | pass@1 scoring script |
| Math / reasoning | GSM8K | Compare thinking-mode on/off → verify that thinking mode is actually helping |
| Basic | validation loss / perplexity | in-house |

---

## 8. Realistic Expectations and Scaling Paths

- **Pretraining nano/small from scratch**: you get fluent generation on par with the original
  GPT-2, plus basic instruction-following. Don't expect real coding ability out of this — it's not
  going to happen at this scale.
- **To actually move the needle on coding performance**: ① raise the code-data ratio (The Stack),
  ② scale the model up to 350M-1B+, ③ add thinking mode plus coding SFT data (e.g.,
  Magicoder-OSS-Instruct).
- **The shortcut (and the recommended track)**: grab a public small base model (Qwen2.5-0.5B/1.5B,
  etc.) and just run the SFT → Thinking → inference pipeline from
  [POST-TRAINING.en.md](POST-TRAINING.en.md) yourself. You pick up the exact same architectural
  knowledge, while ending up with real-world usable quality that blows the original GPT-2 out of
  the water.
- Reference implementations: `karpathy/nanoGPT`, `karpathy/build-nanogpt`, HuggingFace `smol-course`.

<div align="center">

---

[← GLOSSARY.en.md](GLOSSARY.en.md) &nbsp;|&nbsp; [README.en.md](README.en.md) &nbsp;|&nbsp; [POST-TRAINING.en.md →](POST-TRAINING.en.md)

</div>
