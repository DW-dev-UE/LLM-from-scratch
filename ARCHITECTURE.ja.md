[한국어](ARCHITECTURE.md) · [English](ARCHITECTURE.en.md) · [日本語](ARCHITECTURE.ja.md)

[← README](README.ja.md) · 用語で詰まったら [GLOSSARY](GLOSSARY.ja.md)

---

この文書は **モデルがどうできていて、どう学習・推論するか** を書いたものです。
GPT-2 級の小さな汎用 LLM を自分で組み、`<THINKING>` 思考モードまで載せる過程です。

---

## 1. 全体ロードマップ

最初から完璧なパイプラインを組んだわけではありません。だいたいこの順で手を動かしました。

| 段階 | 内容 | 成果物 |
|---|---|---|
| 0 | 環境構築 (PyTorch + CUDA) | 開発環境 |
| 1 | トークナイザー (BPE) | `tokenizer.json` |
| 2 | モデルアーキテクチャ (Decoder-only Transformer) | `model.py` |
| 3 | 事前学習 (Pretraining) | base チェックポイント |
| 4 | 教師あり微調整 (SFT、対話形式) | chat チェックポイント |
| 5 | 思考モード SFT (`<THINKING>` データ) | thinking チェックポイント |
| 6 | 推論エンジン (KV キャッシュ + サンプリング + 思考モードのパース) | `inference.py` |
| 7 | (デプロイ後) 人間フィードバックによる後続学習 | 整列済みチェックポイント — [POST-TRAINING.md](POST-TRAINING.ja.md) |

---

## 2. モデルアーキテクチャ設計

### 2.1 基本構造 — Decoder-only Transformer

最初は GPT-2 そのままでもいいかな、と思いました。
でも最近のレシピ（LLaMA 周り）を見ると、変えるところが結構あります。
用語が馴染めなければ [GLOSSARY](GLOSSARY.ja.md#モデル構造) を見てください。

```
Input Tokens
   │
Token Embedding (weight tying で出力層と共有)
   │
[ Transformer Block ] × N
   ├─ RMSNorm (Pre-Norm)
   ├─ Self-Attention (Causal, GQA + RoPE + Flash Attention)
   ├─ Residual 接続
   ├─ RMSNorm
   ├─ FFN (SwiGLU)
   └─ Residual 接続
   │
RMSNorm (最終)
   │
LM Head (Linear → vocab logits)
```

| 構成要素 | GPT-2 (2019) | 本設計 | 理由 |
|---|---|---|---|
| 正規化 | LayerNorm | **RMSNorm** | よりシンプル、学習が安定 |
| 位置情報 | 学習型の絶対位置 | **RoPE** | 長さ拡張が比較的しやすい |
| 活性化関数 | GELU | **SwiGLU** | 性能向上 |
| Attention | MHA | **GQA** (Grouped-Query) | KV キャッシュのメモリ削減 |
| バイアス | あり | **削除** | パラメータ節約、安定 |

一言でいうと: **学習が崩れにくく、推論メモリを節約でき、長さ拡張がまだしもしやすい** から、こう選びました。

### 2.2 モデルサイズ (GPU 予算別)

| プリセット | パラメータ | layers | d_model | heads (Q/KV) | ctx | 必要 VRAM(学習) |
|---|---|---|---|---|---|---|
| **nano** (チュートリアル) | ~30M | 6 | 384 | 6 / 2 | 1024 | ~4GB |
| **small** (GPT-2 級) | ~124M | 12 | 768 | 12 / 4 | 2048 | ~12GB |
| **base** (GPT-2 超え) | ~350M | 24 | 1024 | 16 / 4 | 4096 | ~24GB (または A100 クラウド) |

- vocab: **32,000** (BPE) — 韓国語を含める場合は 48k~64k 推奨
- FFN hidden: `d_model × 8/3` (SwiGLU 基準、例: 768 → 2048)

実装 ([llm/](llm/)) は **nano** で学習・推論・フィードバックループまで回してみました。
small・base に伸ばす話は [README §2](README.ja.md#2-パラメータはゆっくり大きくする) をそのまま踏襲しています。

### 2.3 コアコードのスケルトン (PyTorch)

<details>
<summary><b>model.py 全体のコードを見る</b> (クリックで展開)</summary>

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
        if kv_cache is not None:                      # 推論用 KV キャッシュ
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[:] = [k, v]
        k = k.repeat_interleave(self.n_head // self.n_kv, dim=1)   # GQA 拡張
        v = v.repeat_interleave(self.n_head // self.n_kv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.d_model * 8 / 3 / 64) * 64   # 64 倍数に整列
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
        # RoPE 事前計算 (cos/sin buffer) は省略 — register_buffer を使う

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        cos, sin = self.rope_cache(idx.shape[1])            # 実装省略
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

## 3. トークナイザー

- **BPE (Byte-level)** — HuggingFace `tokenizers`
- 特殊トークン (思考モードのため重要):

```
<|bos|>  <|eos|>  <|pad|>
<|user|>  <|assistant|>  <|system|>       ← 対話の役割
<THINKING>  </THINKING>                    ← 思考モードの区間
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

## 4. データセット戦略

下の戦略に合わせて、韓国語・英語・日本語の自然言語 + 11 言語のコードデータを、
認証不要な公開ソースから取り、[llm/datasets/](llm/datasets/) に整理してあります。
約 11GB、事前学習テキスト + SFT/THINKING 32 万行ほど。
一覧・出典は [llm/datasets/README.md](llm/datasets/README.md)。

### 4.1 事前学習 (Pretraining)

| プリセット | データセット | トークン数 | 備考 |
|---|---|---|---|
| nano | **TinyStories** | ~500M | 1 日で完走、文法的な生成の確認用 |
| small | **FineWeb-Edu (10B サンプル)** | 5~10B | GPT-2 再現級 |
| base | FineWeb-Edu + **The Stack (smol)** | 10~30B | **コーディング**はコード比率 15~25% が鍵 |
| 韓国語 | + AI Hub / 모두의말뭉치 / ko-wikipedia | +α | 韓国語が必要なとき |

Chinchilla の要点 ([README §2](README.ja.md#2-パラメータはゆっくり大きくする)): 最適トークン数 ≈ パラメータ × 20  
(124M モデルなら最低 2.5B トークン)。

### 4.2 SFT (対話形式)

- **OpenHermes-2.5**、**UltraChat**、**KoAlpaca** などの公開 instruction データ
- 形式:

```
<|system|>You are a helpful assistant.<|eos|>
<|user|>Pythonでクイックソートを実装して<|eos|>
<|assistant|>...コード...<|eos|>
```

### 4.3 思考モードデータ

- **OpenThoughts / OpenR1-Math / Raiden-DeepSeek-R1** などの reasoning トレースを `<THINKING>` 形式に変換
- 形式:

```
<|user|>3桁の素数のうち最大のものは?<|eos|>
<|assistant|><THINKING>
3桁の最大数は999。999=3×333で合成数。998は偶数。997を確認:
√997≈31.6、2~31の素数で割ってみると… 割り切れない → 素数。
</THINKING>
最大の3桁の素数は**997**です。<|eos|>
```

- **損失マスキング**: user トークンは `-100` (損失から除外)。  
  `<THINKING>` を含む assistant 全体に損失をかけ、思考過程も自分で書けるようにします。

---

## 5. 学習パイプライン

### 5.1 ハイパーパラメータ (small 基準)

| 項目 | 値 |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| LR スケジュール | Warmup 2k step → Cosine decay |
| Peak LR | 6e-4 (pretrain) / 2e-5 (SFT) |
| Batch | 実効バッチ ~0.5M トークン (grad accumulation) |
| 精度 | bfloat16 (mixed precision) |
| Grad clip | 1.0 |
| その他 | `torch.compile`、Flash Attention (SDPA) |

### 5.2 学習ループのスケルトン

```python
model = GPT(cfg).cuda()
model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4,
                        betas=(0.9, 0.95), weight_decay=0.1)

for step, (x, y) in enumerate(loader):        # x,y: (B, T) uint16 memmap 推奨
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x.cuda(), y.cuda())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    lr_scheduler.step()
    if step % 1000 == 0:
        save_checkpoint(model, opt, step)      # + val loss をロギング
```

### 5.3 段階ごとの順序

```
[1] Pretraining  : 生テキスト+コード、next-token prediction     (数日~数週間)
[2] SFT          : 対話形式、user マスキング                      (数時間)
[3] Thinking SFT : <THINKING> データ混合 (通常:思考 = 3:7)      (数時間)
    ※ 2・3をまとめて一度にやってもよい。思考モード on/off は
      system プロンプト("think step by step in <THINKING>")で制御可能
[4] (任意) DPO/GRPO : 正解検証できる数学・コーディング問題で強化学習 → 思考品質向上 (POST-TRAINING.md)
```

---

## 6. 推論エンジン

### 6.1 生成ループ (KV キャッシュ + 思考モード)

```python
@torch.no_grad()
def generate(model, tok, prompt, thinking=True,
             max_new=2048, temperature=0.7, top_p=0.9):
    sys = "You are a helpful assistant."
    if thinking:
        sys += " Reason step-by-step inside <THINKING> tags before answering."
    ids = tok.encode(f"<|system|>{sys}<|eos|><|user|>{prompt}<|eos|><|assistant|>")
    if thinking:
        ids += tok.encode("<THINKING>")        # 思考モード強制開始(プリフィル)

    kv_caches = [[] for _ in model.blocks]     # レイヤーごとの KV キャッシュ
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
        x = next_id                             # キャッシュのおかげで新トークン1つだけ forward

    text = tok.decode(out)
    # 思考区間を分離 → UI で折りたたみ/非表示
    thought, _, answer = text.partition("</THINKING>")
    return {"thinking": thought.strip(), "answer": answer.strip()}
```

### 6.2 思考モードが回る仕組み

1. **学習**: `<THINKING>...</THINKING>` がある assistant 応答で SFT → 「まず思考、そのあと答」パターンを覚える  
2. **推論**: assistant 直後に `<THINKING>` をプリフィルして思考を強制 (off ならプリフィル省略 + system 変更)  
3. **後処理**: `</THINKING>` を境に思考/答を分離。マルチターン履歴には **答だけ** 保存 (コンテキスト節約)  
4. **安全装置**: 思考トークン上限 (例: 1024) を超えたら `</THINKING>` を強制挿入  

---

## 7. 評価

| 能力 | ベンチマーク | ツール |
|---|---|---|
| 言語理解 | HellaSwag, ARC, MMLU(一部) | `lm-evaluation-harness` |
| コーディング | **HumanEval, MBPP** | pass@1 採点スクリプト |
| 数学・思考 | GSM8K | 思考モード on/off 比較 |
| 基本 | validation loss / perplexity | 自前 |

---

## 8. 現実的な期待値と拡張

- **nano/small 単独の事前学習**: GPT-2 元の水準の流暢さ + 簡単な指示まではいけます。  
  本格的なコーディング力は期待しない方がいいです。
- **コーディングを本当に上げるには**: ① コードデータ比率を増やし (The Stack)、② 350M~1B+ に大きくし、  
  ③ 思考モード + コーディング SFT (例: Magicoder-OSS-Instruct) を足します。
- **近道**: 公開の小型ベース (Qwen2.5-0.5B/1.5B など) を持ってきて  
  [POST-TRAINING](POST-TRAINING.ja.md) の SFT→Thinking→推論だけを自分でやっても、  
  同じ知識を学びながら GPT-2 よりずっと使える品質が取れます。
- 参考: `karpathy/nanoGPT`、`karpathy/build-nanogpt`、HuggingFace `smol-course`。

---

[GLOSSARY](GLOSSARY.ja.md) · [README](README.ja.md) · [POST-TRAINING](POST-TRAINING.ja.md) · [BENCHMARK](BENCHMARK-v1.ja.md)
