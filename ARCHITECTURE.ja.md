<div align="center">

# モデルアーキテクチャ・学習・推論

<img src="https://img.shields.io/badge/CUDA-enabled-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />

[한국어](ARCHITECTURE.md) &nbsp;·&nbsp; [English](ARCHITECTURE.en.md) &nbsp;·&nbsp; **[日本語](ARCHITECTURE.ja.md)**

[← README.md](README.ja.md) &nbsp;|&nbsp; 用語に馴染みがなければ[GLOSSARY.md](GLOSSARY.ja.md)から

</div>

---

GPT-2(初期のGPT)を超えるクラスの小型汎用LLMを、実装から学習・推論まで全部自分の手で組んで、
`<THINKING>`タグベースの思考モードまで載せてみた過程をまとめる。

## 目次

1. [全体ロードマップ](#1-全体ロードマップ)
2. [モデルアーキテクチャ設計](#2-モデルアーキテクチャ設計)
3. [トークナイザー](#3-トークナイザー)
4. [データセット戦略](#4-データセット戦略)
5. [学習パイプライン](#5-学習パイプライン)
6. [推論エンジン](#6-推論エンジン)
7. [評価](#7-評価)
8. [現実的な期待値と拡張の道筋](#8-現実的な期待値と拡張の道筋)

---

## 1. 全体ロードマップ

| 段階 | 内容 | 成果物 |
|---|---|---|
| 0 | 環境構築(PyTorch + CUDA) | 開発環境 |
| 1 | トークナイザー(BPE) | `tokenizer.json` |
| 2 | モデルアーキテクチャ(Decoder-only Transformer) | `model.py` |
| 3 | 事前学習(Pretraining) | baseチェックポイント |
| 4 | 教師ありファインチューニング(SFT、対話形式) | chatチェックポイント |
| 5 | 思考モードSFT(`<THINKING>`データ) | thinkingチェックポイント |
| 6 | 推論エンジン(KVキャッシュ + サンプリング + 思考モードのパース) | `inference.py` |
| 7 | (リリース後)人間のフィードバックによる後続学習 | アライメント済みチェックポイント — [POST-TRAINING.md](POST-TRAINING.ja.md) |

---

## 2. モデルアーキテクチャ設計

### 2.1 基本構造 — Decoder-only Transformer(モダンなレシピ)

GPT-2のオリジナルより一歩進んだ、今どきの構成要素を使っている(LLaMA系のレシピ)。各コンポーネントの
名前にピンとこなかったら、[GLOSSARY.md](GLOSSARY.ja.md#モデル構造の用語)で一つずつ調べてみてほしい。

```
Input Tokens
   │
Token Embedding (weight tyingで出力層と共有)
   │
[ Transformer Block ] × N
   ├─ RMSNorm (Pre-Norm)
   ├─ Self-Attention (Causal, GQA + RoPE + Flash Attention)
   ├─ Residual接続
   ├─ RMSNorm
   ├─ FFN (SwiGLU)
   └─ Residual接続
   │
RMSNorm (最終)
   │
LM Head (Linear → vocab logits)
```

| 構成要素 | GPT-2(2019) | 本設計(推奨) | 理由 |
|---|---|---|---|
| 正規化 | LayerNorm | **RMSNorm** | よりシンプルで学習が安定 |
| 位置情報 | 学習型の絶対位置 | **RoPE** | コンテキスト長の拡張がしやすい |
| 活性化関数 | GELU | **SwiGLU** | 性能向上 |
| Attention | MHA | **GQA**(Grouped-Query) | KVキャッシュのメモリ削減 |
| バイアス | あり | **削除** | パラメータ節約、安定性向上 |

### 2.2 モデルサイズ(GPU予算別3段階)

| プリセット | パラメータ | layers | d_model | heads (Q/KV) | ctx | 必要VRAM(学習) |
|---|---|---|---|---|---|---|
| **nano**(チュートリアル用) | ~30M | 6 | 384 | 6 / 2 | 1024 | ~4GB |
| **small**(GPT-2クラス) | ~124M | 12 | 768 | 12 / 4 | 2048 | ~12GB |
| **base**(GPT-2超え) | ~350M | 24 | 1024 | 16 / 4 | 4096 | ~24GB(またはA100クラウド) |

- vocab: **32,000**(BPE) — 韓国語を含める場合は48k~64k推奨
- FFN hidden: `d_model × 8/3`(SwiGLU基準、例: 768 → 2048)

> このリポジトリの実装([llm/](llm/))は、**nano**プリセットで学習・推論・フィードバックループまで
> 一通り動作確認済みだ。small・baseまでスケールアップする戦略は[README.mdの§2](README.ja.md#2-パラメータ数)に
> そのまま従っている。

### 2.3 コアコードのスケルトン(PyTorch)

<details>
<summary><b>model.py全体のコードを見る</b>(クリックで展開)</summary>

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
        if kv_cache is not None:                      # 推論用KVキャッシュ
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[:] = [k, v]
        k = k.repeat_interleave(self.n_head // self.n_kv, dim=1)   # GQA拡張
        v = v.repeat_interleave(self.n_head // self.n_kv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.d_model * 8 / 3 / 64) * 64   # 64の倍数に整列
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
        # RoPE事前計算(cos/sinバッファ)は省略 — register_bufferを使用

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

- **BPE(バイトレベル)** — HuggingFaceの`tokenizers`ライブラリを使用
- 特殊トークン設計(思考モードの核心):

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

> **実装状況**: 以下の戦略に沿って、実際の韓国語・英語・日本語自然言語 + 11言語のコードデータセットを
> 認証不要な公開ソースからダウンロードし、[llm/datasets/](llm/datasets/)にまとめてある — 合計約11GB、
> 事前学習用テキスト + SFT/思考モード用32万行以上。一覧・出典は
> [llm/datasets/README.md](llm/datasets/README.md)を参照(韓国語のみ)。

### 4.1 事前学習(Pretraining)

| プリセット | データセット | トークン数 | 備考 |
|---|---|---|---|
| nano | **TinyStories** | ~500M | 1日で完走できる、文法的な生成の確認用 |
| small | **FineWeb-Edu(10Bサンプル)** | 5~10B | GPT-2再現クラスの品質 |
| base | FineWeb-Edu + **The Stack(smol)** | 10~30B | **コーディング性能**はコードデータ比率15~25%が鍵 |
| 韓国語 | + AI Hub / 모두의말뭉치 / ko-wikipedia | +α | 韓国語能力が必要な場合 |

> **Chinchilla則**([README.mdの§2](README.ja.md#2-パラメータ数)参照): 最適トークン数 ≈ パラメータ数 × 20
> (124Mモデル → 最低2.5Bトークン)。

### 4.2 SFT(対話形式)

- **OpenHermes-2.5**、**UltraChat**、**KoAlpaca**(韓国語)など公開instructionデータ
- フォーマット:

```
<|system|>You are a helpful assistant.<|eos|>
<|user|>Pythonでクイックソートを実装して<|eos|>
<|assistant|>...コード...<|eos|>
```

### 4.3 思考モードデータ(核心)

- **OpenThoughts / OpenR1-Math / Raiden-DeepSeek-R1**など公開reasoningトレースデータを`<THINKING>`形式に変換
- フォーマット:

```
<|user|>3桁の素数のうち最大のものは?<|eos|>
<|assistant|><THINKING>
3桁の最大数は999。999=3×333で合成数。998は偶数。997を確認:
√997≈31.6、2~31の素数で割ってみると… 割り切れない → 素数。
</THINKING>
最大の3桁の素数は**997**です。<|eos|>
```

- **損失マスキング**: userトークンは`-100`扱い(損失計算から除外)、`<THINKING>`内部を含む
  assistantトークン全体に損失を適用 → モデルが思考過程を自分で生成できるように学習させる

---

## 5. 学習パイプライン

### 5.1 ハイパーパラメータ(small基準)

| 項目 | 値 |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| LRスケジュール | Warmup 2kステップ → Cosine decay |
| Peak LR | 6e-4 (pretrain) / 2e-5 (SFT) |
| Batch | 実効バッチ ~0.5Mトークン (grad accumulation活用) |
| 精度 | bfloat16 (mixed precision) |
| Grad clip | 1.0 |
| その他 | `torch.compile`、Flash Attention (SDPA) |

### 5.2 学習ループのスケルトン

```python
model = GPT(cfg).cuda()
model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4,
                        betas=(0.9, 0.95), weight_decay=0.1)

for step, (x, y) in enumerate(loader):        # x,y: (B, T)、uint16のmemmapを推奨
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x.cuda(), y.cuda())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    lr_scheduler.step()
    if step % 1000 == 0:
        save_checkpoint(model, opt, step)      # + val lossをロギング
```

### 5.3 段階ごとの順序

```
[1] Pretraining  : 生テキスト+コード、next-token prediction     (数日~数週間)
[2] SFT          : 対話形式、userマスキング                       (数時間)
[3] Thinking SFT : <THINKING>データを混合 (通常:思考 = 3:7)      (数時間)
    ※ 2・3をまとめて一度に行ってもよい。思考モードのon/offは
      systemプロンプト("think step by step in <THINKING>")で制御可能
[4] (任意) DPO/GRPO : 正解検証可能な数学・コーディング問題で強化学習 → 思考の質を向上 (POST-TRAINING.md参照)
```

---

## 6. 推論エンジン

### 6.1 生成ループ(KVキャッシュ + 思考モード)

```python
@torch.no_grad()
def generate(model, tok, prompt, thinking=True,
             max_new=2048, temperature=0.7, top_p=0.9):
    sys = "You are a helpful assistant."
    if thinking:
        sys += " Reason step-by-step inside <THINKING> tags before answering."
    ids = tok.encode(f"<|system|>{sys}<|eos|><|user|>{prompt}<|eos|><|assistant|>")
    if thinking:
        ids += tok.encode("<THINKING>")        # 思考モードを強制開始(プリフィル)

    kv_caches = [[] for _ in model.blocks]     # レイヤーごとのKVキャッシュ
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
        x = next_id                             # キャッシュのおかげで新しいトークン1つだけforward

    text = tok.decode(out)
    # 思考区間を分離 → UIで折りたたみ/非表示処理
    thought, _, answer = text.partition("</THINKING>")
    return {"thinking": thought.strip(), "answer": answer.strip()}
```

### 6.2 思考モードの動作原理まとめ

1. **学習**: `<THINKING>...</THINKING>`を含むassistant応答でSFT → モデルが「まず考えて、それから答える」
   パターンを体に染み込ませる
2. **推論**: assistantターン開始直後に`<THINKING>`トークンをプリフィルして思考を強制スタートさせる
   (offモードではプリフィルを省略し、systemプロンプトも変更)
3. **後処理**: `</THINKING>`を境に思考/回答を分離し、マルチターンの履歴には**回答だけ**を保存する
   (コンテキストの節約)
4. **安全装置**: 思考トークンの上限(例: 1024)を超えたら`</THINKING>`を強制挿入

---

## 7. 評価

| 能力 | ベンチマーク | ツール |
|---|---|---|
| 言語理解 | HellaSwag, ARC, MMLU(一部) | `lm-evaluation-harness` |
| コーディング | **HumanEval, MBPP** | pass@1採点スクリプト |
| 数学・推論 | GSM8K | 思考モードon/offの比較 → 思考モードの効果を検証 |
| 基本 | validation loss / perplexity | 自前 |

---

## 8. 現実的な期待値と拡張の道筋

- **nano/small単独の事前学習**: GPT-2オリジナル並みの流暢な生成と、簡単な指示への対応まではいける。
  本格的なコーディング能力は期待しない方がいい。
- **コーディング性能を実際に上げるには**: ①コードデータの比率を増やし(The Stack)、②モデルを
  350M~1B+に大きくし、③思考モード + コーディングSFTデータ(例: Magicoder-OSS-Instruct)を追加する。
- **近道(おすすめのトラック)**: 公開されている小型ベースモデル(Qwen2.5-0.5B/1.5Bなど)を持ってきて、
  [POST-TRAINING.md](POST-TRAINING.ja.md)のSFT→Thinking→推論の過程だけを自分の手でやってみれば、
  同じアーキテクチャの知識を学びながらもGPT-2をはるかに超える実用品質を確保できる。
- 参考実装: `karpathy/nanoGPT`、`karpathy/build-nanogpt`、HuggingFaceの`smol-course`。

<div align="center">

---

[← GLOSSARY.md](GLOSSARY.ja.md) &nbsp;|&nbsp; [README.md](README.ja.md) &nbsp;|&nbsp; [POST-TRAINING.md →](POST-TRAINING.ja.md)

</div>
