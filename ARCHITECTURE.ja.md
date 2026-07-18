# ゼロから作る LLM

[![KO](https://img.shields.io/badge/KO-lightgrey)](ARCHITECTURE.md) [![EN](https://img.shields.io/badge/EN-lightgrey)](ARCHITECTURE.en.md) [![JA](https://img.shields.io/badge/JA-0969da)](ARCHITECTURE.ja.md)

[← README](README.ja.md) · 用語で詰まったら [GLOSSARY](GLOSSARY.ja.md)

---

この文書は **モデルがどうできていて、どう学習・推論するか** を書いたものです。
§2 でトランスフォーマー順伝播を数値例でたどり、§3 から本リポジトリの本番設計（トークナイザー · 学習 · 推論 · 思考モード）に進みます。

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

## 2. トランスフォーマーの動作原理（教育用ウォークスルー）

教育用の数値例です（GPT-2 スタイルの GELU など）。本リポジトリの本番設計は次節（§3）の LLaMA スタイル（RMSNorm / RoPE / SwiGLU / GQA）です。

### トランスフォーマーアーキテクチャ

> LLM がどのように学習されるかについての記述です。

LLM を学習するにはデータセットが必要です。データセットは「コーパス」とも呼ばれます。この例のコーパスは「私は猫が好きです」です。


#### 1) トークナイザー

コーパスをコンピュータの言語に変える道具を **トークナイザー** と呼びます。

"私は猫が好きです" → <kbd>私は</kbd> <kbd>猫</kbd> <kbd>が</kbd> <kbd>好き</kbd> <kbd>です</kbd>

こう分割すると **vocab** という辞書ができます。

```text
vocab = {"私は": 1, "猫": 2, "が": 3, "好き": 4, "です": 5}
```

token ids: `[1, 2, 3, 4, 5]`


#### 2) 埋め込みテーブル

各トークンごとにランダムな実数でマスを埋めます。

| # | token | d1 | d2 | d3 | d4 |
|--:|-------|---:|---:|---:|---:|
| 1 | 私は | 0.12 | -0.53 | 0.33 | 0.90 |
| 2 | 猫 | -0.51 | 0.30 | -2.10 | 0.87 |
| 3 | が | 0.05 | -0.44 | 1.32 | -0.06 |
| 4 | 好き | 0.71 | 0.18 | -0.29 | 0.55 |
| 5 | です | -0.33 | 0.92 | 0.14 | -0.78 |

マスの幅を **埋め込み次元**（d_model）と呼び、次元が多いほどトークンを説明する余地が増え、表現力も上がります。

実数一つを何バイトで持つかは **精度**（FP32、FP16 など）の問題です。


#### 3) 問題を作る

トークン列を一マスずつずらして入力/正解ペアを作ります。

入力: `[1, 2, 3, 4]` = "私は 猫 が 好き"<br>
正解: `[2, 3, 4, 5]` = "猫 が 好き です"

<kbd>私は</kbd> → ?（正解: 猫）<br>
<kbd>私は</kbd> <kbd>猫</kbd> → ?（正解: が）


#### 4) 正規化

x = 埋め込み値。問題 [私は, 猫, が, 好き] とすると、x は「好き」の埋め込みです。

x = <kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd>

正規化後の N:

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>


#### 5) アテンション

文脈を混ぜる段階 — ここからがトランスフォーマーアーキテクチャの核心です。<br>
従来の bigram アーキテクチャは文脈を見られません。同じ「好き」でも前が「猫」なのか別の語なのかを区別できません。

> [!TIP]
> 📌 **ランダム行列 3 つが N を三つの見方へ変換**

問題 4 — トークン `[1, 2, 3, 4]` = "私は 猫 が 好き"、正解 `[2, 3, 4, 5]` = "猫 が 好き です"。<br>
予測を行う位置は「好き」です。

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>

Query（質問）`Q = N · W_Q`。W_Q、W_K、W_V は学習開始時にランダムで、学習が進むにつれて調整されます。

K(私は) · Q ÷ √4 = 0.1<br>
K(猫) · Q ÷ √4 = 2.0<br>
K(が) · Q ÷ √4 = −0.6<br>
K(好き) · Q ÷ √4 = 0.5

softmax(<kbd>0.1</kbd> <kbd>2.0</kbd> <kbd>-0.6</kbd> <kbd>0.5</kbd>) = <kbd>0.10</kbd> <kbd>0.69</kbd> <kbd>0.05</kbd> <kbd>0.15</kbd>

「好き」が「猫」を 69% 参照しました — 前の文脈によって同じ語でも別のベクトルになる理由です。

V の加重和: `0.10·V(私は) + 0.69·V(猫) + 0.05·V(が) + 0.15·V(好き)`

アテンション出力 = <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd> — 文脈が混ざった新しいベクトルです。


#### 6) 残差接続

x とアテンション出力を足して new v を作ります。元の意味を保ちつつ文脈情報を載せるやり方です。

<kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd> + <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd><br>
= <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd>

まだ数字に過ぎませんが、文脈を捉え学習する準備ができました。

---

### ブロック（Block）

ここからがブロックアーキテクチャです。

#### 1) 正規化

new v を正規化します。

N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd>


#### 2) FFN 1段階 — W₁ で拡張

N₂ を特定の倍率で広げます。GPT-2 は 4 倍で拡張するので、この例も同じにします。<br>
アテンション出力は V たちの加重和、つまり混ぜて平均する役割に過ぎませんが、広げられた tensor は「特定パターン検知器」のように働きます。

拡張: `U = N·W₁ + b₁`（4×4 行列 · 4×16 行列 = 4×16、b₁ はバイアス 16 マス、初期値 0 なので省略）

> [!NOTE]
> **凡例**: <span style="color:#1d9e75">■</span> 通過予定（正） · <span style="color:#d85a30">■</span> 遮断予定（負） · <span style="color:#888780">■</span> 0 に近い

**表 A — GELU 適用前（U）**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 私は | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-1.53</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#1d9e75">1.96</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#d85a30">-2.58</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.29</span> | <span style="color:#d85a30">-1.37</span> | <span style="color:#d85a30">-0.28</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-0.26</span> | <span style="color:#d85a30">-0.65</span> | <span style="color:#1d9e75">2.15</span> | <span style="color:#1d9e75">1.48</span> |
| 猫 | <span style="color:#d85a30">-1.96</span> | <span style="color:#d85a30">-0.90</span> | <span style="color:#1d9e75">1.93</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-1.33</span> | <span style="color:#1d9e75">0.89</span> | <span style="color:#d85a30">-2.79</span> | <span style="color:#1d9e75">0.39</span> | <span style="color:#1d9e75">2.57</span> | <span style="color:#d85a30">-1.80</span> | <span style="color:#d85a30">-0.58</span> | <span style="color:#1d9e75">0.77</span> | <span style="color:#d85a30">-1.60</span> | <span style="color:#1d9e75">1.57</span> | <span style="color:#d85a30">-1.06</span> | <span style="color:#d85a30">-2.67</span> |
| が | <span style="color:#1d9e75">2.35</span> | <span style="color:#1d9e75">0.64</span> | <span style="color:#d85a30">-2.43</span> | <span style="color:#d85a30">-2.82</span> | <span style="color:#1d9e75">2.28</span> | <span style="color:#d85a30">-1.91</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">0.50</span> | <span style="color:#d85a30">-2.27</span> | <span style="color:#1d9e75">0.94</span> | <span style="color:#1d9e75">0.31</span> | <span style="color:#d85a30">-0.76</span> | <span style="color:#1d9e75">0.90</span> | <span style="color:#d85a30">-1.31</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.10</span> |
| 好き | <span style="color:#d85a30">-0.88</span> | <span style="color:#d85a30">-2.45</span> | <span style="color:#1d9e75">1.63</span> | <span style="color:#1d9e75">3.40</span> | <span style="color:#d85a30">-1.20</span> | <span style="color:#1d9e75">2.67</span> | <span style="color:#d85a30">-3.24</span> | <span style="color:#1d9e75">2.07</span> | <span style="color:#1d9e75">0.79</span> | <span style="color:#d85a30">-0.62</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.21</span> | <span style="color:#1d9e75">0.82</span> | <span style="color:#d85a30">-0.63</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-1.77</span> |


#### 3) FFN 2段階 — GELU

GELU がないと `W₂(W₁·N) = (W₂W₁)·N`、つまり行列一つを掛けたのと数学的に同じになり、16 マス拡張が無駄になります。<br>
非線形が入って初めて「この特徴はオン、あの特徴はオフ」という条件付き動作が生まれます。

`GELU(x) = x × Φ(x)` — 入力を Φ(x) の比率だけ通します。<br>
Φ(x) は標準正規分布で x 以下が出る確率（0〜1）なので、x が大きいほど通過率は 1 に近づき、小さいほど 0 に近づきます。

x が -2.6 なら通過率 0.5% = <span style="color:#d85a30">-0.01</span>（事実上遮断）<br>
x が 3.10 なら通過率 99.9% = <span style="color:#1d9e75">3.10</span>（事実上そのまま）

ただし Φ(x) の確率は「次のトークンが来る確率」とは無関係です。反応値 x が大きいほど多く通す固定の数学的比率に過ぎず、次トークン確率は softmax で初めて出ます。

**表 B — GELU 適用後**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 私は | <span style="color:#1d9e75">1.37</span> | <span style="color:#d85a30">-0.10</span> | <span style="color:#d85a30">-0.13</span> | <span style="color:#1d9e75">1.91</span> | <span style="color:#1d9e75">2.24</span> | <span style="color:#d85a30">-0.13</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.11</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#d85a30">-0.11</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#d85a30">-0.10</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">2.12</span> | <span style="color:#1d9e75">1.37</span> |
| 猫 | <span style="color:#d85a30">-0.05</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#1d9e75">0.72</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">0.25</span> | <span style="color:#1d9e75">2.55</span> | <span style="color:#d85a30">-0.06</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#1d9e75">0.60</span> | <span style="color:#d85a30">-0.09</span> | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-0.15</span> | <span style="color:#d85a30">-0.01</span> |
| が | <span style="color:#1d9e75">2.32</span> | <span style="color:#1d9e75">0.47</span> | <span style="color:#d85a30">-0.02</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-0.05</span> | <span style="color:#1d9e75">1.77</span> | <span style="color:#1d9e75">0.34</span> | <span style="color:#d85a30">-0.03</span> | <span style="color:#1d9e75">0.78</span> | <span style="color:#1d9e75">0.19</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">0.73</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">3.10</span> |
| 好き | <span style="color:#d85a30">-0.17</span> | <span style="color:#d85a30">-0.02</span> | <span style="color:#1d9e75">1.55</span> | <span style="color:#1d9e75">3.39</span> | <span style="color:#d85a30">-0.14</span> | <span style="color:#1d9e75">2.66</span> | <span style="color:#888780">0.00</span> | <span style="color:#1d9e75">2.03</span> | <span style="color:#1d9e75">0.62</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.09</span> | <span style="color:#1d9e75">0.65</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#d85a30">-0.07</span> |

表 A の「好き」 d7 = <span style="color:#d85a30">-3.24</span> → 表 B で <span style="color:#888780">0.00</span> にほぼ消滅 — GELU が強い負を「遮断」するのをそのまま見られます。<br>
逆に d4 = <span style="color:#1d9e75">3.40</span> → <span style="color:#1d9e75">3.39</span> でほぼそのまま「通過」しました。

📌 予測対象の「好き」位置だけを引き続き追跡します。<br>
N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd> → GELU 通過結果（表 B の「好き」行）:<br>
`-0.17 -0.02 1.55 3.39 -0.14 2.66 0.00 2.03 0.62 -0.17 0.00 -0.09 0.65 -0.17 -0.16 -0.07`


#### 4) 縮小

W₂ 行列積（16×4）。「16 個の信号を加重和して再び 4 マスに要約」する段階です。<br>
結果: <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd>


#### 5) FFN 出力

この段階までがブロックの最後です。<br>
new v <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd> + FFN出力 <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd><br>
= <kbd>7.86</kbd> <kbd>0.78</kbd> <kbd>-3.57</kbd> <kbd>6.56</kbd>

こうしたブロックを複数適用したものが多層トランスフォーマーアーキテクチャで、GPT-2 は 12 層を使います。<br>
`x → ブロック1 → h₁ → ブロック2 → h₂ → … → ブロックN → h_N →（出口行列積 → logit → softmax → loss）` の順で動きます。

これが文脈を把握できる LLM を作る原理であり、次にモデルは次の語を当て、その値を確率に変える作業をします。

---

### 📌 logit: 出口行列積

logit = 各語が「次のトークン」である生スコア（raw score）。h と各語埋め込みの内積です。

| token | logit | 備考 |
|---|---:|---|
| 好き | 10.36 | ← 最大（モデルの 1 位推測） |
| 猫 | 9.43 | |
| 私は | 5.26 | |
| が | -5.06 | |
| です | -7.49 | ← 最小（だがこれが正解） |

高いほどモデルはその語だと信じますが、肝心の「です」が正解でした。（重みがランダムなので当然の結果）<br>
そのために softmax → 確率を求めてから loss を計算できます。

softmax = logit（任意の範囲）→ すべて正 + 和=1.00 の確率へ変換

| token | logit | gap(=logit-10.36) | e^gap | 確率 |
|---|---:|---:|---:|---:|
| 好き | 10.36 | 0.00 | 1.0000 | 71.4% |
| 猫 | 9.43 | -0.93 | 0.3946 | 28.2% |
| 私は | 5.26 | -5.10 | 0.0061 | 0.4% |
| が | -5.06 | -15.42 | 0.0000002 | ≈0% |
| です | -7.49 | -17.85 | 0.0000000177 | ≈0% |

和 = 1.4007

ここまでが順伝播で、loss は `−ln(です の確率) = −ln(0.0000000126) ≈ 18.2` です。かなり高い数字で、ここから逆伝播で重みを改善していきます。

実際の実装スタックは下記 [§3 モデルアーキテクチャ設計](#3-モデルアーキテクチャ設計) を参照してください。

---

## 3. モデルアーキテクチャ設計

### 3.1 基本構造 — Decoder-only Transformer

最初は GPT-2 そのままでもいいかな、と思いました。
でも最近のレシピ（LLaMA 周り）を見ると、変えるところが結構あります。
用語が馴染めなければ [GLOSSARY](GLOSSARY.ja.md#モデル構造) を見てください。

```text
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

### 3.2 モデルサイズ (GPU 予算別)

| プリセット | パラメータ | layers | d_model | heads (Q/KV) | ctx | 必要 VRAM(学習) |
|---|---|---|---|---|---|---|
| **nano** (チュートリアル) | ~30M | 6 | 384 | 6 / 2 | 1024 | ~4GB |
| **small** (GPT-2 級) | ~124M | 12 | 768 | 12 / 4 | 2048 | ~12GB |
| **base** (GPT-2 超え) | ~350M | 24 | 1024 | 16 / 4 | 4096 | ~24GB (または A100 クラウド) |

- vocab: **32,000** (BPE) — 韓国語を含める場合は 48k~64k 推奨
- FFN hidden: `d_model × 8/3` (SwiGLU 基準、例: 768 → 2048)

実装 ([llm/](llm/)) は **nano** で学習・推論・フィードバックループまで回してみました。
small・base に伸ばす話は [README §2](README.ja.md#2-パラメータはゆっくり大きくする) をそのまま踏襲しています。

### 3.3 コアコードのスケルトン (PyTorch)

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

## 4. トークナイザー

- **BPE (Byte-level)** — HuggingFace `tokenizers`
- 特殊トークン (思考モードのため重要):

```text
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

## 5. データセット戦略

下の戦略に合わせて、韓国語・英語・日本語の自然言語 + 11 言語のコードデータを、
認証不要な公開ソースから取り、[llm/datasets/](llm/datasets/) に整理してあります。
約 11GB、事前学習テキスト + SFT/THINKING 32 万行ほど。
一覧・出典は [llm/datasets/README.md](llm/datasets/README.md)。

### 5.1 事前学習 (Pretraining)

| プリセット | データセット | トークン数 | 備考 |
|---|---|---|---|
| nano | **TinyStories** | ~500M | 1 日で完走、文法的な生成の確認用 |
| small | **FineWeb-Edu (10B サンプル)** | 5~10B | GPT-2 再現級 |
| base | FineWeb-Edu + **The Stack (smol)** | 10~30B | **コーディング**はコード比率 15~25% が鍵 |
| 韓国語 | + AI Hub / 모두의말뭉치 / ko-wikipedia | +α | 韓国語が必要なとき |

Chinchilla の要点 ([README §2](README.ja.md#2-パラメータはゆっくり大きくする)): 最適トークン数 ≈ パラメータ × 20  
(124M モデルなら最低 2.5B トークン)。

### 5.2 SFT (対話形式)

- **OpenHermes-2.5**、**UltraChat**、**KoAlpaca** などの公開 instruction データ
- 形式:

```text
<|system|>You are a helpful assistant.<|eos|>
<|user|>Pythonでクイックソートを実装して<|eos|>
<|assistant|>...コード...<|eos|>
```

### 5.3 思考モードデータ

- **OpenThoughts / OpenR1-Math / Raiden-DeepSeek-R1** などの reasoning トレースを `<THINKING>` 形式に変換
- 形式:

```text
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

## 6. 学習パイプライン

### 6.1 ハイパーパラメータ (small 基準)

| 項目 | 値 |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| LR スケジュール | Warmup 2k step → Cosine decay |
| Peak LR | 6e-4 (pretrain) / 2e-5 (SFT) |
| Batch | 実効バッチ ~0.5M トークン (grad accumulation) |
| 精度 | bfloat16 (mixed precision) |
| Grad clip | 1.0 |
| その他 | `torch.compile`、Flash Attention (SDPA) |

### 6.2 学習ループのスケルトン

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

### 6.3 段階ごとの順序

```text
[1] Pretraining  : 生テキスト+コード、next-token prediction     (数日~数週間)
[2] SFT          : 対話形式、user マスキング                      (数時間)
[3] Thinking SFT : <THINKING> データ混合 (通常:思考 = 3:7)      (数時間)
    ※ 2・3をまとめて一度にやってもよい。思考モード on/off は
      system プロンプト("think step by step in <THINKING>")で制御可能
[4] (任意) DPO/GRPO : 正解検証できる数学・コーディング問題で強化学習 → 思考品質向上 (POST-TRAINING.md)
```

---

## 7. 推論エンジン

### 7.1 生成ループ (KV キャッシュ + 思考モード)

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

### 7.2 思考モードが回る仕組み

1. **学習**: `<THINKING>...</THINKING>` がある assistant 応答で SFT → 「まず思考、そのあと答」パターンを覚える  
2. **推論**: assistant 直後に `<THINKING>` をプリフィルして思考を強制 (off ならプリフィル省略 + system 変更)  
3. **後処理**: `</THINKING>` を境に思考/答を分離。マルチターン履歴には **答だけ** 保存 (コンテキスト節約)  
4. **安全装置**: 思考トークン上限 (例: 1024) を超えたら `</THINKING>` を強制挿入  

---

## 8. 評価

| 能力 | ベンチマーク | ツール |
|---|---|---|
| 言語理解 | HellaSwag, ARC, MMLU(一部) | `lm-evaluation-harness` |
| コーディング | **HumanEval, MBPP** | pass@1 採点スクリプト |
| 数学・思考 | GSM8K | 思考モード on/off 比較 |
| 基本 | validation loss / perplexity | 自前 |

---

## 9. 現実的な期待値と拡張

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
