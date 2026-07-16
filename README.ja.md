> **このリポジトリは継続更新中です。**  
> 学習結果・ベンチ・ドキュメントはバージョンごとに変わります。最新コミットを見てください。Issue / PR 歓迎。

> 始める前に、この文章は読みやすさ重視で磨かれていないかもしれません。
> AI に「読みやすく整えて」と頼むこともできますが、そうすると自分が何を考え、どんな過程を経たかが消えてしまうことがあります。
> AI で学ぶのは良いですが、文を何度も書き直し自分の知識を整理する方が大切です。

---

では、始めます。

## このプロジェクトがやること

NVIDIA CUDA 上で動く **汎用 LLM** と、コーディング・文書作業のような実務向け AI を **ゼロから** 作る実験です。

- トークナイザー学習
- モデルアーキテクチャ（Decoder-only Transformer）
- 事前学習 · SFT · DPO · GRPO
- KV キャッシュ推論 · 思考モード（THINKING）
- 人のフィードバック収集 UI · 昇格ゲート

既存の OSS LLM をファインチューニングする方がはるかに速いですが、それでも最初から書き直す理由は一つだけです。

LLM の内部を自分で書かないと、結局は他人のコードを借りる側に留まる。

> LLM の内部を自分で書かないと、結局は他人のコードを借りる側に留まる。

そのためこのリポジトリは **外部の商用/オープン LLM 重みに依存しません。**

---

## ドキュメント案内

| 文書 | 何を見るか |
|:-----|:-----------|
| **README**（本頁） | 設計思想、パラメータ戦略、構成、実行 |
| [GLOSSARY](GLOSSARY.ja.md) | Transformer、RoPE、DPO などの用語 |
| [ARCHITECTURE](ARCHITECTURE.ja.md) | モデル · トークナイザー · 学習 · 推論 |
| [POST-TRAINING](POST-TRAINING.ja.md) | デプロイ後の人間フィードバックループ |
| [BENCHMARK v1](BENCHMARK-v1.ja.md) | Base モデルベンチ（学習過程 · Q&A 含む） |

[한국어](README.md) · [English](README.en.md) · [日本語](README.ja.md)

---

## 1. アーキテクチャ

### トランスフォーマーアーキテクチャ

> LLM がどのように学習されるかについての記述です。

LLM を学習するにはデータセットが必要です。データセットは「コーパス」とも呼ばれます。この例のコーパスは「私は猫が好きです」です。

---

#### 1) トークナイザー

コーパスをコンピュータの言語に変える道具を **トークナイザー** と呼びます。

"私は猫が好きです" → <kbd>私は</kbd> <kbd>猫</kbd> <kbd>が</kbd> <kbd>好き</kbd> <kbd>です</kbd>

こう分割すると **vocab** という辞書ができます。

```text
vocab = {"私は": 1, "猫": 2, "が": 3, "好き": 4, "です": 5}
```

token ids: `[1, 2, 3, 4, 5]`

---

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

---

#### 3) 問題を作る

トークン列を一マスずつずらして入力/正解ペアを作ります。

入力: `[1, 2, 3, 4]` = "私は 猫 が 好き"<br>
正解: `[2, 3, 4, 5]` = "猫 が 好き です"

<kbd>私は</kbd> → ?（正解: 猫）<br>
<kbd>私は</kbd> <kbd>猫</kbd> → ?（正解: が）

---

#### 4) 正規化

x = 埋め込み値。問題 [私は, 猫, が, 好き] とすると、x は「好き」の埋め込みです。

x = <kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd>

正規化後の N:

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>

---

#### 5) アテンション

文脈を混ぜる段階 — ここからがトランスフォーマーアーキテクチャの核心です。<br>
従来の bigram アーキテクチャは文脈を見られません。同じ「好き」でも前が「猫」なのか別の語なのかを区別できません。

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

---

#### 6) 残差接続

x とアテンション出力を足して new v を作ります。元の意味を保ちつつ文脈情報を載せるやり方です。

<kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd> + <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd><br>
= <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd>

まだ数字に過ぎませんが、文脈を捉え学習する準備ができました。

---

### ブロック（Block）

ここからがブロックアーキテクチャです。

#### ① 正規化

new v を正規化します。

N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd>

---

#### ② FFN 1段階 — W₁ で拡張

N₂ を特定の倍率で広げます。GPT-2 は 4 倍で拡張するので、この例も同じにします。<br>
アテンション出力は V たちの加重和、つまり混ぜて平均する役割に過ぎませんが、広げられた tensor は「特定パターン検知器」のように働きます。

拡張: `U = N·W₁ + b₁`（4×4 行列 · 4×16 行列 = 4×16、b₁ はバイアス 16 マス、初期値 0 なので省略）

> **凡例**: <span style="color:#1d9e75">■</span> 通過予定（正） · <span style="color:#d85a30">■</span> 遮断予定（負） · <span style="color:#888780">■</span> 0 に近い

**表 A — GELU 適用前（U）**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 私は | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-1.53</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#1d9e75">1.96</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#d85a30">-2.58</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.29</span> | <span style="color:#d85a30">-1.37</span> | <span style="color:#d85a30">-0.28</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-0.26</span> | <span style="color:#d85a30">-0.65</span> | <span style="color:#1d9e75">2.15</span> | <span style="color:#1d9e75">1.48</span> |
| 猫 | <span style="color:#d85a30">-1.96</span> | <span style="color:#d85a30">-0.90</span> | <span style="color:#1d9e75">1.93</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-1.33</span> | <span style="color:#1d9e75">0.89</span> | <span style="color:#d85a30">-2.79</span> | <span style="color:#1d9e75">0.39</span> | <span style="color:#1d9e75">2.57</span> | <span style="color:#d85a30">-1.80</span> | <span style="color:#d85a30">-0.58</span> | <span style="color:#1d9e75">0.77</span> | <span style="color:#d85a30">-1.60</span> | <span style="color:#1d9e75">1.57</span> | <span style="color:#d85a30">-1.06</span> | <span style="color:#d85a30">-2.67</span> |
| が | <span style="color:#1d9e75">2.35</span> | <span style="color:#1d9e75">0.64</span> | <span style="color:#d85a30">-2.43</span> | <span style="color:#d85a30">-2.82</span> | <span style="color:#1d9e75">2.28</span> | <span style="color:#d85a30">-1.91</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">0.50</span> | <span style="color:#d85a30">-2.27</span> | <span style="color:#1d9e75">0.94</span> | <span style="color:#1d9e75">0.31</span> | <span style="color:#d85a30">-0.76</span> | <span style="color:#1d9e75">0.90</span> | <span style="color:#d85a30">-1.31</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.10</span> |
| 好き | <span style="color:#d85a30">-0.88</span> | <span style="color:#d85a30">-2.45</span> | <span style="color:#1d9e75">1.63</span> | <span style="color:#1d9e75">3.40</span> | <span style="color:#d85a30">-1.20</span> | <span style="color:#1d9e75">2.67</span> | <span style="color:#d85a30">-3.24</span> | <span style="color:#1d9e75">2.07</span> | <span style="color:#1d9e75">0.79</span> | <span style="color:#d85a30">-0.62</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.21</span> | <span style="color:#1d9e75">0.82</span> | <span style="color:#d85a30">-0.63</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-1.77</span> |

---

#### ③ FFN 2段階 — GELU

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

---

#### ④ 縮小

W₂ 行列積（16×4）。「16 個の信号を加重和して再び 4 マスに要約」する段階です。<br>
結果: <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd>

---

#### ⑤ FFN 出力

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

---

ここまでが順伝播で、loss は `−ln(です の確率) = −ln(0.0000000126) ≈ 18.2` です。かなり高い数字で、ここから逆伝播で重みを改善していきます。

### 最初に考えていた道

最初の計画は単純でした。

| 段 | 内容 |
|:--:|:-----|
| **1** | 自然言語データだけで先に LLM を作る |
| **2** | その上にコーディング特化データを載せる |
| **3** | 人が強化学習で整える |

コーディング AI でも質問は自然言語で来るので、  
**「言葉を理解する力」を先に完成させる** つもりでした。

### 論文が示したこと

[Llama 3](https://ar5iv.labs.arxiv.org/html/2407.21783) を見ると、その順序はありません。  
事前学習の段階から既に一つのミックスに混ぜます。

| 比重（おおよそ） | データ |
|:----------------:|:-------|
| 50% | 自然言語 |
| 25% | 数学 · 推論 |
| 17% | コード |
| 8% | 多言語 |

後続学習で instruction、preference、coding、tool-use を強化するだけです。  
**「自然言語を先に終えてからコードを載せる」段階はありません。**

[DeepSeek-Coder](https://arxiv.org/html/2401.14196v1) も同じ方向です。  
コード 87% + コード関連英語 10% + その他 3% で最初から学習し、  
自然言語が 13% しかなくても当時のオープンコード LLM で上位でした。

### だから変えたパイプライン

| | 順序 |
|:--|:-----|
| 以前考えていたこと | 自然言語のみ → コードのみ → instruction |
| **いま使っていること** | **混合 pretrain** → コード比重強化 → instruction → preference |

```text
  自然言語 + コード + 数学 + 文書
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

### データミックスの目安（300M ~ 1B）

| 種類 | 比率 | 例 |
|:-----|:----:|:---|
| Raw source code | 35–45% | ソースファイル |
| Code-adjacent | 15–20% | README、issue、PR、commit |
| 技術自然言語 | 20–30% | 英語 · 韓国語の技術文書 |
| Math / reasoning | 5–10% | 数学 · 論理 |
| Logs / tests / diffs | 5–10% | ログ、テスト、diff |
| 韓国語 · bilingual | 5–10% | 指示 · 二言語 |

---

## 2. パラメータはゆっくり大きくする

パラメータはモデルの大きさです。（用語は [GLOSSARY](GLOSSARY.ja.md)）  
大きいほど余地は広がりますが、このプロジェクトは **300M → 1B → 3B** のように段階を分けます。

[Chinchilla](https://arxiv.org/abs/2203.15556) の要点:

> 同じ compute 予算なら **モデルサイズとトークン数を一緒に** 増やす。

当時の大型モデルはサイズに対してデータが足りないことが多く、  
より小さな Chinchilla（70B、1.4T トークン）がより大きな Gopher より良いスコアを出すこともありました。

なので原則はこうです。

1. 小さなモデルで **tokenizer · dataloader · loss · eval · checkpoint** が回ることを確認する  
2. その後スケールアップする  

[StarCoder2](https://arxiv.org/abs/2402.19173) も 3B / 7B / 15B をそれぞれ別トークン予算で学習した例です。

### このプロジェクトが踏む段階

| 規模 | 目的 |
|:-----|:-----|
| 10M ~ 50M | 学習ループ、tokenizer、loss 減少の確認 |
| 100M ~ 300M | completion、FIM、基本 instruction |
| 1B | 小さな coding assistant 実験 |
| 3B | 社内ツール連携の候補 |
| 7B+ | 外部公開を考える最小規模 |

いまベンチに載せた最新チェックポイントは **`sft_base_v6`**（base 約 327M）です。→ [§5 ベンチマーク一覧](#5-ベンチマーク一覧) · バージョン別詳細記録 [BENCHMARK v1](BENCHMARK-v1.ja.md)

---

## 3. モデル vs RAG · Tool

会社の知識を全部重みに入れようとすると耐えられません。  
コード、文書、ビルドログは変わり続けるからです。

| 担当 | 担うこと |
|:-----|:---------|
| **モデル（重み）** | 考え方、コーディングパターン、文脈理解、回答のトーン、ツールの **使い方** |
| **RAG / Tools** | 最新の事実、社内文書、repo、テスト実行、ビルド結果 |

> モデル = **どう考え、どう道具を使うか**  
> RAG / Tool = **いま何が事実か**

（RAG · Tool ランタイム連携はロードマップ上。いまの公開範囲の中心は学習 · 推論 · 整列パイプラインです。）

---

## 4. プロジェクト構成

```text
AI/
├── README.md                 設計思想 · 実行案内
├── GLOSSARY.md               用語
├── ARCHITECTURE.md           モデル · 学習 · 推論
├── POST-TRAINING.md          フィードバック後続学習
├── BENCHMARK-v1.md           ベンチレポート
└── llm/                      実装
    ├── model.py              RMSNorm + RoPE + SwiGLU + GQA
    ├── tokenizer.py
    ├── data.py
    ├── train.py              pretrain / sft / dpo
    ├── infer.py              チャット · ログ
    ├── feedback.py           フィードバック Web UI
    ├── reward.py · rlhf.py   RM + GRPO / RLVR
    └── eval_gate.py          昇格ゲート
```

> 大容量の重み（`.pt`）、生コーパス、一部ソースはリポジトリに載せません。  
> 文書で設計とベンチを先に公開する形です。

---

## 5. ベンチマーク一覧

**学習 GPU**: H100, A100

`base`（~327M）基準、同一 14 問 × THINKING on/off。  
最新スナップショット: **`sft_base_v6`**（`ckpt/benchmark_sft_base_v6.json`、2026-07-15）。

| チェックポイント | 要約 |
|:-----------------|:-----|
| pretrain v1 | チャット形式で事実上 0 点（指示未学習） |
| sft v1 | 指示は試す。正答率は低い。THINKING 回答 0/14（終了タグ未閉じ） |
| sft v2 | THINKING 終了はほぼ復旧（非空回答 13/14）。コーディングは依然弱い |
| sft v3 | コーディング（通常チャット）4/5 完全通過。英語は強い、韓国語は全問 0 点。THINKING コーディングは依然 0/5 |
| sft v4 | 韓国語 SFT 比率↑（10.3%→22.5%）。ベンチはほぼ横ばい、韓国語回復はわずか |
| sft v5 | ベースを pretrain_v2 に交換 + greedy デコーディング。コーディング初満点（5/5）、韓国語が回復し始める（8/30） |
| **sft v6** | **THINKING ハンドオフを解決。** prep_sft 前処理 + 推論予算分離だけで思考平均 0.50→2.93、思考コーディング初通過（4/5） |

**sft_base_v6 · 通常チャットモード（THINKING オフ）** · 平均スコア 3.93 / 5

| 領域 | 結果 |
|:-----|:-----|
| 韓国語 | 平均 **2.67 / 5**（fact 問題で初満点） |
| 日本語 | 平均 3.00 / 5 |
| 英語 | 平均 **4.33 / 5** |
| コーディング | テスト **25 / 25**（完全通過 **5 / 5**） |
| THINKING オン | 平均 **2.93 / 5** · 非空回答 **14/14** · コーディング完全通過 **4/5**（+prime 部分 3/5） |

v6 ハイライト（v5 比）:

- **THINKING ハンドオフ問題を解決**: 空回答 4/14 → 0/14（全問で回答生成）、THINKING 平均 0.50 → 2.93（約 6 倍）
- THINKING コーディング **0/5 → 4/5** — プロジェクト 6 バージョン目にして初の実コード出力（`+prime` 部分通過 3/5）
- 通常チャット平均も連動して上昇: **3.21 → 3.93**、コーディングは 5/5 満点を維持
- 韓国語 fact 初満点 · 日本語数学が両モードとも初正解（韓国語合計 8/30 → 10/30）
- ベース·SFT ソースミックスは v5 と完全に同一 — 変化はデータ前処理（`prep_sft`）と推論予算分離のみ
- 残る課題: 韓国語算術は依然崩壊、長い生成での反復/退化、日→英翻訳が未形成

学習過程·データセット·設問ごとの Q&A（バージョン別詳細記録）:

- [BENCHMARK-v1.md](BENCHMARK-v1.md)（韓国語）  
- [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md)  
- [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md)  

まだ early-stage です。バージョンを上げるたび同じ設問で比較する予定です。

---

## 6. 参考文献

| 論文 | リンク |
|:-----|:-------|
| Llama 3 Herd of Models | https://ar5iv.labs.arxiv.org/html/2407.21783 |
| DeepSeek-Coder | https://arxiv.org/html/2401.14196v1 |
| Chinchilla (compute-optimal) | https://arxiv.org/abs/2203.15556 |
| StarCoder 2 | https://arxiv.org/abs/2402.19173 |

---

<div align="center">

**次に読む**

[用語](GLOSSARY.ja.md) · [アーキテクチャ](ARCHITECTURE.ja.md) · [後続学習](POST-TRAINING.ja.md) · [ベンチマーク](BENCHMARK-v1.ja.md)

</div>
