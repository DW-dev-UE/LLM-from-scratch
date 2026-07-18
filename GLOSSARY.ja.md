# GLOSSARY

[![KO](https://img.shields.io/badge/KO-lightgrey)](GLOSSARY.md) [![EN](https://img.shields.io/badge/EN-lightgrey)](GLOSSARY.en.md) [![JA](https://img.shields.io/badge/JA-0969da)](GLOSSARY.ja.md)

[← README](README.ja.md)

---

> [!TIP]
> この文書は辞書です。  
> [ARCHITECTURE](ARCHITECTURE.ja.md) や [POST-TRAINING](POST-TRAINING.ja.md) を読んでいて詰まったら、ここに戻ってきてください。  
> 最初から全部覚える必要はありません。知らない単語だけ探せば十分です。

---

## モデル構造

| 用語 | 説明 |
|---|---|
| **Transformer (トランスフォーマー)** | Self-Attention を中核演算に使うニューラルネット構造。2017 年以降に出たほぼすべての LLM (GPT、Llama、DeepSeek など) がこの上で動いている。 |
| **Decoder-only** | 「これまで出たトークンを見て次のトークンを予測する」デコーダブロックだけを積み重ねた構造。GPT 系列。このプロジェクトもこの方式。 |
| **Parameter (パラメータ)** | 学習で値が変わる数字の個数。「300M モデル」ならパラメータが 3 億個。よくモデルの「大きさ」と呼ばれる。 |
| **Weight (重み)** | パラメータが実際に持っている値。例: `nn.Linear(768, 768)` は 768×768 個の重み。学習はこの数字を少しずつ動かす過程。 |
| **Weight tying** | 入力埋め込みと出力層 (LM Head) が同じ重み行列を共有。パラメータを減らし、学習を安定させる。 |
| **トークナイザー (Tokenizer)** | テキストを整数トークン ID に変える道具。「Pythonは」→ 4123 のような感じ。結果はベクトルではなく整数で、ベクトルにするのは次の Embedding の仕事。 |
| **Embedding (埋め込み)** | トークン一つを固定長の数値ベクトルに変える層。文字そのものではなく、ベクトル演算で言語を扱えるようにする。 |
| **Vision Encoder (ビジョンエンコーダ)** | 画像をモデルが理解できるベクトルに変える別構造。「画像もピクセル=数字」だからといって、テキスト LLM がすぐ画像を理解するわけではない。このプロジェクトはテキスト・コード専用なので持っていない。 |
| **RMSNorm** | 層正規化の一種。LayerNorm より計算がシンプルで、性能は同程度。 |
| **Pre-Norm** | 正規化を attention・FFN の **前** に適用。層が深くても学習が崩れにくい。 |
| **RoPE (Rotary Position Embedding)** | トークン位置を回転変換で attention に入れる方式。数式で計算するので長さ拡張 (NTK-aware、YaRN など) が比較的しやすい。それでも学習長よりずっと長く使うと性能は落ちる。 |
| **Self-Attention** | 文中のトークン同士が互いを見て関連度を計算する演算。Transformer の核心。 |
| **Causal masking** | 未来のトークンを先読みできないように隠す。next-token prediction に必須。 |
| **MHA (Multi-Head Attention)** | Self-Attention を複数 head に分けて並列計算。head ごとに違う関係 (文法、意味など) を学習できる。 |
| **GQA (Grouped-Query Attention)** | Query head より KV head を少なくして、推論時の KV キャッシュを減らす。MHA よりメモリ節約。 |
| **KV キャッシュ (Key-Value Cache)** | すでに計算した Key/Value を保存し、次のトークンで再利用。毎トークン全体を再計算しない。 |
| **Flash Attention / SDPA** | Self-Attention を GPU メモリ少なめ・速めに回すカーネル。PyTorch `F.scaled_dot_product_attention`。 |
| **FFN (Feed-Forward Network)** | Attention のあとに来る、トークンごとに独立適用される全結合網。 |
| **SwiGLU** | FFN 活性化の組み合わせ。GELU より良いとされ、最近の LLM がよく使う。 |
| **Residual connection (残差接続)** | 層の入力を出力にもう一度足す。深く積んでも情報が消えにくく、学習が安定。 |
| **LM Head** | 最後の hidden state を vocab サイズの logits に変える最終線形層。 |
| **Logits** | ソフトマックス前、次トークン候補ごとの「どれくらいもっともらしいか」の生スコア。 |

---

## 学習 (Training)

| 用語 | 説明 |
|---|---|
| **Pretraining (事前学習)** | 大量の生テキスト・コードで next-token prediction。言語構造・知識・コーディングパターンを体に染み込ませる最初の段階。 |
| **Next-token prediction** | 「これまでのトークンを見て、すぐ次を当てろ」。例: 「Pythonは」→「プログラミング」→ … 繰り返すと文が完成する。 |
| **Fine-tuning (微調整)** | 事前学習モデルを、より小さく目的がはっきりしたデータでもう一度学習。 |
| **SFT (Supervised Fine-Tuning)** | (質問, お手本の答) ペアで教師あり学習する微調整。 |
| **Instruction tuning** | 指示に従えるよう対話形式で微調整。SFT の一形態。 |
| **Loss (損失)** | 予測が正解とどれだけ違うか。学習はこの値を下げる方向にパラメータを動かす。 |
| **Cross-entropy** | 言語モデルでいちばんよく使う損失。正解トークンに低い確率を与えるほど大きくなる。 |
| **Loss masking (損失マスキング)** | 特定トークン (例: ユーザーの質問) を損失から除外 (`-100`)。モデルがその部分を「生成」するよう学ばせず、答側だけを学ばせる。 |
| **Perplexity** | validation loss を指数に変換。低いほど次トークン予測がうまくいっている。 |
| **Epoch / Step / Batch** | 全体一周 (epoch)、パラメータ一回更新 (step)、一度にまとめるデータ (batch)。 |
| **Gradient accumulation** | 大きなバッチを載せられないとき、複数ステップの gradient を積んでから一度に更新。 |
| **Optimizer / AdamW** | 損失を下げる方向にパラメータを実際に更新。AdamW が事実上の標準。 |
| **Learning rate / Warmup / Cosine decay** | 歩幅 (lr) を最初はゆっくり上げ (warmup)、その後コサインで下げる (decay)。 |
| **Gradient clipping** | gradient が大きくなりすぎないよう切り、発散を防ぐ。 |
| **Mixed precision / bfloat16** | 重い演算は 16 ビット (bfloat16)、重要な重み・オプティマイザ状態は 32 ビット。速度とメモリ節約。 |
| **Checkpoint (チェックポイント)** | 重みのスナップショットファイル。`ckpt/sft_nano.pt` のようなもの。 |
| **Chinchilla scaling law** | 決まった compute の中で、モデルサイズとトークン数をどの比率で伸ばすか。[README §2](README.ja.md#2-パラメータはゆっくり大きくする)。 |
| **FIM (Fill-In-the-Middle)** | 前後を与えて真ん中を埋めさせる学習。コード補完に重要。 |

---

## 推論 · 生成

| 用語 | 説明 |
|---|---|
| **Context window (コンテキストウィンドウ)** | 一度に参照できる最大トークン長。 |
| **Sampling (サンプリング)** | logits → 確率分布 → 次トークンをランダムに選ぶ。毎回同じ答にならないようにする。 |
| **Temperature** | 分布を平らに (多様) / 尖らせて (確信) 調整。 |
| **Top-p (nucleus sampling)** | 累積確率 p を超える上位候補だけ残してサンプリング。変な希少トークンを減らす。 |
| **THINKING モード (Chain-of-Thought)** | 答の前に `<THINKING>...</THINKING>` で思考過程を先に書かせる。[ARCHITECTURE](ARCHITECTURE.ja.md)。 |
| **Prefill** | プロンプト全体を一度通して KV キャッシュを先に埋める段階。 |

---

## RAG / Tool

| 用語 | 説明 |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | 関連文書を外部から検索してプロンプトに入れ、生成。再学習なしで最新知識を入れられる。 |
| **Tool use / Function calling** | モデルが「いまは電卓 / コード実行」のような外部ツールを呼ぶ能力。 |

---

## 後続学習 · 整列

| 用語 | 説明 |
|---|---|
| **RLHF** | 人の選好フィードバックを報酬にして、強化学習でモデルを合わせる方法一式。 |
| **Preference pair / Chosen / Rejected** | 同じ質問への二つの答のうち、良い方 (chosen) とそうでない方 (rejected)。 |
| **DPO (Direct Preference Optimization)** | 報酬モデルなしで選好ペアだけで整列。RLHF より実装が単純で安定。 |
| **Reward Model, RM** | 答に点をつけるよう学習したモデル。最後の層をスカラー出力に差し替える。 |
| **GRPO** | 同じ質問に複数答をサンプリングし、グループ内の相対優劣 (advantage) で更新。PPO と違い value モデル不要。 |
| **RLVR** | 数学・コーディングのように自動採点できる問題で、RM の代わりに「正解一致」などのルール報酬。 |
| **KL divergence / KL penalty** | 新しいモデルが参照モデルから遠く離れすぎないように抑える項。 |
| **Rejection sampling** | 複数候補のうち基準を通ったものだけを選んで学習データにする。 |
| **Reward hacking (報酬ハッキング)** | 実際の品質は上げず、報酬スコアだけ稼ぐ抜け道を学んでしまう現象。 |

---

## 評価

| 用語 | 説明 |
|---|---|
| **Benchmark (ベンチマーク)** | 標準問題セットで能力を測る。例: HellaSwag・MMLU (言語)、HumanEval・MBPP (コーディング)、GSM8K (数学)。 |
| **pass@1** | コードを一回生成したとき、テストを通る割合。 |

---

[README](README.ja.md) · [ARCHITECTURE](ARCHITECTURE.ja.md) · [POST-TRAINING](POST-TRAINING.ja.md)
