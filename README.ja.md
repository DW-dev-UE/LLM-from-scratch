<div align="center">

# 汎用LLMと社内業務用AIアーキテクチャ

<img src="https://img.shields.io/badge/CUDA-enabled-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/from--scratch-no%20external%20LLM-blueviolet?style=flat" alt="From scratch" />
<img src="https://img.shields.io/badge/status-actively%20updated-brightgreen?style=flat" alt="Actively updated" />

[한국어](README.md) &nbsp;·&nbsp; [English](README.en.md) &nbsp;·&nbsp; **[日本語](README.ja.md)**

</div>

---

> **🚧 このリポジトリは継続的に更新中です。**  
> 学習・ベンチマーク・ドキュメント・パイプラインを定期的に手入れしている。チェックポイント版
> （v1 → v2 …）、ベンチマークレポート、コマンド一覧も一緒に変わることがあるので、最新コミットを
> 基準に読んでほしい。Issue・PR歓迎。

NVIDIA CUDA上で動く汎用LLMと、コーディングやデザインといった分野で、会社の実務にそのまま使えるAIを、ゼロから
作るプロジェクトだ。トークナイザーから学習・推論エンジン、リリース後に人のフィードバックでモデルを
直していくパイプラインまで、全部自分の手で書いた。

既存のオープンソースLLMを持ってきてファインチューニングする方がずっと速くて実用的なのは分かっている。
それでもゼロから作り直す理由は一つ——LLMが内部でどう動いているかを自分の手で書いてみないと、結局は
他人のコードを借りて使うだけの人間で終わってしまうからだ。だからこのリポジトリは外部のLLMに頼らない。

ドキュメントは以下のように分かれている。順番に読んでもいいし、必要なところだけつまみ食いしてもいい。

| ドキュメント | 内容 |
|---|---|
| **README.ja.md**（今読んでいる文書） | 設計思想、パラメータ戦略、RAG・Toolの役割分担、プロジェクト構成、実行方法 |
| [GLOSSARY.ja.md](GLOSSARY.ja.md) | Transformer、weight、RoPE、DPOといった用語に馴染みがなければまずここから |
| [ARCHITECTURE.ja.md](ARCHITECTURE.ja.md) | モデル構造、トークナイザー、学習・推論エンジン——実際のコード込み |
| [POST-TRAINING.ja.md](POST-TRAINING.ja.md) | リリース後に人のフィードバックでモデルを直し続ける循環構造 |
| [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md) | Base V1 ベンチマークレポート（pretrain vs SFT、言語・コーディング詳細） |

実際の実装コードは [llm/](llm/) フォルダにあり、コマンドは
[llm/명령어-정리.md](llm/명령어-정리.md) にまとめてある。

## 目次

1. [アーキテクチャ設計の過程](#1-アーキテクチャ設計の過程)
2. [パラメータ数](#2-パラメータ数)
3. [RAGとTool](#3-ragとtool)
4. [プロジェクト構成](#4-プロジェクト構成)
5. [実行方法](#5-実行方法)
6. [ベンチマーク（V1）](#6-ベンチマークv1)
7. [参考文献](#参考文献)

---

## 1. アーキテクチャ設計の過程

最初はこんな流れで行くつもりだった。

1️⃣ 純粋な自然言語データセットでまずLLMを作る
2️⃣ その上にコーディング特化のデータセットを乗せる
3️⃣ 人間が直接強化学習で仕上げる

理由は単純だった。どれだけコーディングが得意なAIでも、人間は結局自然言語で質問を投げてくる。だから
「ユーザーがどんな文脈で何を求めているかを汲み取る力」を真っ先に身につけさせるべきだと考えていた。

ところがLlama 3の論文（*The Llama 3 Herd of Models*）を読んでみたら、そもそもこの順番自体が
存在しなかった。

> https://ar5iv.labs.arxiv.org/html/2407.21783

Llama 3はdense Transformer構造で、事前学習の段階でnext-token predictionによって言語構造と
世界知識を身につけたあと、後続学習の段階でinstruction following・preference alignment・
coding・reasoning・tool-useを強化していく。ただ、その事前学習自体が、最初から自然言語（約50%）＋
数学・推論（25%）＋コード（17%）＋多言語（8%）を一つのデータミックスとして混ぜ込んだ状態で進む。
「自然言語を先に完成させてからコードを乗せる」という段階は最初から存在しないわけだ。

DeepSeek-Coderは逆の方向から同じことを裏付けている。1.3B〜33Bのモデルを2Tトークンで最初から
学習させ、repository単位のコーパスと16Kコンテキストでのfill-in-the-blank（FIM）タスクを使った。
データ構成は87%がソースコード、10%がコード関連の英語、3%がコードと無関係な中国語だった。

> https://arxiv.org/html/2401.14196v1

自然言語の比率がたった13%しかないのに、当時のオープンソースcode LLMの中で最高の性能を出した。
両方の論文とも「自然言語を単独で先に完成させる」という段階を踏まず、最初から自然言語とコードを
一緒に学習させている。

つまり、最初に立てた順番が間違っていたということだ。

**捨てた順番**

```
自然言語のみ学習 → コードのみ学習 → instruction tuning
```

自然言語の能力はもちろん必要だ。ただ、それを切り離して先に「完成」させる必要はなかった。だから
順番を変えた。

**変えた順番**

```
自然言語 + コード + 数学 + 文書 混合 base pretraining
  → code-heavy continued pretraining
    → instruction tuning
      → preference optimization
```

### データ構成比率 (300M ~ 1B 基準)

| データ種別 | 比率 | 備考 |
|---|---|---|
| Raw source code | 35% ~ 45% | |
| Code-adjacent text | 15% ~ 20% | README、docs、issue、PR、commit |
| General technical natural language | 20% ~ 30% | 英語 + 韓国語の技術文書 |
| Math / reasoning | 5% ~ 10% | |
| Logs / tests / errors / diffs | 5% ~ 10% | |
| Korean instruction / bilingual data | 5% ~ 10% | |

---

## 2. パラメータ数

パラメータはモデルの大きさだ（用語がややこしければ[GLOSSARY.ja.md](GLOSSARY.ja.md)を参照）。
大きければ大きいほど賢くなる可能性は高いが、このプロジェクトでは300M → 1B → 3Bとゆっくり
大きくしていくやり方を選んだ。

Chinchillaの論文（*Training Compute-Optimal Large Language Models*）は、同じcompute予算
なら、モデルサイズと学習トークン数を一緒に増やすべきだと言っている。

> https://arxiv.org/abs/2203.15556

70M〜16B超のパラメータ、5B〜500Bトークンの範囲で400以上のモデルを回してみた結果、モデルサイズが
2倍になればトークン数も2倍にしないとcompute-optimalにならない、という結論だった。GPT-3、
Gopher、Jurassic-1、Megatron-Turing NLGといった当時のモデルは実際サイズの割にデータが足りて
おらず、Chinchilla（70B、1.4Tトークン）は4倍大きいGopher（280B、300Bトークン）と同じcompute
でMMLUなどより良いスコアを出した。

だから闇雲にパラメータから増やすんじゃなくて、小さいモデルでtokenizer・dataloader・loss curve・
eval・checkpointのパイプラインがちゃんと回るか先に確認してから規模を上げるのが安全だ。StarCoder2
はこれを実際にやって見せた例だ——3B/7B/15Bをそれぞれ違うトークン数（約3.1T/3.5T/4.1T）で個別に
学習させて、デプロイ予算に合わせて選べるモデルファミリーを一気に出してきた。（15Bモデルは2倍以上
大きいCodeLlama-34Bに匹敵する性能を出した。）

> https://arxiv.org/abs/2402.19173

このプロジェクトが実際に踏んでいくステップはこうだ。

| 規模 | 目的 |
|---|---|
| 10M ~ 50M | 学習ループ、tokenizer、lossが下がるかの確認 |
| 100M ~ 300M | コードcompletion、FIM、基本的なinstructionの実験 |
| 1B | 小規模なcoding assistantの実験 |
| 3B | 社内ツールと連携できる実用候補 |
| 7B+ | 外部サービスとして公開できる最小規模 |

---

## 3. RAGとTool

企業特化LLMだからといって、あらゆる知識をモデルの重みに詰め込もうとしてはいけない。最新のコード、
最新のドキュメント、テスト結果、ビルドログのように絶えず変わる情報は、RAGとtool呼び出しでつなぐ
方がいい。新しい技術が出るたびにモデル全体を再学習するなんて、とても回らない。

役割はこう分ける。

| 担当 | 項目 |
|---|---|
| **Model（重み）** | 思考の仕方、コーディングパターン、文脈理解、回答スタイル、tool使用能力 |
| **RAG / Tools（外部接続）** | 最新知識、社内文書、実際のrepo、テスト実行、ビルド結果、デプロイ環境 |

モデルは「どう考えて、どう道具を使うか」を担当し、RAG/Toolは「今、何が事実なのか」を担当する。

---

## 4. プロジェクト構成

```
AI/
├── README.md              # この文書 — 設計思想、パラメータ戦略、プロジェクト構成、実行方法
├── README.en.md            # English
├── README.ja.md            # 日本語
├── GLOSSARY.md              # 用語整理 (ko / en / ja)
├── GLOSSARY.en.md
├── GLOSSARY.ja.md
├── ARCHITECTURE.md          # モデルアーキテクチャ・学習・推論 (ko / en / ja)
├── ARCHITECTURE.en.md
├── ARCHITECTURE.ja.md
├── POST-TRAINING.md         # 後続学習（人間フィードバック） (ko / en / ja)
├── POST-TRAINING.en.md
├── POST-TRAINING.ja.md
├── BENCHMARK-v1.md          # Base V1 ベンチマークレポート (ko / en / ja)
├── BENCHMARK-v1.en.md
├── BENCHMARK-v1.ja.md
└── llm/                      # 実際の実装
    ├── model.py               # アーキテクチャ — RMSNorm+RoPE+SwiGLU+GQA, KVキャッシュ, hidden()
    ├── tokenizer.py            # トークナイザー学習/ロード
    ├── data.py                # データ準備 — demo / pretrain / sft・校正・混合 / ratings / hygiene
    ├── train.py                # 学習ループ — pretrain / sft / dpo, --tag バージョン・スナップショット
    ├── infer.py                # 推論エンジン + CLIチャット、対話ログ自動保存
    ├── feedback.py             # LLM Feedback Studio — フィードバック収集Web UI
    ├── reward.py               # 報酬モデル — GPTバックボーン + スカラーhead、ランキング損失
    ├── rlhf.py                 # GRPOポリシー最適化（RLVRルールベース報酬 / RMスカラー報酬）
    ├── eval_gate.py            # 昇格ゲート — ホールドアウト回帰loss + win-rate
    ├── 명령어-정리.md            # 全パイプライン実行コマンド集
    ├── tokenizer.json
    ├── data/                   # corpus, sft/preference/feedback/ratings jsonl, rated_sft.jsonl, train/val bin
    ├── ckpt/                   # pretrain/sft/dpo/rm/grpo チェックポイント + ベンチマークJSON
    └── logs/                   # chat_log.jsonl — 推論対話ログ
```

---

## 5. 実行方法

全体のコマンドは[llm/명령어-정리.md](llm/명령어-정리.md)にステップごとにまとめてある。最初から
最後まで一通り動かしてみるなら:

```bash
cd llm
pip install torch tokenizers numpy

python data.py demo                                        # 1. サンプルデータ生成
python tokenizer.py --input data/corpus.txt data/sft.jsonl # 2. トークナイザー学習
python data.py pretrain                                    # 3. 事前学習データ準備
python train.py --mode pretrain --preset nano --steps 2000 # 4. 事前学習
python data.py sft                                         # 5. SFTデータ準備
python train.py --mode sft --init ckpt/pretrain_nano.pt --steps 300  # 6. SFT
python infer.py --ckpt ckpt/sft_nano.pt                    # 7. チャット（ログ自動記録）
```

リリース後に人のフィードバックで1サイクル回す方法は[POST-TRAINING.ja.md](POST-TRAINING.ja.md)で
扱う。かいつまんで見ると:

```bash
python infer.py --ckpt ckpt/sft_nano.pt          # 1. 対話 → logs/chat_log.jsonl 収集
python feedback.py                               # 2. Web UIで評価・校正・選好ラベリング

# トラックA — 校正SFT
python data.py sft --input data/sft.jsonl data/feedback.jsonl --weight 1 3
python train.py --mode sft --init ckpt/sft_nano.pt --steps 100 --tag v1

# トラックB — DPO
python train.py --mode dpo --init ckpt/sft_nano.pt --steps 100 --tag v1

# トラックC — 報酬モデル + GRPO / RLVR
python reward.py --init ckpt/sft_nano.pt --steps 200
python rlhf.py --mode rm   --init ckpt/sft_nano.pt --rm ckpt/rm_nano.pt --data data/sft.jsonl
python rlhf.py --mode rlvr --init ckpt/sft_nano.pt --data data/sft.jsonl

# 昇格ゲートを通過したら、新しいチェックポイントで1番から繰り返す
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano_v1.pt
```

実運用では`data/corpus.txt`を実際のコーパス（TinyStories、FineWeb-Eduなど）に、
`data/sft.jsonl`を公開instruction/reasoningデータに差し替えて、`--preset small|base`で
規模を上げればいい。

---

## 6. ベンチマーク（V1）

`base`プリセット（約327M）で学習した **pretrain_base_v1** と **sft_base_v1** を、同一14問・2モード
（THINKING on/off）で測った結果。詳しい表・解釈・設問ごとのノートは言語別レポートを参照。

| ドキュメント | 言語 |
|---|---|
| [BENCHMARK-v1.md](BENCHMARK-v1.md) | 한국어 |
| [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md) | English |
| [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md) | 日本語 |

**一行結果（sft_base_v1, no-thinking）**

| 領域 | 結果 |
|---|---|
| 韓国語平均 | 1.33 / 5 |
| 日本語平均 | 0.33 / 5 |
| 英語平均 | 2.67 / 5 |
| コーディングテスト | 4 / 17（完全通過 1/5 — `is_palindrome`） |
| THINKINGモード | **回答 0/14**（`</THINKING>` 未終了 — `--no-thinking` 推奨） |

pretrainだけでは28問すべて0点（チャットSFTなしの教科書的失敗）。SFT 5,000 step後に
「指示を試す」段階へ上がる。まだearly-stageで、後続バージョンとベンチマークは継続追加予定。

元JSON: [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json),
[`llm/ckpt/benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json)

---

## 参考文献

- Llama Team, AI @ Meta. *The Llama 3 Herd of Models.* — https://ar5iv.labs.arxiv.org/html/2407.21783
- Guo, D. et al. *DeepSeek-Coder: When the Large Language Model Meets Programming — The Rise of Code Intelligence.* — https://arxiv.org/html/2401.14196v1
- Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* — https://arxiv.org/abs/2203.15556
- Lozhkov, A. et al. *StarCoder 2 and The Stack v2: The Next Generation.* — https://arxiv.org/abs/2402.19173

<div align="center">

---

**次に読むといいドキュメント**

[📖 GLOSSARY.ja.md](GLOSSARY.ja.md) — まずは用語を確認 &nbsp;|&nbsp;
[🏗️ ARCHITECTURE.ja.md](ARCHITECTURE.ja.md) — モデル構造・学習・推論 &nbsp;|&nbsp;
[🔁 POST-TRAINING.ja.md](POST-TRAINING.ja.md) — 人のフィードバックによる後続学習 &nbsp;|&nbsp;
[📊 BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md) — Base V1 ベンチマーク

</div>
