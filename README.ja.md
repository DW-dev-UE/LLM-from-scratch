<div align="center">

# 汎用 LLM · 社内向け AI

**トークナイザーから学習 · 推論 · フィードバックまで、外部 LLM なしで自作**

<br/>

![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?style=flat-square&logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![From scratch](https://img.shields.io/badge/from%20scratch-no%20external%20LLM-7c3aed?style=flat-square)
![Status](https://img.shields.io/badge/status-actively%20updated-22c55e?style=flat-square)

<br/>

[한국어](README.md) · [English](README.en.md) · [日本語](README.ja.md)

</div>

---

> **このリポジトリは継続更新中です。**  
> 学習結果・ベンチ・ドキュメントは変わります。最新コミットを見てください。Issue / PR 歓迎。

---

## 何をするか

NVIDIA CUDA 上の **汎用 LLM** と、コーディングなど実務向け AI を  
**ゼロから** 作る実験です。

- トークナイザー学習  
- Decoder-only Transformer  
- 事前学習 · SFT · DPO · GRPO  
- KV キャッシュ推論 · 思考モード  
- 人のフィードバック UI · 昇格ゲート  

既存 OSS をファインチューニングする方が速いのは分かっています。  
それでも作り直す理由は一つです。

> 内部を自分で書かないと、結局は他人のコードを借りるだけになる。

外部の LLM 重みには依存しません。

---

## ドキュメント

| 文書 | 内容 |
|:-----|:-----|
| **README**（本頁） | 設計思想 · パラメータ · 構成 · 実行 |
| [GLOSSARY](GLOSSARY.ja.md) | 用語 |
| [ARCHITECTURE](ARCHITECTURE.ja.md) | モデル · 学習 · 推論 |
| [POST-TRAINING](POST-TRAINING.ja.md) | フィードバック後続学習 |
| [BENCHMARK v1](BENCHMARK-v1.ja.md) | Base ベンチ（学習過程 · Q&A） |

実装: [`llm/`](llm/)  
コマンド: [`llm/명령어-정리.md`](llm/명령어-정리.md)

---

## 目次

1. [設計をこう変えた](#1-設計をこう変えた)  
2. [パラメータは段階的に](#2-パラメータは段階的に)  
3. [モデル vs RAG / Tool](#3-モデル-vs-rag--tool)  
4. [構成](#4-構成)  
5. [実行](#5-実行)  
6. [ベンチの要約](#6-ベンチの要約)  
7. [参考文献](#7-参考文献)

---

## 1. 設計をこう変えた

### 最初の案

| 段 | 内容 |
|:--:|:-----|
| **1** | 自然言語だけで先に LLM を作る |
| **2** | その上にコード特化データを載せる |
| **3** | 人が強化学習で整える |

コード AI でも質問は自然言語なので、  
**理解力を先に完成させる** つもりでした。

### 論文が示す流れ

[Llama 3](https://ar5iv.labs.arxiv.org/html/2407.21783) にその順序はありません。  
事前学習から既に混ぜます。

| おおよそ | データ |
|:--------:|:-------|
| 50% | 自然言語 |
| 25% | 数学 · 推論 |
| 17% | コード |
| 8% | 多言語 |

後続学習で instruction などを強化するだけです。  
**「言語を完成してからコード」はありません。**

[DeepSeek-Coder](https://arxiv.org/html/2401.14196v1) も同様で、  
最初からコード比率が高くても強い結果を出しています。

### いまのパイプライン

| | 順序 |
|:--|:-----|
| 以前の案 | 自然言語のみ → コードのみ → instruction |
| **現在** | **混合 pretrain** → コード強化 → instruction → preference |

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

### データ比率の目安（300M〜1B）

| 種類 | 比率 | 例 |
|:-----|:----:|:---|
| Raw source code | 35–45% | ソース |
| Code-adjacent | 15–20% | README, issue, PR |
| 技術自然言語 | 20–30% | EN / KO 技術文書 |
| Math / reasoning | 5–10% | 数学 · 論理 |
| Logs / tests / diffs | 5–10% | ログ · テスト |
| 韓国語 · bilingual | 5–10% | 指示 · 二言語 |

---

## 2. パラメータは段階的に

パラメータ = モデルの大きさ（[GLOSSARY](GLOSSARY.ja.md)）。  
大きいほど余地はあるが、本プロジェクトは **300M → 1B → 3B** と段階を踏みます。

[Chinchilla](https://arxiv.org/abs/2203.15556) の要点:

> 同じ compute なら **モデルとトークンを一緒に** 増やす。

手順:

1. 小さいモデルで tokenizer · loader · loss · eval · checkpoint を通す  
2. その後スケールアップ  

[StarCoder2](https://arxiv.org/abs/2402.19173) も 3B / 7B / 15B を別トークン予算で学習した例です。

### 段階

| 規模 | 目的 |
|:-----|:-----|
| 10M〜50M | ループ · tokenizer · loss |
| 100M〜300M | completion · FIM · 基本 instruction |
| 1B | 小さな coding assistant |
| 3B | 社内ツール連携候補 |
| 7B+ | 外部公開を考える最小規模 |

V1 ベンチの実学習は **base 約 327M**。[BENCHMARK v1](BENCHMARK-v1.ja.md)

---

## 3. モデル vs RAG / Tool

社内知識を全部重みに入れるのは無理があります。  
コードと文書は変わり続けます。

| 担当 | 内容 |
|:-----|:-----|
| **モデル（重み）** | 考え方 · コーディング型 · 文体 · ツールの **使い方** |
| **RAG / Tools** | 今の事実 · 社内文書 · repo · テスト · ビルド |

> モデル = どう考え、どう道具を使うか  
> RAG / Tool = 今何が正しいか  

（RAG 実行系はロードマップ。公開範囲の中心は学習 · 推論 · 整列です。）

---

## 4. 構成

```text
AI/
├── README*.md
├── GLOSSARY*.md
├── ARCHITECTURE*.md
├── POST-TRAINING*.md
├── BENCHMARK-v1*.md
└── llm/                 実装
    ├── model.py
    ├── train.py · infer.py
    ├── feedback.py · reward.py · rlhf.py
    └── eval_gate.py
```

> 大きな重みや生コーパスはリポジトリに含めません。  
> いまは設計とベンチの文書が主な公開面です。

---

## 5. 実行

詳細は [llm/명령어-정리.md](llm/명령어-정리.md)。

```bash
cd llm
pip install torch tokenizers numpy

python data.py demo
python tokenizer.py --input data/corpus.txt data/sft.jsonl
python data.py pretrain
python train.py --mode pretrain --preset nano --steps 2000
python data.py sft
python train.py --mode sft --init ckpt/pretrain_nano.pt --steps 300
python infer.py --ckpt ckpt/sft_nano.pt
```

### フィードバック 1 周（要約）

```bash
python infer.py --ckpt ckpt/sft_nano.pt
python feedback.py

python data.py sft --input data/sft.jsonl data/feedback.jsonl --weight 1 3
python train.py --mode sft --init ckpt/sft_nano.pt --steps 100 --tag v1
python train.py --mode dpo --init ckpt/sft_nano.pt --steps 100 --tag v1

python reward.py --init ckpt/sft_nano.pt --steps 200
python rlhf.py --mode rm   --init ckpt/sft_nano.pt --rm ckpt/rm_nano.pt --data data/sft.jsonl
python rlhf.py --mode rlvr --init ckpt/sft_nano.pt --data data/sft.jsonl

python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano_v1.pt
```

本番では公開コーパスに差し替え、`small` / `base` で拡大します。

---

## 6. ベンチの要約

同一 14 問 × THINKING on/off、`base` 約 327M。

| チェックポイント | 要約 |
|:-----------------|:-----|
| pretrain v1 | チャット形式でほぼ 0 点 |
| **sft v1** | 指示は試す · 正答率はまだ低い |

**sft_base_v1 · 通常チャット（THINKING オフ）**

| 領域 | 結果 |
|:-----|:-----|
| 韓国語 | 平均 1.33 / 5 |
| 日本語 | 平均 0.33 / 5 |
| 英語 | 平均 2.67 / 5 |
| コーディング | テスト 4 / 17（完全通過 1/5） |
| THINKING オン | 回答 0 / 14（終了タグ問題） |

詳細（学習 · データ · 設問ごと）:

- [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md)  
- [BENCHMARK-v1.md](BENCHMARK-v1.md) · [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md)  

まだ early-stage。以降も同じ 14 問で比較します。

---

## 7. 参考文献

| 論文 | リンク |
|:-----|:-------|
| Llama 3 | https://ar5iv.labs.arxiv.org/html/2407.21783 |
| DeepSeek-Coder | https://arxiv.org/html/2401.14196v1 |
| Chinchilla | https://arxiv.org/abs/2203.15556 |
| StarCoder 2 | https://arxiv.org/abs/2402.19173 |

---

<div align="center">

**次に読む**

[用語](GLOSSARY.ja.md) · [アーキテクチャ](ARCHITECTURE.ja.md) · [後続学習](POST-TRAINING.ja.md) · [ベンチ](BENCHMARK-v1.ja.md)

</div>
