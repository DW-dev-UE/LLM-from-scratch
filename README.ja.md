[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](README.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](README.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-0969DA?style=flat-square)](README.ja.md)

# ゼロから作る LLM

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)

> [!NOTE]
> **このリポジトリは継続更新中です。**  
> 学習結果・ベンチ・ドキュメントはバージョンごとに変わります。最新コミットを見てください。Issue / PR 歓迎。

> 始める前に、この文章は読みやすさ重視で磨かれていないかもしれません。
> AI に「読みやすく整えて」と頼むこともできますが、そうすると自分が何を考え、どんな過程を経たかが消えてしまうことがあります。
> AI で学ぶのは良いですが、文を何度も書き直し自分の知識を整理する方が大切です。

> [!IMPORTANT]
> **APEX-1（1B 級）— Pretrain・SFT・DPO 完了**
>
> 実測 **1,119.5M** パラメータ。( 24層 · d_model 2048 · GQA（16Q/4KV）+ RoPE + SwiGLU + RMSNorm )
>
> - 📏 **コンテキスト**: max 4096（学習は 2048）
> - 🔥 **Pretrain**: 51K step · 20B トークン · v3-en 80.8GB · 英語専用
>
> | モデル | 学習トークン | HellaSwag | ARC(平均) | PIQA | GSM8K | HumanEval |
> |:---|---:|---:|---:|---:|---:|---:|
> | **Apex-1 DPO(1.1B)** | 0.02T | 46.9 | 41.6 | 68.6 | 1.9 | 8.5 |
> | Pythia-1.0B | 0.3T | 47.2 | 38.0 | 69.2 | — | 1.8 |
> | OPT-1.3B | 0.18T | 53.7 | 40.1 | 72.4 | — | — |
> | TinyLlama-1.1B | 3T | 59.2 | 42.7 | 73.3 | — | 9.2 |
> | OLMo-1B | 2T | 62.5 | 46.3 | 73.7 | — | — |
> | Llama-3.2-1B | 9T | 61.2 | 49.2 | 74.8 | 7.6 | 18.9 |
> | Qwen2.5-1.5B | 18T | 66.4 | 58.5 | 76.1 | 61.7 🏆 | 37.2 🏆 |
> | SmolLM2-1.7B | 11T | 68.7 🏆 | 60.5 🏆 | 77.6 🏆 | 31.1 | 22.6 |
>
> **DPO ≥ SFT、alignment tax なし** · Apex-1 は15倍多く学習した Pythia と常識平均で同等、ARC は上回る(トークン効率は最上位) · Llama-3.2・Qwen2.5 とのギャップはアーキテクチャでなくデータ規模450〜900倍の差 → [BENCHMARK v2 §5.4](BENCHMARK-v2.ja.md#54-標準ベンチマーク-lm-evaluation-harness)
>
> <sub>**ベンチマーク出典**(各分野の標準的な被引用論文): HellaSwag [Zellers+19](https://arxiv.org/abs/1905.07830) · ARC [Clark+18](https://arxiv.org/abs/1803.05457) · PIQA [Bisk+19](https://arxiv.org/abs/1911.11641) · GSM8K [Cobbe+21](https://arxiv.org/abs/2110.14168) · HumanEval [Chen+21](https://arxiv.org/abs/2107.03374)</sub>  
> <sub>**比較モデル出典**: Pythia [Biderman+23](https://arxiv.org/abs/2304.01373) · OPT [Zhang+22](https://arxiv.org/abs/2205.01068) · TinyLlama [Zhang+24](https://arxiv.org/abs/2401.02385) · OLMo [Groeneveld+24](https://arxiv.org/abs/2402.00838) · Qwen2.5 [Qwen+24](https://arxiv.org/abs/2412.15115) · SmolLM2 [Ben Allal+25](https://arxiv.org/abs/2502.02737) · Llama-3.2(arXiv レポートなし、Meta ブログのみ)</sub>
>
> - ⏸️ **見送り**: MoE · YaRN · FP8 · マルチGPU（1B 検証後の次スケール）

> [!NOTE]
> 🚀 **次の目標 — APEX-2（7B 級）**: アーキテクチャ開発進行中

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

```ini
[326.7M 級モデルを開発して得た教訓]

1. 多言語の非効率

小規模モデルの制約下で多言語をサポートするのは、~7B 級モデルまでは致命的。v7 が示したように、小型の多言語モデルは三言語とも中途半端な状態に収束する。

70B 級以上になると言語間転移がむしろ得になるが、小規模モデルでは英語圏コーパスが量・質ともに圧倒的なので、小規模モデルでは英語に焦点を絞るべき。

2. コーパスの質がかなり重要

性能 = パラメータ数 x トークン数 x データ品質。既存の公開コーパスであっても、質の良いコーパスを探すことが最優先。

3. ベンチマークのノイズ

学習を終えてベンチマークを回すとき、temperature 0.7 で問題ごとに 1 回だけサンプリングしていた。同じモデルでも実行のたびに違う答えを出すので、バージョン間のスコア差が実力差なのかサイコロなのか区別できなかった。

解決: v5 から greedy プロトコルを使い、同じモデルは常に同じ答えを出すようにした。

ただし、val loss にも騙された。

val はコーパスファイル順の最後の 1% だったが、ko-wiki の末尾という単一ドメインで、全体の能力を代表していなかった。(v2 では ko-wiki val は悪化したが、下流ベンチマークは過去最高だった。)

測定がランダム 16 シーケンス 1 バッチだったため ±0.25 のノイズがあり、この揺れを実際の改善/悪化と誤読して二度判断を誤った。つまり二度騙された。

解決: 複数バッチ平均でのみ判断 + v3-en ダウンローダーにソースごと 1% ずつの混合 val を構造的に組み込んだ。

4. thinking の崩壊 (v1〜v5、最大のミス)

正解を `<THINKING>` の中に書いて、肝心の回答欄は空白になっていた。6 バージョン連続で thinking コーディング 0/5。
原因は全部で 3 つあった。

> thinking 学習データの 99% が max_len=1024 で思考の途中で切断されていた。
> THINK_SUFFIX が学習/推論間で不一致だった。
> infer.py で思考/回答のトークン予算が分離されておらず、思考が予算を使い切ると回答が空白になっていた。

解決: data.py で thinking の末尾を保存(長ければ思考の途中を切っても閉じタグ+回答は保存: 47,273 件、それでも長すぎる場合は no-thinking に降格: 17,540 件) + THINK_SUFFIX を条件付きで付与 + infer.py で予算を分離・EOS ガードを追加。

この対応を入れると、thinking の回答が 10/14→14/14、平均が 0.50→2.93、初のコード出力が 4/5 のスコアになった。

共通する教訓は、三つとも「モデルが馬鹿だから」ではなく、測定器か配管が壊れていたということ。学習モデルにばかり焦点を当てていたせいで起きた問題であり、次回からはスコアがおかしくても、まずサンプリング・val 設計・前処理の順に疑うことにする。
```

---

## ドキュメント案内

| 文書 | 何を見るか |
|:-----|:-----------|
| **README**（本頁） | 設計思想、パラメータ戦略、構成、実行 |
| [GLOSSARY](GLOSSARY.ja.md) | Transformer、RoPE、DPO などの用語 |
| [ARCHITECTURE](ARCHITECTURE.ja.md) | トランスフォーマーのウォークスルー · モデル設計 · トークナイザー · 学習 · 推論 |
| [POST-TRAINING](POST-TRAINING.ja.md) | デプロイ後の人間フィードバックループ |
| [BENCHMARK v2](BENCHMARK-v2.ja.md) | APEX-1(1B)ベンチ・RLVR → DPO 転換記録 |
| [BENCHMARK v1](BENCHMARK-v1.ja.md) | Base モデル(327M)ベンチ(学習過程 · Q&A 含む) |
| [ThinkingLab](ThinkingLab/ThinkingLab.ja.md) | 仮説・ブレインストーミングログ（まだ検証されていない考え） |

---

## 1. アーキテクチャ

トークン → 埋め込み → アテンション → FFN → logit までの **数値例ウォークスルー** は [ARCHITECTURE — トランスフォーマーの動作原理](ARCHITECTURE.ja.md#2-トランスフォーマーの動作原理教育用ウォークスルー) に移しました。（教育用例と実際の LLaMA スタイル設計の違いは、その文書で続けて説明します。）

### 最初に考えていた道

最初の計画は単純でした。

1. 自然言語データだけで先に LLM を作る
2. その上にコーディング特化データを載せる
3. 人が強化学習で整える

コーディング AI でも質問は自然言語で来るので、  
**「言葉を理解する力」を先に完成させる** つもりでした。

### 論文が示したこと

[Llama 3](https://ar5iv.labs.arxiv.org/html/2407.21783) を見ると、その順序はありません。  
事前学習の段階から既に全ドメインを一つに混ぜ合わせます。

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

> [!TIP]
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

1B(APEX-1)まで Pretrain・SFT・DPO を終え、3B は飛ばして直接 **APEX-2(7B 級)** を開発中です。

2つの系列を並行しています。1B `Apex-1` 系列の最新は **`dpo_Apex-1_v1`**(pretrain+SFT+DPO 完了、標準ベンチ11種測定済み)、327M `base` 系列の最新は **`sft_base_v6`** です。次の目標は **APEX-2(7B 級)** です。→ [§5 ベンチマーク一覧](#5-ベンチマーク一覧) · バージョン別詳細記録 [BENCHMARK v2](BENCHMARK-v2.ja.md)

---

## 3. モデル vs RAG · Tool

会社の知識を全部重みに入れようとすると耐えられません。  
コード、文書、ビルドログは変わり続けるからです。

| 担当 | 担うこと |
|:-----|:---------|
| **モデル（重み）** | 考え方、コーディングパターン、文脈理解、回答のトーン、ツールの **使い方** |
| **RAG / Tools** | 最新の事実、社内文書、repo、テスト実行、ビルド結果 |

> [!IMPORTANT]
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
├── BENCHMARK-v2.md           APEX-1(1B) ベンチレポート
├── BENCHMARK-v1.md           327M ベンチレポート
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

2つの系列が並行しています: **1B `Apex-1`**(英語専用)と **327M `base`**(韓/日/英 多言語)。セットも別物です。

### 5.1 1B `Apex-1` 系列 ⭐ 主力

英語専用15問 × THINKING on/off(コーディング5問は327Mセットと同一問題)。  
最新スナップショット: **`sft_Apex-1_v1`**(`ckpt/benchmark_sft_Apex-1_v1_raw.json`、2026-07-22)。Pretrain 51K step(20B トークン)+ SFT 8.4K step + RLVR 廃止後の DPO まで全て完了(§6 in [BENCHMARK v2](BENCHMARK-v2.ja.md))。

**sft_Apex-1_v1**

| | 💬 通常チャット | 🧠 THINKING オン |
|:--|:--:|:--:|
| QA 正解(キーワード) | 5/8 | 4/8 |
| コーディング完全通過 | **5/5**(テスト25/25) | 3/5(テスト15/25) |
| 空回答 | 0件 | 0件 |
| 長さ超過 | 8/10 | 5/10 |
| 平均反復率 | 0.032 | 0.042 |

327M 系列の最大の失敗だった THINKING ハンドオフ(回答欄が空)は、この系列では **最初から発生しませんでした。** 代わりに新しいボトルネックは冗長さ・繰り返しで、THINKING をオンにするとコーディング・QA ともむしろ下がります(思考が回答予算を先に消費する)。

能力(コーディング5/5満点)は十分と判断し、最初は **RLVR** を選びましたが、実際に回してみると GSM8K 正解率が~0%で GRPO グループ内の報酬分散が0になり、学習信号が全く無いことが分かりました。冗長さ・指示不遵守という元の問題は正解能力が不要な **DPO** で直接狙えるため転換し、60K 選好ペアで学習を完了しました。

詳細記録: [BENCHMARK-v2.ja.md](BENCHMARK-v2.ja.md)

### 5.2 標準ベンチマーク(lm-evaluation-harness)

`sft_Apex-1_v1` と `dpo_Apex-1_v1` を HF 形式に変換し(ロジット一致検証 max diff 2.3e-5)、標準ハネスで**公開スコアのある他モデルと並べて**測定しました。常識7種は0-shot、MMLU は5-shot、GSM8K は5-shot、HumanEval・MBPP は0-shot pass@1 — TinyLlama 論文(Table 2・3)と同じプロトコルです。

| 項目 | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B |
|:---|---:|---:|---:|---:|
| 常識7種平均(0-shot) | 49.54 | 49.65 | 52.99 | 48.30 |
| MMLU(5-shot) | 24.83 | 24.90 | 25.34 | 25.70 |
| GSM8K(5-shot、strict) | 1.44 | 1.90 | — | — |
| HumanEval(pass@1) | 8.54 | 8.54 | 9.15 | 1.83 |
| MBPP(pass@1) | 4.80 | 5.20 | — | — |

> [!IMPORTANT]
> **DPO ≥ SFT、alignment tax なし** — 全項目で DPO が SFT と同等以上。**BoolQ 62.20 は比較4モデル(TinyLlama-1.1B・Pythia-1.0B・OPT-1.3B 含む)中1位**で、~20Bトークンしか学習していない Apex-1 が300Bトークンの Pythia-1.0B を HumanEval で大差(8.54 vs 1.83)で上回っています。

**比較モデル**: TinyLlama-1.1B(1.1Bパラメータ・3Tトークン、2024年)・Pythia-1.0B(1.0Bパラメータ・300Bトークン、EleutherAI)・OPT-1.3B(1.3Bパラメータ・180Bトークン、Meta)— Apex-1(1.12Bパラメータ・~20Bトークン)対比でそれぞれトークン数150倍・15倍・9倍です。

常識7種・MMLU・GSM8K・HumanEval・MBPP がそれぞれ何を測定し、業界でどれほど権威があるか、詳細な7種内訳(OPT含む)は → [BENCHMARK-v2.ja.md §5.4](BENCHMARK-v2.ja.md#54-標準ベンチマーク-lm-evaluation-harness)

**より広い比較** — 学習トークン規模がずっと大きい2024〜2025年の小型オープンモデルまで含めると(🏆 = 各列1位):

| モデル | 学習トークン | HellaSwag | ARC(平均) | PIQA | GSM8K | HumanEval |
|:---|---:|---:|---:|---:|---:|---:|
| **Apex-1 DPO(1.1B)** | 0.02T | 46.9 | 41.6 | 68.6 | 1.9 | 8.5 |
| Pythia-1.0B | 0.3T | 47.2 | 38.0 | 69.2 | — | 1.8 |
| OPT-1.3B | 0.18T | 53.7 | 40.1 | 72.4 | — | — |
| TinyLlama-1.1B | 3T | 59.2 | 42.7 | 73.3 | — | 9.2 |
| OLMo-1B | 2T | 62.5 | 46.3 | 73.7 | — | — |
| Llama-3.2-1B | 9T | 61.2 | 49.2 | 74.8 | 7.6 | 18.9 |
| Qwen2.5-1.5B | 18T | 66.4 | 58.5 | 76.1 | 61.7 🏆 | 37.2 🏆 |
| SmolLM2-1.7B | 11T | 68.7 🏆 | 60.5 🏆 | 77.6 🏆 | 31.1 | 22.6 |

> [!IMPORTANT]
> **トークン効率で読むと** — Apex-1(0.02T)は15倍多く学習した Pythia-1.0B(0.3T)と常識平均で同等、ARC はむしろ上回ります。HumanEval 8.5 は Pythia(1.83)の**4.7倍**で、3兆トークンの TinyLlama(9.15)にも近い水準 — ~20Bトークンのモデルとしては異例です。Llama-3.2-1B(9T)・Qwen2.5-1.5B(18T)とのギャップはアーキテクチャではなく**データ規模450〜900倍の差**です。

SmolLM2・Qwen2.5 は Apex-1 より550〜900倍多いトークンで学習されています — ここで劣るのは想定内で、**学習トークン規模が近いモデルとの比較(上の表)** の方がこのプロジェクトの実際の立ち位置を公正に示します。詳細: [BENCHMARK-v2.ja.md §5.4](BENCHMARK-v2.ja.md#54-標準ベンチマーク-lm-evaluation-harness)

<details>
<summary><b>5.3 327M `base` 系列(展開して見る)</b></summary>

<br/>

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

</details>

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

[用語](GLOSSARY.ja.md) · [アーキテクチャ](ARCHITECTURE.ja.md) · [後続学習](POST-TRAINING.ja.md) · [ベンチマーク v1](BENCHMARK-v1.ja.md) · [ベンチマーク v2](BENCHMARK-v2.ja.md) · [思考実験室](ThinkingLab/ThinkingLab.ja.md)

</div>
