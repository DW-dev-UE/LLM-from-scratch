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
> **次の目標 — APEX-1（1B 級）**
>
> 実測 **1,119.5M** パラメータ。モデル名 **APEX-1**。
>
> - **構造**: 24層 · d_model 2048 · GQA（16Q/4KV）+ RoPE（θ=500K）+ SwiGLU + RMSNorm · weight tying
> - **コンテキスト**: max 4096（学習は 2048）
> - **学習**: bf16 · lr 3e-4 コサイン · 51K step（20B トークン）· コーパス v3-en 80.8GB（英語+コード）· vocab 32K・英語専用
> - **見送り**: MoE · YaRN · FP8 · マルチGPU（1B 検証後の次スケール）

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
| [BENCHMARK v1](BENCHMARK-v1.ja.md) | Base モデルベンチ（学習過程 · Q&A 含む） |
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

いまベンチに載せた最新チェックポイントは **`sft_base_v6`**（base 約 327M）です。→ [§5 ベンチマーク一覧](#5-ベンチマーク一覧) · バージョン別詳細記録 [BENCHMARK v1](BENCHMARK-v1.ja.md)

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

[用語](GLOSSARY.ja.md) · [アーキテクチャ](ARCHITECTURE.ja.md) · [後続学習](POST-TRAINING.ja.md) · [ベンチマーク](BENCHMARK-v1.ja.md) · [思考実験室](ThinkingLab/ThinkingLab.ja.md)

</div>
