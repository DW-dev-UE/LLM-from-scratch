<div align="center">

# ベンチマークレポート — Base V1

<img src="https://img.shields.io/badge/checkpoint-sft__base__v1-blue?style=flat" alt="sft_base_v1" />
<img src="https://img.shields.io/badge/params-~327M-informational?style=flat" alt="params" />
<img src="https://img.shields.io/badge/status-early--stage-orange?style=flat" alt="early-stage" />

[한국어](BENCHMARK-v1.md) &nbsp;·&nbsp; [English](BENCHMARK-v1.en.md) &nbsp;·&nbsp; **[日本語](BENCHMARK-v1.ja.md)**

[← README.ja.md](README.ja.md) &nbsp;|&nbsp; 元JSON: [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json) · [`benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json)

</div>

---

> **一行まとめ**  
> `pretrain_base_v1` は採点全項目で **0/28** — チャット形式に一切従わない。  
> `sft_base_v1`（SFT 5,000 steps）は **指示に従おうとし始める** が、正答率はまだ低い。  
> **THINKING モードは全14問で空回答（0/14）** — 当面は `--no-thinking` を推奨する。

## 目次

1. [測定条件](#1-測定条件)
2. [チェックポイント](#2-チェックポイント)
3. [主要結果サマリー](#3-主要結果サマリー)
4. [Pretrain vs SFT](#4-pretrain-vs-sft)
5. [カテゴリ別詳細](#5-カテゴリ別詳細)
6. [コーディングベンチマーク](#6-コーディングベンチマーク)
7. [解釈と次のステップ](#7-解釈と次のステップ)
8. [元データ](#8-元データ)

---

## 1. 測定条件

| 項目 | 内容 |
|---|---|
| プロンプト数 | 14問 × 2モード（THINKING on / off）= **28生成** |
| カテゴリ | 韓国語QA・要約 · 日本語QA・翻訳 · 英語QA・要約 · Pythonコーディング5問 |
| QA / open 採点 | 0〜5点（トランスクリプトに基づく定性採点） |
| コーディング採点 | ユニットテスト実行（客観的 pass/fail） |
| 生成日時 | 2026-07-09 |
| 採点日時 | 2026-07-10 |

同一設問・同一ルーブリックで pretrain チェックポイントと SFT チェックポイントを比較した。

---

## 2. チェックポイント

| 名前 | パス | 学習 | 役割 |
|---|---|---|---|
| **pretrain_base_v1** | `ckpt/pretrain_base_v1.pt` | pretrain 18,000 steps、preset `base` | 純粋な next-token 事前学習 |
| **sft_base_v1** | `ckpt/sft_base_v1.pt` | SFT 5,000 steps（pretrain の上） | 対話・指示の微調整（**本レポートの主対象**） |

- パラメータ: 約 **326.7M**（`base` プリセット）
- トークナイザー: BPE vocab 64k（本学習版）
- メタ: [`llm/ckpt/SFT_base_v1/sft_base_v1.json`](llm/ckpt/SFT_base_v1/sft_base_v1.json)

---

## 3. 主要結果サマリー

### sft_base_v1 — THINKING モード

| 指標 | 値 |
|---|---|
| 回答生成成功 | **0 / 14（0%）** |
| 平均スコア | **0.0** |

> **致命的な問題**  
> すべての THINKING 生成で、モデルが `</THINKING>` を閉じる前に `<|eos|>` を出す。  
> `infer.py` はこの場合、出力全体を thinking バケットに入れ、**answer を空文字で返す**。  
> → このチェックポイントでチャットするときは **`--no-thinking`** を使う。

### sft_base_v1 — no-thinking モード

| 領域 | 平均 / 通過率 |
|---|---|
| 韓国語（3問） | **1.33 / 5** |
| 日本語（3問） | **0.33 / 5** |
| 英語（3問） | **2.67 / 5** |
| コーディングテスト | **4 / 17（23.5%）** |
| コーディング問題の完全通過 | **1 / 5**（`is_palindrome` のみ） |

英語の事実想起（Paris）が最も強く、韓日の事実・演算QAと指示追従（翻訳・要約）は弱い。コーディングはほぼ失敗し、回文判定1問だけが完全正解。

---

## 4. Pretrain vs SFT

| 指標 | pretrain_base_v1 | sft_base_v1 |
|---|---|---|
| 全体（28問） | **0 / 28** | no-think QA/open で非ゼロ多数 |
| コーディング（両モード） | **0 / 34** | no-think **4 / 17** |
| 指示追従 | 質問エコー / 繰り返し崩壊 | 形式は試す（内容はしばしば誤り） |
| 解釈 | チャットSFTなしの教科書的失敗 | わずか5k SFTでも明確なジャンプ |

SFTにより *「指示に全く従えない」→「従おうとするが内容は誤りが多い」* への変化は明らか。ただし実用品質にはまだ遠い。

---

## 5. カテゴリ別詳細

スコアは **no-thinking** 基準（THINKING は全問 score 0・空回答）。

### 5.1 韓国語

| ID | 種類 | スコア | キーワード | 要約 |
|---|---|---|---|---|
| `ko_fact_1` | QA | **2** | 서울 ✓ | 「ソウル」は出るが自己矛盾・循環論理 |
| `ko_math_1` | QA | **1** | 2 ✗ | 引き算未実行、誤答 |
| `ko_summary_1` | open | **1** | — | 要約ではなく原文をそのまま反復 |

### 5.2 日本語

| ID | 種類 | スコア | キーワード | 要約 |
|---|---|---|---|---|
| `ja_fact_1` | QA | **0** | 東京 ✗ | 首都ではなく人口の話、循環論理 |
| `ja_math_1` | QA | **1** | 5 ✗ | 質問の言い直しのみ、演算なし |
| `ja_translate_1` | open | **0** | — | 英訳指示を無視し日本語のみ出力 |

### 5.3 英語

| ID | 種類 | スコア | キーワード | 要約 |
|---|---|---|---|---|
| `en_fact_1` | QA | **4** | Paris ✓ | 核心は正解、その後地理幻覚・冗長 |
| `en_math_1` | QA | **1** | 120 ✗ | 誤った係数（×0.5）、120未到達 |
| `en_summary_1` | open | **3** | — | 要旨は掴むが4文 + 幻覚ディテール |

---

## 6. コーディングベンチマーク

プロンプトは関数シグネチャレベル（`is_prime(n)` など）。抽出コードをユニットテストで実行。

### no-thinking

| ID | 関数 | テスト | スコア | 備考 |
|---|---|---|---|---|
| `code_prime` | `is_prime` | 0/5 | 0 | 実質 `is_even` ロジック |
| `code_reverse` | `reverse_string` | 0/3 | 0 | `.lower()` 返却 + markdown fence 漏出で SyntaxError |
| `code_factorial` | `factorial` | 0/3 | 1 | 再帰形は合っているが `n==0` base 欠落 |
| `code_palindrome` | `is_palindrome` | **3/3** | **5** | コーディング全体で **唯一の完全通過** |
| `code_maxlist` | `find_max` | 1/3 | 1 | 変数シャドウ・比較誤り、1要素のみ偶然通過 |

### THINKING モード（コーディング）

5問すべて **0テスト · score 0** — `</THINKING>` 未終了でコード抽出不可。

---

## 7. 解釈と次のステップ

### うまくいった点

- Pretrain → SFT で **測定可能な能力ジャンプ** が確認できた
- 英語事実QA・一部の指示形式・回文コード1件は有効なシグナル
- 失敗モードはロードマップの初期検証段階と一致（回帰ではない）

### 詰まっている点

1. **THINKING 終了バグ** — 実用上のブロッカー（v2で大幅緩和、別レポート予定）
2. **演算・多言語事実** — 韓日が弱い
3. **厳密な指示遵守** — 要約・翻訳でエコー・指示無視
4. **コーディング** — 完全通過は1/5

### 推奨フォローアップ（ロードマップと整合）

- `</THINKING>` を閉じるパターンを強化したSFT（v2方向）
- コーディング・演算のSFT / RLVR 比重拡大
- DPO・校正SFTによる指示アライメント
- 同一14問セットでバージョン間比較を維持

---

## 8. 元データ

| ファイル | 内容 |
|---|---|
| [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json) | SFT v1 全項目・スコア・ノート |
| [`llm/ckpt/benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json) | Pretrain 同一設問比較 |
| [`llm/ckpt/SFT_base_v1/sft_base_v1.json`](llm/ckpt/SFT_base_v1/sft_base_v1.json) | 学習メタ（steps、lr、データハッシュ） |

本レポートは上記JSONを読みやすく整理したもの。数値・引用は元JSONに従う。

<div align="center">

---

[← README.ja.md](README.ja.md) &nbsp;|&nbsp; [ARCHITECTURE.ja.md](ARCHITECTURE.ja.md) &nbsp;|&nbsp; [POST-TRAINING.ja.md](POST-TRAINING.ja.md)

</div>
