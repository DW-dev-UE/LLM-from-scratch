<div align="center">

# ベンチマークレポート · Base V1

**どう学習し、何のデータを使い、実際にどう答えたか**

<br/>

![ckpt](https://img.shields.io/badge/sft__base__v1-326.7M-3b82f6?style=flat-square)
![pretrain](https://img.shields.io/badge/pretrain-18k%20steps-22c55e?style=flat-square)
![sft](https://img.shields.io/badge/SFT-5k%20steps-22c55e?style=flat-square)
![status](https://img.shields.io/badge/status-early%20stage-f59e0b?style=flat-square)

<br/>

[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.ja.md)

</div>

---

### 一行で

| | 結果 |
|:--|:-----|
| **Pretrain のみ** | チャット形式にほぼ反応しない（約 0 / 28） |
| **SFT 後** | 形式は試す · 内容はしばしば誤り |
| **THINKING モード** | 回答が空（0 / 14）→ 通常チャットで評価 |

まだ実用品質ではありません。  
**パイプラインが動くこと**と、SFT が何を変えるかの最初のスナップショットです。

---

## 目次

1. [学習](#1-学習)  
2. [データセット](#2-データセット)  
3. [採点方法](#3-採点方法)  
4. [スコア要約](#4-スコア要約)  
5. [Pretrain と SFT](#5-pretrain-と-sft)  
6. [質問と回答](#6-質問と回答)  
7. [コーディング](#7-コーディング)  
8. [所感と次](#8-所感と次)

---

## 1. 学習

### モデル概要

| 項目 | 値 |
|:-----|:---|
| 系統 | Decoder-only Transformer（LLaMA 系部品） |
| パラメータ | 約 **326.7M**（`base`） |
| 深さ · 幅 | 24 layer · d_model 1024 |
| Attention | GQA（Q 16 · KV 4）+ RoPE |
| FFN | SwiGLU |
| 正規化 | RMSNorm（Pre-Norm） |
| その他 | weight tying、bias なし |

```text
tokens
   │
embedding ───────────────────────────┐
   │                                 │ (tied)
[ Block × 24 ]                       │
  RMSNorm → Attention → +            │
  RMSNorm → SwiGLU    → +            │
   │                                 │
RMSNorm                              │
   │                                 │
LM Head ◄────────────────────────────┘
   │
next-token logits
```

### 手順

```text
  ① BPE トークナイザー（vocab 64k）
           │
           ▼
  ② Pretrain · 18,000 step · lr 6e-4
           │
           ▼  pretrain_base_v1
  ③ SFT · 5,000 step · lr 3e-5
           │
           ▼  sft_base_v1
  ④ 本ベンチ
```

| 段 | チェックポイント | steps | 開始 | データ指紋 |
|:---|:-----------------|------:|:-----|:-----------|
| Pretrain | `pretrain_base_v1` | 18,000 | ゼロから | `train.bin` · `53f815d47abc4887` |
| SFT | `sft_base_v1` | 5,000 | pretrain から | `sft.pt` · `bbc2211091309d3c` |

共通: AdamW（β 0.9 / 0.95、wd 0.1）、grad clip 1.0、warmup + cosine、CUDA。

### Loss

**Pretrain**

| step | train | val |
|----:|------:|----:|
| 0 | 11.29 | |
| 500 | 3.42 | 4.35 |
| 1,000 | 2.76 | 4.08 |
| 1,500 | 3.00 | 3.25 |
| 16,500 | | 2.72 |
| 17,500 | | 2.81 |
| 終盤 | 約 2.1〜2.5 | |

**SFT**

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

Loss 低下 = 対話形式への適応の兆し。ベンチ正答とは別問題です。

---

## 2. データセット

生の巨大ファイルは GitHub に含めていません。  
公開 HF ソースを取得し `data.py` で前処理しました。

### トークナイザー

| | |
|:--|:--|
| 方式 | Byte-level BPE |
| サイズ | **64,000** vocab |
| サンプル | fineweb、wiki（ko/ja）、tinystories など約 900MB |
| 特殊 | 役割トークン、思考区間トークン |

### Pretrain · 約 19.1 億トークン

`train 1,893,646,391` · `val 19,127,741`

<details>
<summary><b>自然言語（EN / KO / JA）</b></summary>

<br/>

| 言語 | ファイル | 出典 | 規模 |
|:----:|:---------|:-----|:-----|
| EN | fineweb_edu | HuggingFaceFW/fineweb-edu | ~1.3GB |
| EN | tinystories | roneneldan/TinyStories | ~2.1GB |
| KO | fineweb2_ko | HuggingFaceFW/fineweb-2 | ~1.1GB |
| KO | wikipedia_ko | wikimedia 20231101.ko | ~1.3GB |
| JA | fineweb2_ja | HuggingFaceFW/fineweb-2 | ~1.1GB |
| JA | wikipedia_ja | wikimedia 20231101.ja | ~1.6GB |

</details>

<details>
<summary><b>コード</b></summary>

<br/>

| 内容 | 出典 | メモ |
|:-----|:-----|:-----|
| 11 言語 | codeparrot/github-code-clean | 各約 1,200 ファイル |
| 関数単位 | Fsoft-AIC/the-vault-function | 約 4 万 rows |
| コミット + diff | bigcode/commitpack | 約 8 千 rows |

</details>

<details>
<summary><b>SFT · 321,367 例</b></summary>

<br/>

| 領域 | 出典 | rows |
|:-----|:-----|----:|
| コード指示 | Magicoder-OSS-Instruct | 75,197 |
| 英語一般 | OpenHermes-2.5 | 60,000 |
| 英語マルチ | UltraChat 200k | 35,000 |
| 日本語 | shisa / llm-jp / dolly-ja 混合 | 65,015 |
| 韓国語 | KoAlpaca v1.1a | 21,155 |
| 数学 CoT | OpenR1-Math | 25,000 |
| 思考 CoT | OpenThoughts3 | 40,000 |

例:

```json
{
  "user": "2 + 3 * 4 = ?",
  "thinking": "掛け算が先。3*4=12、2+12=14。",
  "assistant": "答えは 14 です。"
}
```

user / system は loss から除外します。

</details>

### 比率の感覚

```text
Pretrain
  EN web · stories   ████████
  KO/JA wiki · web   ████████
  code               ██

SFT
  EN chat            ██████████  ~30%
  code instruct      ████████    ~23%
  日本語             ██████      ~20%
  思考 · 数学        ██████      ~20%
  韓国語             ██          ~7%
```

---

## 3. 採点方法

| 項目 | 内容 |
|:-----|:-----|
| 設問 | 14 × 2 モード = 28 生成 |
| モード | THINKING オン / オフ |
| QA · 記述 | 0〜5（全文を読む） |
| コード | ユニットテスト |
| 日時 | 生成 2026-07-09 · 採点 2026-07-10 |

---

## 4. スコア要約

### sft_base_v1

| モード | 結果 |
|:-------|:-----|
| THINKING **オン** | 回答 **0 / 14** · 平均 0.0 |
| 通常 · 韓国語 | 平均 **1.33 / 5** |
| 通常 · 日本語 | 平均 **0.33 / 5** |
| 通常 · 英語 | 平均 **2.67 / 5** |
| 通常 · コード | テスト **4 / 17** · 完全通過 **1 / 5** |

THINKING は閉じタグの前に生成が終わり、推論側が空回答を返します。  
**このチェックポイントは通常チャットで使うのが妥当です。**

---

## 5. Pretrain と SFT

| | Pretrain | SFT 後 |
|:--|:---------|:-------|
| 全体 | ほぼ全部 0 | 通常モードでスコア発生 |
| コード | 0 / 34 | 4 / 17（通常） |
| 振る舞い | エコー · 繰り返し | 形は試す（中身は誤り多い） |

SFT 5,000 step で  
**「全く従えない → 従おうとする」** へ一段上がります。  
その先の品質はまだ遠いです。

---

## 6. 質問と回答

スコアは **sft_base_v1 · 通常チャット** 基準です。

---

### 韓国語

#### 事実 · 首都

| | |
|:--|:--|
| **点** | 2 / 5 |
| **質問** | 대한민국의 수도는 어디인가요? |
| **期待** | ソウル |
| **回答** | 「서울」は出るが自己矛盾の長い文 |
| **メモ** | キーワードのみ |

THINKING: 回答欄が空。

---

#### 演算 · りんご

| | |
|:--|:--|
| **点** | 1 / 5 |
| **質問** | りんご 5 個、3 個食べた。残りは? |
| **期待** | 2 |
| **回答** | 3 個ある、と言う |
| **メモ** | 引き算なし |

---

#### 要約

| | |
|:--|:--|
| **点** | 1 / 5 |
| **質問** | 韓国語の公園文を一文で要約 |
| **回答** | 原文のほぼエコー |
| **メモ** | 指示無視 |

---

### 日本語

#### 事実 · 首都

| | |
|:--|:--|
| **点** | 0 / 5 |
| **質問** | 日本の首都はどこですか? |
| **期待** | 東京 |
| **回答** | 人口の話に逸れる |
| **メモ** | 的外れ |

---

#### 演算 · りんご

| | |
|:--|:--|
| **点** | 1 / 5 |
| **質問** | 太郎はりんごを7個…2個食べました。残りは? |
| **期待** | 5 |
| **回答** | 太郎は2個食べています。 |
| **メモ** | 言い直しのみ |

---

#### 翻訳

| | |
|:--|:--|
| **点** | 0 / 5 |
| **質問** | 日本語文を英語に翻訳して |
| **回答** | 日本語のまま |
| **メモ** | 英訳指示無視 |

---

### 英語

#### 事実 · 首都

| | |
|:--|:--|
| **点** | **4 / 5** |
| **質問** | What is the capital of France? |
| **期待** | Paris |
| **回答** | Paris は正しい。その後やや地理の脱線 |
| **メモ** | 今回いちばん良い事実シグナル |

Pretrain のみ: ドイツの首都の質問にすり替えてエコー → 0。

---

#### 演算 · 列車

| | |
|:--|:--|
| **点** | 1 / 5 |
| **質問** | 60 km/h で 2 時間、距離は? |
| **期待** | 120 |
| **回答** | 0.5 時間として 60 km |
| **メモ** | 計算ミス |

---

#### 要約

| | |
|:--|:--|
| **点** | 3 / 5 |
| **質問** | 公園の文を **one sentence** で要約 |
| **回答** | 4 文 + 存在しない swings / flowers |
| **メモ** | 雰囲気は近い、指示は失敗 |

---

## 7. コーディング

関数シグネチャ → 抽出コードをテスト。  
**通常チャットのみ**（THINKING は全滅）。

| 問題 | テスト | 点 | 一言 |
|:-----|:------:|:--:|:-----|
| is_prime | 0 / 5 | 0 | 偶数判定に変質 |
| reverse_string | 0 / 3 | 0 | lower だけ |
| factorial | 0 / 3 | 1 | 0 の base 欠落 |
| **is_palindrome** | **3 / 3** | **5** | **唯一の完答** |
| find_max | 1 / 3 | 1 | 変数シャドウ |

### 成功例

```python
def is_palindrome(s):
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]
```

### 失敗例（prime → even）

```python
def is_prime(n):
    if n <= 1:
        return False
    return n % 2 == 0
```

### 失敗例（reverse）

```python
def reverse_string(s):
    return s.lower()
```

---

## 8. 所感と次

### うまくいったこと

- Pretrain loss 11 → 約 2、SFT 2.67 → 約 1.05 → **学習は動いている**
- SFT で「答えようとする」段階が測定できる
- 英語事実 1 問 · 回文コード 1 問は明確な信号

### 詰まっていること

1. THINKING 終了タグ  
2. 韓日事実 · 算術の弱さ  
3. 要約 · 翻訳の指示無視  
4. コード完全通過 1 / 5  

### 次

- 閉じタグを強化した SFT（v2 方向）  
- コード · 算術の比重、必要なら RLVR  
- DPO · 校正 SFT  
- **同じ 14 問**でバージョン比較  

---

### ローカル元データ

| パス | 内容 |
|:-----|:-----|
| `llm/ckpt/benchmark_sft_base_v1.json` | SFT 全設問 |
| `llm/ckpt/benchmark_pretrain_base_v1.json` | Pretrain 比較 |
| `llm/DATA_SOURCES.md` | 出典一覧 |

重みと生コーパスは GitHub に含めていません。

---

<div align="center">

[README](README.ja.md) · [Architecture](ARCHITECTURE.ja.md) · [Post-training](POST-TRAINING.ja.md)

</div>
