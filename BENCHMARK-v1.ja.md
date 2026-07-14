[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.ja.md)

---

# 📊 BENCHMARK · base ~327M

> **学習日記**に近い文書です。きれいな順位表より、  
> *どう学習したか · 何を使ったか · 何と答えたか* を残します。

| | |
|:--|:--|
| 🆕 **最新** | `sft_base_v5` · 2026-07-14 |
| 📦 **モデル** | Decoder-only · 約 **326.7M** (`base`) |
| 🧪 **セット** | 同一 **14 問 × THINKING on/off**（バージョン比較用） |
| 📁 **原文** | [`ckpt/benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) |

> ⚠️ まだ **実使用段階ではありません。**  
> パイプラインが回るか、チェックポイントやプロトコルが変わると何が変わるかを見るスナップショットです。

### 📑 目次

1. [一目で](#1-一目で)
2. [学習の過程](#2-学習の過程)
3. [データセット](#3-データセット)
4. [採点方法](#4-採点方法)
5. [sft_base_v5 · 最新結果](#5-sft_base_v5--最新結果)
6. [以前のバージョン](#6-以前のバージョン)
7. [所感 · 次に](#7-所感--次に)

---

## 1. 一目で

太字の列が **いま見ている最新** です。

> ⚠️ **読み方の注意**  
> · **プロトコル**: v1–v4 は temp `0.7` 単一サンプル · **v5 は greedy (temp `0.0`)**  
> · **ベース分岐**: v1–v4 は `pretrain_base_v1` · **v5 は `pretrain_base_v2` 上の SFT**  
> · したがって **v5 > v4 > v3 を「純粋な SFT の階段」と読まないこと**  
> · コーディング比較はテスト分母の差があるため **完全通過 N/5** が安全です

| | Pretrain v1 | SFT v1 | SFT v2 | SFT v3 | SFT v4 | ⭐ **SFT v5** |
|:--|:-----------:|:------:|:------:|:------:|:------:|:------------:|
| 通常チャット平均 | ~0 | 低い† | 低い† | 2.57 | 2.07 | **3.21 / 5** |
| コーディング完全通過 (通常) | 0/5 | 1/5 | 1/5 | 4/5 | 3/5 | **5/5** |
| コーディング完全通過 (思考) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **0/5** |
| THINKING 非空回答 | — | **0/14** | **13/14** | 12/14 | 11/14 | **10/14** |
| THINKING 平均 | ~0 | 0.0 | 0.71 | 0.21 | 0.50 | **0.50** |
| 韓国語合計 (思考+通常) | 0/30 | — | 2/30 | 0/30 | 1/30 | **8/30** |
| ベース | v1 | v1 | v1 | v1 | v1 | **v2** |
| サンプリング | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | **0.0 greedy** |
| 印象 | チャット不可 | 形式を試す | 終了タグ復旧 | コーディング↑ · KO↓ | 横ばい · ノイズ | コーディング満点 · KO↑ |

† v1/v2 の通常チャットは言語別平均のみ:  
v1 KO **1.33** / JA **0.33** / EN **2.67** · v2 KO **0.33** / JA **2.0** / EN **1.0**

```text
                    ┌─► sft_v1 ─► sft_v2 ─► sft_v3 ─► sft_v4
pretrain_v1 ────────┤   形式      終了タグ    コーディング4/5  KO強化·横ばい
   18k · lr 6e-4    │
                    └─► pretrain_v2 (25k · lr 1e-4 · corpus v2)
                              │
                              └─► sft_v5 ★  (v4 と同じ sft.pt · greedy ベンチ)
                                    コーディング 5/5 · 通常 3.21 · KO 8/30
```

| Ver | うまくいったこと | 詰まっていること |
|:---:|:-----------------|:-----------------|
| v1 | 指示形式を追おうとする | THINKING の答が全部空 |
| v2 | `</THINKING>` 終了がほぼ復旧 | コーディング·品質はまだ弱い |
| v3 | コーディング **4/5**, 英語 fact/math 回復 | **韓国語崩壊**, THINKING コーディング 0/5 |
| v4 | KO SFT 比率↑ · thinking 内にソウル正解が出現 | ベンチ KO はほぼ回復せず · item flip |
| **v5** | **コーディング 5/5**, 通常 **3.21**, KO **8/30** | **thinking→answer ハンドオフ** · 思考コーディング 0/5 |

---

## 2. 学習の過程

### 🧩 モデルカード

| 項目 | 値 |
|:-----|:---|
| 系統 | Decoder-only Transformer (LLaMA スタイル) |
| パラメータ | 約 **326.7M** |
| 深さ · 幅 | 24 layer · `d_model` 1024 |
| Attention | GQA (Q 16 · KV 4) + RoPE |
| FFN | SwiGLU |
| 正規化 | RMSNorm (Pre-Norm) |
| その他 | weight tying · bias なし |

```text
入力トークン
   │
埋め込み ──────────────────────────────┐
   │                                 │ weight tying
[ Block × 24 ]
  RMSNorm → Attention → +
  RMSNorm → SwiGLU    → +
   │
RMSNorm
   │
LM Head ◄────────────────────────────┘
   │
次トークンの確率
```

### 🛤️ 学習の順序

```text
① トークナイザー (BPE · vocab 64k)
        ▼
② Pretrain v1 · 18k step · lr 6e-4
        ▼  pretrain_base_v1
        ├─► ③ SFT v1 · 5k · lr 3e-5 ──► sft_base_v1
        ├─► ④ SFT v2 · ~5.7k · THINKING 終了強化 ──► sft_base_v2
        ├─► ⑤ SFT v3 · 8k · コーディングデータ補強 ──► sft_base_v3
        ├─► ⑥ SFT v4 · 9.2k · 韓国語 SFT 補強 ──► sft_base_v4
        │
        └─► ②′ Pretrain v2 · 25k · lr 1e-4 · corpus v2
                ▼  pretrain_base_v2
                └─► ⑦ SFT v5 · 9.2k · (v4 と同じ sft.pt) ──► sft_base_v5 ★
```

| 段階 | チェックポイント | steps | init | データハッシュ · メモ |
|:-----|:-----------------|------:|:-----|:----------------------|
| Pretrain v1 | `pretrain_base_v1` | 18,000 | — | `53f815d47abc4887` · lr 6e-4 |
| SFT v1 | `sft_base_v1` | 5,000 | pretrain_v1 | `bbc2211091309d3c` |
| SFT v2 | `sft_base_v2` | 5,700 | pretrain_v1 | `fc177a36965df49a` · THINKING 終了 |
| SFT v3 | `sft_base_v3` | 8,000 | pretrain_v1 | `db63d09ddfa41388` · コーディング補強 |
| SFT v4 | `sft_base_v4` | 9,200 | pretrain_v1 | `975c1771bfff1919` · KO +70k |
| Pretrain v2 | `pretrain_base_v2` | 25,000 | pretrain_v1 | `4ad58fc7307962c0` · lr 1e-4 |
| ⭐ SFT v5 | `sft_base_v5` | 9,200 | **pretrain_v2** | **`975c1771bfff1919`** (v4 と同じ) |

共通: **AdamW** (β 0.9 / 0.95, wd 0.1) · grad clip 1.0 · warmup + cosine · CUDA · SFT lr `3e-5`

> 🔑 **v5 の主変数は新しい SFT ミックスではなくベース (pretrain_v2)** です。  
> v4 と v5 は同じ `sft.pt` (`975c…`) · 同じ 9,200 step · init 重みだけが違います。

### 📉 Loss の推移

<details>
<summary><b>Pretrain · SFT の loss 表</b></summary>

<br/>

**Pretrain v1**

| step | train | val |
|----:|------:|----:|
| 0 | 11.29 | — |
| 500 | 3.42 | 4.35 |
| 1,000 | 2.76 | 4.08 |
| 1,500 | 3.00 | 3.25 |
| 16,500 | | 2.72 |
| 17,500 | | 2.81 |
| 終盤 | ~2.1–2.5 | |

**SFT v1** (init = pretrain_v1)

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

**SFT v5** (init = pretrain_v2 · 同じ sft ミックス)

| step | train |
|----:|------:|
| 0 | **2.35** (v1 系 init ~2.8 帯より低い) |
| 500 | 1.32 |
| 1,000 | 1.40 |
| 終盤 | ~1.1–1.5 帯 |

</details>

> 💡 Loss が下がる = *対話形式に適応中* という信号です。  
> ベンチの点数がすぐ上がる意味では **ありません。**  
> ただし v5 の **低い SFT 初期 loss** は corpus-v2 continued pretrain が SFT 側に効いたヒントです。

---

## 3. データセット

元コーパス (数十 GB) はリポジトリにありません。  
公開 HuggingFace などから取り、`data.py` で前処理しました。

### 🔤 トークナイザー

| | |
|:--|:--|
| 方式 | Byte-level BPE |
| サイズ | **64,000** vocab |
| サンプル | fineweb, wiki(ko/ja), tinystories など ~900MB |
| 特殊トークン | 役割トークン · `THINKING` 区間トークン |

### 📚 Pretrain

**v1** · 約 19.1 億トークン · `train 1,893,646,391` · `val 19,127,741` · hash `53f815…`

<details>
<summary><b>🌍 自然言語 (EN / KO / JA) · v1 規模</b></summary>

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
<summary><b>💻 コード · v1</b></summary>

<br/>

| 内容 | 出典 | メモ |
|:-----|:-----|:-----|
| python, c, cpp, java, js, ts, go, rust, … | codeparrot/github-code-clean | 言語あたり ~1,200 ファイル |
| 関数単位 | Fsoft-AIC/the-vault-function | ~4 万 rows |
| コミット + diff | bigcode/commitpack | ~8 千 rows |

</details>

**v2** · v1 からの continued pretrain · **25,000 step · lr 1e-4** · data hash `4ad58f…`  
拡張·再トークン化した corpus v2 (fineweb_edu 拡大、finemath 追加など)。  
一部ソース (vault/commitpack) はダウンロード段階で失敗記録あり。

> 参考: ko-wiki val loss は v1 よりわずかに悪化 (報告: 2.504 → 2.605) したが、  
> SFT 初期 loss と下流ベンチは **v2 ベースの方が有利** でした。

### 🗣️ SFT ミックスの進化

<details>
<summary><b>v1 ミックス · 321,367 例</b></summary>

<br/>

| 領域 | 出典 | rows |
|:-----|:-----|----:|
| コード指示 | Magicoder-OSS-Instruct | 75,197 |
| 英語一般 | OpenHermes-2.5 | 60,000 |
| 英語マルチターン | UltraChat 200k | 35,000 |
| 日本語 | shisa / llm-jp / dolly-ja | 65,015 |
| 韓国語 | KoAlpaca v1.1a | 21,155 |
| 数学 CoT | OpenR1-Math | 25,000 |
| 思考 CoT | OpenThoughts3 | 40,000 |

```json
{
  "user": "2 + 3 * 4 = ?",
  "thinking": "掛け算が先。3*4=12、2+12=14。",
  "assistant": "答えは 14 です。"
}
```

user / system 区間は loss から除外します。**答え側**だけを学ばせる設定です。

</details>

<details>
<summary><b>v3 ミックス · 446,771 例 (コーディング·数学補強)</b></summary>

<br/>

v1 に追加 (倍数付き):

| 追加 | rows × 倍数 |
|:-----|------------:|
| gsm8k | 7,473 ×4 |
| mbpp | 474 ×8 |
| codealpaca_20k | 20,016 ×2 |
| orca_math_ko | 25,000 ×1 |
| jamard_gsm8k_ja | 6,672 ×4 |

→ コーディング完全通過 **1/5 → 4/5** (通常チャット)

</details>

<details>
<summary><b>v4 / v5 ミックス · 516,771 例 (韓国語補強) · hash `975c…`</b></summary>

<br/>

v3 に韓国語を追加:

| 追加 | rows |
|:-----|----:|
| kullm_v2 | 40,000 |
| ko_wikidata_qa | 30,000 |

韓国語比率 約 **10.3% → 22.5%**。  
**v4 と v5 はこの同一 `sft.pt` を共有**します。差は init ベースだけです。

</details>

### 比率の感覚 (v1 基準)

```text
Pretrain トークン
  英語ウェブ·ストーリー  ████████
  韓·日 wiki·ウェブ      ████████
  コード                 ██

SFT 例 (v1)
  英語対話               ██████████  ~30%
  コード指示             ████████    ~23%
  日本語                 ██████      ~20%
  思考·数学              ██████      ~20%
  韓国語                 ██          ~7%
```

---

## 4. 採点方法

| 項目 | 内容 |
|:-----|:-----|
| 設問 | **14 問 × 2 モード = 28 生成** (バージョンごとに同一) |
| モード | 🧠 THINKING オン / 💬 通常チャット |
| QA · 記述 | 0–5 点 (回答全文を採点 · Claude) |
| コーディング | 抽出コードをユニットテスト実行 |
| v1–v4 生成 | temp **`0.7`** · top_p `0.9` · max_new `256` · seed `0` · **1 サンプル** |
| **v5 生成** | temp **`0.0` greedy** · top_p `0.9` · max_new `256` · seed `0` |
| 日時 | v1 `07-09/10` · v2 `07-10` · v3 `07-13` · v4 `07-13` · **v5 `07-14`** (2026) |

> 📌 コーディングテスト分母: v1/v2 合計 **17** · v3+ 合計 **25** (問題あたり 5 ケース)。  
> バージョン比較は **完全通過 N/5** が安全です。

> 🎲 **サンプリングノイズ (v4 の教訓)**  
> temp 0.7 · 単一サンプルでは v3↔v4 の item flip が大きかった  
> (例: en_fact 5→0, code_prime 5→0, code_maxlist 5→0, code_reverse 0→5)。  
> **v5 から greedy** に切り替え、バージョン内ノイズを除去。  
> ただし v5 数値を v3/v4 と並べるときは **プロトコル差の注記が必須** です。

> 🇰🇷 韓国語合計: 思考+通常 各 3 問 × 0–5 = **最大 30 点**。

---

## 5. sft_base_v5 · 最新結果

| | |
|:--|:--|
| 🏁 チェックポイント | `ckpt/sft_base_v5.pt` |
| 🧱 ベース | `pretrain_base_v2` (25k · lr 1e-4 · corpus v2) |
| 📦 SFT データ | v4 と同じ `sft.pt` · `975c1771bfff1919` · 9,200 step |
| 🎲 生成 | **greedy · temp 0.0** (決定的) |
| 📁 JSON | [`benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) |

### 5.1 スコアカード

#### 💬 通常チャット (THINKING オフ) — 全体平均 **3.21 / 5**

| 領域 | スコア | 状態 |
|:-----|:------:|:----:|
| 🇰🇷 韓国語 | **1.67 / 5** (2 / 0 / 3) | 🟡 回復中 |
| 🇯🇵 日本語 | **1.33 / 5** (4 / 0 / 0) | 🟡 fact のみ |
| 🇺🇸 英語 | **3.67 / 5** (5 / 3 / 3) | 🟢 |
| 💻 コーディング | **25/25** テスト · **5/5** 完全通過 | 🟢 初の満点 |

#### 🧠 THINKING オン — 平均 **0.50 / 5**

| 指標 | 値 | 状態 |
|:-----|:--:|:----:|
| 平均スコア | **0.50 / 5** | 🔴 |
| 非空回答 | **10 / 14** | 🟡 (v4 11/14 · v3 12/14) |
| コーディング完全通過 | **0 / 5** | 🔴 散文のみ |
| 韓国語合計 (思考) | 3 / 15 | math のみ部分点 |

### 5.2 v4 → v5 ハイライト (プロトコル·ベース注意)

| | 変化 |
|:--|:-----|
| ✅ | コーディング完全通過 **3/5 → 5/5** (初の満点 · 決定的 greedy) |
| ✅ | 通常平均 **2.07 → 3.21** |
| ✅ | 韓国語合計 **1/30 → 8/30** (初の要約成功 · 思考 math に正解数字) |
| ✅ | SFT 初期 loss 改善シグナル (pretrain_v2 効果) |
| ⚠️ | **同じ SFT ミックス**、違うベース + greedy プロトコル → 「SFT だけが良くなった」ではない |
| ❌ | THINKING コーディングは依然 **コードなし 0/5** |
| ❌ | thinking 内に正解があっても **回答欄が空** (ko_fact, ja_fact, en_summary) |
| ❌ | 韓·日の算術回答はまだ崩れやすい |

---

### 5.3 質問と回答 · 通常チャット

スコアはすべて **sft_base_v5 · THINKING オフ · greedy** 基準です。

#### 🇰🇷 韓国語 — 平均 1.67

| 設問 | 点 | 期待 | モデルの出力 | コメント |
|:-----|:--:|:----:|:-------------|:---------|
| 首都 | **2** | ソウル | 「最大の都市はソウル」など · 首都明示が弱い | v3/v4(0) より改善 |
| りんご 5−3 | **0** | 2 | 「残りは 3 個」 | 依然誤答 |
| 要約 | **3** | 一文 | 「오늘 날씨가 매우 좋아서 공원에 산책을 나갔다.」 | **韓国語要約の初成功** 🟢 |

> 🧠 THINKING 首都: thinking = `대한민국의 수도는 서울특별시입니다.` **完璧** · 回答空 → 0 点。  
> 🧠 THINKING りんご: 回答に `2` が出現 (動詞誤り) → **3 点** · 韓国語 math 初の部分正解。

#### 🇯🇵 日本語 — 平均 1.33

| 設問 | 点 | 期待 | モデルの出力 | コメント |
|:-----|:--:|:----:|:-------------|:---------|
| 首都 | **4** | 東京 | 「首都は東京です」+ 循環反復 | 正解、やや減点 |
| りんご 7−2 | **0** | 5 | 「2個食べました」再陳述 | 正解 5 なし |
| 翻訳 | **0** | 英文 | 公園エッセイへ脱線 | 英語なし |

> 🧠 THINKING 首都: thinking に東京あり · 回答空。

#### 🇺🇸 英語 — 平均 3.67

| 設問 | 点 | 期待 | モデルの出力 | コメント |
|:-----|:--:|:----:|:-------------|:---------|
| 首都 | **5** | Paris | “The capital of France is Paris.” | 完璧 🟢 |
| 電車 60×2 | **3** | 120 | 先頭に 120 km · その後 reasoning が自己矛盾 | 部分 |
| 要約 | **3** | 一文 | 子供/公園中心 · 天気情報が欠落 | まずまず |

---

### 5.4 コーディング · 通常チャット

関数シグネチャだけを与え、抽出コードを **ユニットテストで実行** しました。  
🧠 THINKING モードのコーディング 5 問 → すべて散文のみ → **0/5**。

| 問題 | テスト | 点 | 一行 |
|:-----|:------:|:--:|:-----|
| `is_prime` | **5/5** | **5** | ✅ √n 試し割り |
| `reverse_string` | **5/5** | **5** | ✅ `s[::-1]` (top-level print なし) |
| `factorial` | **5/5** | **5** | ✅ base case 再帰 |
| `is_palindrome` | **5/5** | **5** | ✅ lower + 反転 (重複行 1 は無害) |
| `find_max` | **5/5** | **5** | ✅ 組み込み `max()` · 通過 |

<details>
<summary><b>✅ v5 通過コード</b></summary>

<br/>

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def reverse_string(s):
    return s[::-1]

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def is_palindrome(s):
    s = s.lower()
    s = s.lower()
    return s == s[::-1]

def find_max(lst):
    max_num = max(lst)
    return max_num
```

</details>

---

### 5.5 THINKING ハンドオフ — いま最も明確なボトルネック

thinking 実行 14 件中 4 件が **空回答** で終わり、そのうち 3 件は thinking 内に使える内容がありました。

| 設問 | thinking 内 | answer 欄 | 点 |
|:-----|:------------|:----------|:--:|
| ko_fact | `대한민국의 수도는 서울특별시입니다.` | (空) | 0 |
| ja_fact | 東京を含む | (空) | 0 |
| en_summary | まともな要約 | (空) | 0 |
| ko_summary | 原文コピー | (空) | 0 |

> 知識/要約の **取り出しはできるようになった** 一方、  
> **タグ後の回答欄へ写す段階** が残っています。  
> v1 の「閉じない」バグとは別のボトルネックです (閉じるが渡せない)。

---

## 6. 以前のバージョン

最新との比較用です。必要なときだけ開いてください。

<details>
<summary><b>📕 sft_base_v4 · 2026-07-13 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v4.json`](ckpt/benchmark_sft_base_v4.json)

韓国語 SFT 補強 (+kullm-v2 40k, +ko_wikidata_QA 30k · KO 比率 10.3%→22.5%)。  
全体としては **横ばい**。ベンチ韓国語はほとんど回復せず。

#### スコア

| モード | 結果 |
|:-------|:-----|
| 通常平均 | **2.07 / 5** |
| 🧠 THINKING 平均 | **0.50 / 5** · 非空 **11/14** |
| 🇰🇷 通常 | **0.33/5** (0 / 0 / 1) |
| 🇯🇵 通常 | **1.67/5** (5 / 0 / 0) |
| 🇺🇸 通常 | **2.67/5** (0 / 5 / 3) |
| 💻 通常 | 完全 **3/5** · テスト 15/25 |
| 🇰🇷 合計 (思考+通常) | **1/30** |

#### コーディング (通常 · temp 0.7)

| 問題 | テスト | メモ |
|:-----|:------:|:-----|
| is_prime | 0/5 | 本体空 → IndentationError — サンプリングノイズの可能性 |
| **reverse_string** | **5/5** | `s[::-1]` (v3 の print 副作用を修正) |
| factorial | 5/5 | base case 再帰 |
| is_palindrome | 5/5 | alnum フィルタ + lower · より堅牢 |
| find_max | 0/5 | 変数シャドーイングバグ |

#### ハイライト

| | |
|:--|:--|
| ✅ | thinking 内に初めて完璧な韓国語首都文 (`서울특별시`) — 回答欄は空 |
| ✅ | 思考モード初の math 満点: en_math **5/5** |
| ⚠️ | v3 比で item flip 大 → **temp 0.7 単一サンプルノイズ** が支配 |
| ❌ | 韓国語ベンチ 1/30 — SFT 比率だけでは不足 → pretrain レバーを推奨 |

</details>

<details>
<summary><b>📘 sft_base_v3 · 2026-07-13 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json)

コーディングデータ補強の目標達成。韓国語カテゴリ崩壊。

#### スコア

| モード | 結果 |
|:-------|:-----|
| 通常平均 | **2.57 / 5** |
| 🧠 THINKING 平均 | **0.21 / 5** · 非空 **12/14** |
| 🇰🇷 通常 | **0.00/5** |
| 🇯🇵 通常 | **1.33/5** |
| 🇺🇸 通常 | **4.00/5** |
| 💻 通常 | テスト **20/25** · 完全 **4/5** |
| 🇰🇷 合計 | **0/30** |

#### コーディング (通常)

| 問題 | テスト | メモ |
|:-----|:------:|:-----|
| is_prime | 5/5 | √n 試し割り |
| reverse_string | 0/5 | 本体 OK · top-level `print` → NameError |
| factorial | 5/5 | base case 再帰 (v2 無限再帰を修正) |
| is_palindrome | 5/5 | lower + 反転 (v2 常時 True を修正) |
| find_max | 5/5 | ループを手書き |

#### 通常チャット一目

| 設問 | 点 | メモ |
|:-----|:--:|:-----|
| ko_fact / math / summary | 0 / 0 / 0 | 韓国語崩壊 |
| ja_fact / math / translate | **4** / 0 / 0 | 東京正解 |
| en_fact / math / summary | **5** / **5** / 2 | Paris·120 回復 |
| coding 完全通過 | 4/5 | reverse のみ失敗 |

</details>

<details>
<summary><b>📗 sft_base_v2 · 2026-07-10 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json)

#### スコア

| モード | 結果 |
|:-------|:-----|
| 🧠 THINKING | 答 **13/14** · 平均 **0.71/5** |
| 🇰🇷 通常 | **0.33/5** |
| 🇯🇵 通常 | **2.00/5** |
| 🇺🇸 通常 | **1.00/5** |
| 💻 通常 | テスト **9/17** · 完全 **1/5** (`find_max` → 組み込み `max()`) |

#### v1 → v2

| | |
|:--|:--|
| ✅ | THINKING 終了バグがほぼ解決 (0/14 → 13/14) |
| ✅ | 初の思考モード満点: `ja_fact` **5/5** |
| ⚠️ | 英語 fact = Burgundy 幻覚 (temp 0.7 · 単一サンプル) |
| ❌ | コーディング本質的に弱い · thinking にコードなし |

#### コーディング (通常)

| 問題 | テスト | メモ |
|:-----|:------:|:-----|
| is_prime | 2/5 | 部分 |
| reverse_string | 2/3 | 部分 |
| factorial | 0/3 | 無限再帰など |
| is_palindrome | 2/3 | 常時 True 系 |
| **find_max** | **3/3** | `max()` 委譲 |

</details>

<details>
<summary><b>📙 sft_base_v1 · 2026-07-09/10 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json)

#### スコア

| モード | 結果 |
|:-------|:-----|
| 🧠 THINKING | 答 **0/14** · 平均 0.0 🔴 |
| 🇰🇷 通常 | **1.33/5** |
| 🇯🇵 通常 | **0.33/5** |
| 🇺🇸 通常 | **2.67/5** |
| 💻 通常 | テスト **4/17** · 完全 **1/5** |

> 🧠 閉じタグ前に生成が終わる → 推論コードが回答を空にする。  
> このチェックポイントは **通常チャットモード** で使う方がよいです。

#### Pretrain vs SFT v1

| | Pretrain | SFT v1 |
|:--|:---------|:-------|
| スコア | 事実上すべて 0 | 通常モードで点が出る |
| コーディング | 0/34 | 4/17 |
| 行動 | オウム返し · 反復 | 形式は試す (内容はよく間違う) |

SFT 5,000 step だけで **「全く従えない → 形式は試す」** へ一段上がります。

#### コーディング (通常)

| 問題 | テスト | 点 | メモ |
|:-----|:------:|:--:|:-----|
| is_prime | 0/5 | 0 | 偶数判定に変質 |
| reverse_string | 0/3 | 0 | `.lower()` のみ |
| factorial | 0/3 | 1 | 再帰の骨格、0 処理なし |
| **is_palindrome** | **3/3** | **5** | ✅ 唯一のきれいな満点 |
| find_max | 1/3 | 1 | 運の良い 1 ケース |

```python
# ✅ v1 通過
def is_palindrome(s):
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]

# ❌ v1 is_prime → 実質 is_even
def is_prime(n):
    if n <= 1:
        return False
    return n % 2 == 0
```

</details>

<details>
<summary><b>📓 pretrain_base_v1 · チャット ~0 点</b></summary>

<br/>

📁 [`benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json)

| | 結果 |
|:--|:-----|
| チャット反応 | ほぼなし · ~0 / 28 |
| コーディング | 0 / 34 (両モード合計) |
| 行動 | オウム返し · 同じ言葉の反復 · 質問を言い換えて返す |

SFT 前は **指示形式そのものが未学習** です。  
v1 SFT 後に「形式を試す」変化がベンチに初めて現れます。

> 📭 **pretrain_base_v2 単体のチャットベンチは未測定。**  
> v2 の効果は現在 **sft_base_v5** の下流だけで観測されています。

</details>

---

## 7. 所感 · 次に

### ✅ バージョンを通じてうまくいったこと

- Pretrain loss 11 → ~2、SFT ループ健全 · パイプライン検証 OK
- v1: 「形式を試す」がベンチに出る
- v2: THINKING **閉じ** がほぼ復旧 (0/14 → 13/14)
- v3: 通常チャット **コーディング 4/5** — データ補強目標達成
- v4: KO SFT 比率↑ · thinking 内に韓国語首都の正解一文が初出現 (伝達は失敗)
- **v5: 通常コーディング 5/5 · 平均 3.21 · KO 8/30** — 現状最良 · greedy で再現可能
- continued pretrain (v2) が **同じ SFT ミックス** でも下流を押し上げる

### 🚧 まだ詰まっていること (優先順)

1. 🧠 **thinking → answer ハンドオフ** — 取り出しはできるが回答欄が空
2. 🧠 THINKING モード **コーディング 0/5** — 依然散文のみ
3. 🇰🇷 韓国語 — 改善 (8/30) したが **直ったわけではない** (fact/math 不安定)
4. 韓·日の算術·指示追従がまだ不安定
5. 実使用品質まではまだ遠い

### 🧭 次に

| 優先 | 内容 |
|:----:|:-----|
| 1 | `</THINKING>` の後に **答えを強制生成** する SFT / 制約デコード |
| 2 | THINKING でも **コードを出す** フォーマット学習 (0/5 解消) |
| 3 | 韓国語 fact·算術の追加補強 (pretrain + SFT 併用) |
| 4 | 比較プロトコルを **greedy または multi-sample** に固定 |
| 5 | pretrain_v2 単体チャットベンチ追加 (ベース効果の分離) |
| 6 | DPO / 校正 SFT / RLVR |
| 7 | **同じ 14 問** でバージョン比較を維持 |

### 📐 解釈ガイド (もう一度)

```text
❌  「SFT を回すほど v5 > v4 > v3」
✅  「ベース分岐 + プロトコル変更を分けて読む」
    · v3→v4: 同じベース · 同じ temp0.7 · SFT ミックス変化 (横ばい + ノイズ)
    · v4→v5: 同じ SFT ミックス · 違うベース · greedy 切替 (最高点)
```

---

### 📁 原文ファイル

| パス | 内容 |
|:-----|:-----|
| [`ckpt/benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) | ⭐ **最新** SFT v5 (greedy · pretrain_v2) |
| [`ckpt/benchmark_sft_base_v4.json`](ckpt/benchmark_sft_base_v4.json) | SFT v4 |
| [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) | SFT v3 |
| [`ckpt/benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json) | SFT v2 |
| [`ckpt/benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json) | SFT v1 |
| [`ckpt/benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json) | Pretrain v1 |
| — | pretrain_v2 チャットベンチ **未測定** |

> 重み (`.pt`) と元コーパスは公開リポジトリに含めていません。

---

<div align="center">

[README](README.ja.md) · [ARCHITECTURE](ARCHITECTURE.ja.md) · [POST-TRAINING](POST-TRAINING.ja.md)

</div>
