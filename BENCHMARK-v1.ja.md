[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.ja.md)

---

# 📊 BENCHMARK · base ~327M

> **学習日記**に近い文書です。きれいな順位表より、  
> *どう学習したか · 何を使ったか · 何と答えたか* を残します。

| | |
|:--|:--|
| 🆕 **最新** | `sft_base_v6` · 2026-07-15 |
| 📦 **モデル** | Decoder-only · 約 **326.7M** (`base`) |
| 🧪 **セット** | 同一 **14 問 × THINKING on/off**（バージョン比較用） |
| 📁 **原文** | [`ckpt/benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) |

> ⚠️ まだ **実使用段階ではありません。**  
> パイプラインが回るか、チェックポイントやプロトコルが変わると何が変わるかを見るスナップショットです。

### 📑 目次

1. [一目で](#1-一目で)
2. [学習の過程](#2-学習の過程)
3. [データセット](#3-データセット)
4. [採点方法](#4-採点方法)
5. [sft_base_v6 · 最新結果](#5-sft_base_v6--最新結果)
6. [以前のバージョン](#6-以前のバージョン)
7. [所感 · 次に](#7-所感--次に)

---

## 1. 一目で

太字の列が **いま見ている最新** です。

> ⚠️ **読み方の注意**  
> · **プロトコル**: v1–v4 は temp `0.7` 単一サンプル · **v5·v6 は greedy (temp `0.0`)**  
> · **ベース分岐**: v1–v4 は `pretrain_base_v1` · **v5·v6 は `pretrain_base_v2` 上の SFT**  
> · **v5→v6**: ベース・SFT ソースミックスは同一、**prep_sft 前処理 + 推論予算分離のみ**変更 — このプロジェクトで最も清潔な単一変数比較  
> · したがって **v6 > v5 > v4 > v3 を「純粋な SFT の階段」と読まないこと**  
> · コーディング比較はテスト分母の差があるため **完全通過 N/5** が安全です

| | Pretrain v1 | SFT v1 | SFT v2 | SFT v3 | SFT v4 | SFT v5 | ⭐ **SFT v6** |
|:--|:-----------:|:------:|:------:|:------:|:------:|:------:|:------------:|
| 通常チャット平均 | ~0 | 低い† | 低い† | 2.57 | 2.07 | 3.21 | **3.93 / 5** |
| コーディング完全通過 (通常) | 0/5 | 1/5 | 1/5 | 4/5 | 3/5 | 5/5 | **5/5** |
| コーディング完全通過 (思考) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **4/5** (+prime 3/5) |
| THINKING 非空回答 | — | **0/14** | **13/14** | 12/14 | 11/14 | 10/14 | **14/14** |
| THINKING 平均 | ~0 | 0.0 | 0.71 | 0.21 | 0.50 | 0.50 | **2.93** |
| 韓国語合計 (思考+通常) | 0/30 | — | 2/30 | 0/30 | 1/30 | 8/30 | **10/30** |
| ベース | v1 | v1 | v1 | v1 | v1 | v2 | **v2** |
| サンプリング | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.0 greedy | **0.0 greedy** |
| 印象 | チャット不可 | 形式を試す | 終了タグ復旧 | コーディング↑ · KO↓ | 横ばい · ノイズ | コーディング満点 · KO↑ | ハンドオフ解決 · 思考コーディング開通 |

† v1/v2 の通常チャットは言語別平均のみ:  
v1 KO **1.33** / JA **0.33** / EN **2.67** · v2 KO **0.33** / JA **2.0** / EN **1.0**

```text
                    ┌─► sft_v1 ─► sft_v2 ─► sft_v3 ─► sft_v4
pretrain_v1 ────────┤   形式      終了タグ    コーディング4/5  KO強化·横ばい
   18k · lr 6e-4    │
                    └─► pretrain_v2 (25k · lr 1e-4 · corpus v2)
                              │
                              ├─► sft_v5    (v4 と同じ sft.pt · greedy ベンチ)
                              │               コーディング 5/5 · 通常 3.21 · KO 8/30
                              │
                              └─► sft_v6 ★  (同一ミックス · prep_sft 改善 + 推論予算分離)
                                              思考平均 0.50→2.93 · 思考コーディング 0/5→4/5
```

| Ver | うまくいったこと | 詰まっていること |
|:---:|:-----------------|:-----------------|
| v1 | 指示形式を追おうとする | THINKING の答が全部空 |
| v2 | `</THINKING>` 終了がほぼ復旧 | コーディング·品質はまだ弱い |
| v3 | コーディング **4/5**, 英語 fact/math 回復 | **韓国語崩壊**, THINKING コーディング 0/5 |
| v4 | KO SFT 比率↑ · thinking 内にソウル正解が出現 | ベンチ KO はほぼ回復せず · item flip |
| v5 | **コーディング 5/5**, 通常 **3.21**, KO **8/30** | **thinking→answer ハンドオフ** · 思考コーディング 0/5 |
| **v6** | **ハンドオフ解決**(14/14 非空) · 思考平均 **2.93** · 思考コーディング **4/5** | 韓国語算術は依然崩壊 · 反復/退化 · JA→EN 翻訳 |

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
                ├─► ⑦ SFT v5 · 9.2k · (v4 と同じ sft.pt) ──► sft_base_v5
                └─► ⑧ SFT v6 · 9.5k · prep_sft 改善 + 推論予算分離 ──► sft_base_v6 ★
```

| 段階 | チェックポイント | steps | init | データハッシュ · メモ |
|:-----|:-----------------|------:|:-----|:----------------------|
| Pretrain v1 | `pretrain_base_v1` | 18,000 | — | `53f815d47abc4887` · lr 6e-4 |
| SFT v1 | `sft_base_v1` | 5,000 | pretrain_v1 | `bbc2211091309d3c` |
| SFT v2 | `sft_base_v2` | 5,700 | pretrain_v1 | `fc177a36965df49a` · THINKING 終了 |
| SFT v3 | `sft_base_v3` | 8,000 | pretrain_v1 | `db63d09ddfa41388` · コーディング補強 |
| SFT v4 | `sft_base_v4` | 9,200 | pretrain_v1 | `975c1771bfff1919` · KO +70k |
| Pretrain v2 | `pretrain_base_v2` | 25,000 | pretrain_v1 | `4ad58fc7307962c0` · lr 1e-4 |
| SFT v5 | `sft_base_v5` | 9,200 | pretrain_v2 | `975c1771bfff1919` (v4 と同じ) |
| ⭐ SFT v6 | `sft_base_v6` | 9,500 | **pretrain_v2** | 同一 14 ファイルミックス · **`prep_sft` 改善**（thinking tail-preservation: trimmed 47,273 / demoted 17,540） |

共通: **AdamW** (β 0.9 / 0.95, wd 0.1) · grad clip 1.0 · warmup + cosine · CUDA · SFT lr `3e-5` · v6 は A100 40GB · batch 8 × accum 16

> 🔑 **v5 の主変数は新しい SFT ミックスではなくベース (pretrain_v2)** です。  
> v4 と v5 は同じ `sft.pt` (`975c…`) · 同じ 9,200 step · init 重みだけが違います。

> 🔑 **v6 の主変数はベースでも SFT ソースミックスでもなく、データ前処理と推論コード**です。  
> `data.py` の `prep_sft` は長い THINKING 区間の扱い方を変更しました（末尾保存優先で trim/demote）。  
> `infer.py` は thinking と answer が一つの `max_new` 予算を共有していたバグを修正し、**両者を分離した予算**で生成するようになりました。  
> ベース・SFT 元ファイルは v5 と完全に同一なので、v5→v6 のデルタは**この二つの修正だけの純粋な効果**として読めます。

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
| v5 生成 | temp **`0.0` greedy** · top_p `0.9` · max_new `256` · seed `0`（thinking·answer 予算共有） |
| **v6 生成** | temp **`0.0` greedy** · top_p `0.9` · max_new `256` · seed `0` · **thinking/answer 分離予算** |
| 日時 | v1 `07-09/10` · v2 `07-10` · v3 `07-13` · v4 `07-13` · v5 `07-14` · **v6 `07-15`** (2026) |

> 📌 コーディングテスト分母: v1/v2 合計 **17** · v3+ 合計 **25** (問題あたり 5 ケース)。  
> バージョン比較は **完全通過 N/5** が安全です。

> 🎲 **サンプリングノイズ (v4 の教訓)**  
> temp 0.7 · 単一サンプルでは v3↔v4 の item flip が大きかった  
> (例: en_fact 5→0, code_prime 5→0, code_maxlist 5→0, code_reverse 0→5)。  
> **v5 から greedy** に切り替え、バージョン内ノイズを除去。  
> ただし v5 数値を v3/v4 と並べるときは **プロトコル差の注記が必須** です。

> 🐛 **予算衝突バグ (v6 の教訓)**  
> v6 の最初のベンチ実行では `max_think` と `max_new` が同じ予算 (256) を共有し、  
> thinking が長くなると answer 予算が 0 になるバグがありました。  
> `generate()` で二つの予算を分離した後に再測定したのが下記の v6 の数値です。

> 🇰🇷 韓国語合計: 思考+通常 各 3 問 × 0–5 = **最大 30 点**。

---

## 5. sft_base_v6 · 最新結果

| | |
|:--|:--|
| 🏁 チェックポイント | `ckpt/sft_base_v6.pt` |
| 🧱 ベース | `pretrain_base_v2`（v5 と同じ） |
| 📦 SFT データ | v4·v5 と同じ 14 ファイルミックス · `prep_sft` 改善（thinking tail-preservation: trimmed 47,273 / demoted 17,540）· 9,500 step |
| 🔧 推論修正 | `infer.py` EOS ガード + thinking/answer **分離予算** |
| 🎲 生成 | greedy · temp 0.0 |
| 📁 JSON | [`benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) |

> ⚠️ 最初の v6 ベンチ実行では `max_think` と `max_new` が予算を共有し、回答が空になるバグがありました。  
> `generate()` で予算を分離した後に再測定したのが以下の結果です（§4 参照）。

### 5.1 スコアカード

#### 💬 通常チャット (THINKING オフ) — 全体平均 **3.93 / 5**

| 領域 | スコア | 状態 |
|:-----|:------:|:----:|
| 🇰🇷 韓国語 | **2.67 / 5** (5 / 0 / 3) | 🟢 fact 初の満点 |
| 🇯🇵 日本語 | **3.00 / 5** (5 / 4 / 0) | 🟢 fact+math |
| 🇺🇸 英語 | **4.33 / 5** (5 / 5 / 3) | 🟢 |
| 💻 コーディング | **25/25** テスト · **5/5** 完全通過 | 🟢 満点維持 |

#### 🧠 THINKING オン — 平均 **2.93 / 5**（v5 比 約 6 倍）

| 指標 | 値 | 状態 |
|:-----|:--:|:----:|
| 平均スコア | **2.93 / 5** | 🟢 (v5 0.50) |
| 非空回答 | **14 / 14** | 🟢 全問 (v5 10/14) |
| コーディング完全通過 | **4 / 5**（+prime 部分点 3/5） | 🟢 プロジェクト初 (v5 0/5) |
| 韓国語合計 (思考) | 2 / 15 | 🔴 依然弱い |

### 5.2 v5 → v6 ハイライト

| | 変化 |
|:--|:-----|
| ✅ | **THINKING ハンドオフ問題を解決** — 空回答 4/14 → **0/14**（全問で回答生成） |
| ✅ | THINKING 平均 **0.50 → 2.93**（約 6 倍） |
| ✅ | THINKING コーディング **0/5 → 4/5**（+prime 3/5 部分通過）— **6 バージン目にして初のコード出力** |
| ✅ | 通常チャット平均も連動して上昇 **3.21 → 3.93** |
| ✅ | 韓国語 fact **初の満点**（5点）:「대한민국의 수도는 서울특별시입니다.」 |
| ✅ | 日本語数学が **両モードとも初正解**（7 − 2 = 5） |
| ✅ | 韓国語合計 **8/30 → 10/30** |
| 🔬 | 同じベース · 同じ SFT ソースミックス（14 ファイル）· **prep_sft 前処理 + 推論予算分離だけが変化** — このプロジェクトで最も清潔な単一変数比較 |
| ❌ | 韓国語算術は依然崩壊（思考: 5−3=3、通常: 3+3=9） |
| ❌ | 長い thinking·回答終盤の反復/退化パターンが継続（例: en_fact の思考で “Wait, the capital of France is Paris.” の反復）— RL/DPO 領域の課題として持ち越し |
| ❌ | 日本語翻訳は両モードとも失敗 |

---

### 5.3 質問と回答 · 通常チャット（THINKING オフ）

#### 🇰🇷 韓国語 — 平均 2.67

| 設問 | 点 | 期待 | モデルの出力 | コメント |
|:-----|:--:|:----:|:-------------|:---------|
| 首都 | **5** | ソウル | 「대한민국의 수도는 서울특별시입니다.」 | **韓国語事実問題の初満点** 🟢 |
| りんご 5−3 | **0** | 2 | 「3(사과) + 3(사과) = 9개」 | 引き算が足し算に、算式崩壊 |
| 要約 | **3** | 一文 | 「오늘 날씨가 매우 좋아서 공원에 산책을 나갔다.」 | v5 と同水準を維持 |

#### 🇯🇵 日本語 — 平均 3.00

| 設問 | 点 | 期待 | モデルの出力 | コメント |
|:-----|:--:|:----:|:-------------|:---------|
| 首都 | **5** | 東京 | 「日本の首都は東京です。」 | 完璧 |
| りんご 7−2 | **4** | 5 | 「7 - 2 = 5個です」 | **日本語数学初の正解** · 動詞誤りで小幅減点 |
| 翻訳 | **0** | 英文 | 原文をそのまま反復 | 英語なし |

#### 🇺🇸 英語 — 平均 4.33

| 設問 | 点 | 期待 | モデルの出力 | コメント |
|:-----|:--:|:----:|:-------------|:---------|
| 首都 | **5** | Paris | “The capital of France is Paris.” | 完璧 |
| 電車 60×2 | **5** | 120 | 公式代入 → “120 km” | 完璧（v5 は部分点 3） |
| 要約 | **3** | 一文 | “The weather was nice, and the children were having a great time.” | 散歩/公園情報は脱落 |

---

### 5.4 質問と回答 · THINKING モード

v5 まではこの項目は「ハンドオフ失敗」の事例列挙でした。v6 は **回答欄が 14/14 全て埋まっています** — ただし推論そのものの正確さはまだ回答より低いです。

| 設問 | 点 | thinking 内 | answer 欄 | コメント |
|:-----|:--:|:------------|:----------|:---------|
| ko_fact | **1** | 「서울특별시입니다」と正解で始まるが「거리 1km」反復へ脱線 | 距離関連の雑文（ソウルというエンティティのみ存在） | 取り出しはできたが脱線 |
| ko_math | **0** | 「5개 − 3개 = 3개」（誤答） | 「사과를 3개」の無限反復 | 算式・回答とも崩壊 |
| ko_summary | **1** | 原文再陳述の後、想像フレーズを反復 | 「아이들이 뛰어놀고 있는 모습을 상상해 보세요.」 | 要約ではなく命令文 |
| ja_fact | **5** | 混乱した推論 | 「答えは東京です。」 | **明快な正解 — ハンドオフ成功の典型例** |
| ja_math | **4** | 算式崩壊 (1.7, 5.7 など) | 「答えは5です。」 | 推論は誤りだが答えは正解 |
| ja_translate | **0** | 英語文の反復（誤訳） | 引用符の羅列のみ | 失敗 |
| en_fact | **4** | “Wait, the capital of France is Paris.” の反復ループ | “The capital of France is **Paris**.” + 反復する説明 | 正解明示、冗長さで減点 |
| en_math | **0** | “60/2=30”, “30/2=15” | “The answer is 15.” | 算式崩壊、誤答 |
| en_summary | **3** | 短く無難な要約 | “The answer is that the park was nice and the children were playing.” | 要約は成立、フレームがぎこちない |

> 🔑 **最も重要な変化**: v5 では ko_fact·ja_fact·en_summary で thinking 内に正解があっても回答欄が空でした。  
> v6 は同種の設問で **回答欄に内容が入ります** — 精度はまだばらつきますが、「伝わらない」こと自体は解消されました。

---

### 5.5 コーディング · 両モード比較

通常チャットは v5 に続き **5/5 満点** を維持します。THINKING モードは今回が **プロジェクト初めて実際にコードを出力**したバージョンです。

| 問題 | 通常チャット | THINKING | コメント |
|:-----|:-----------:|:--------:|:---------|
| `is_prime` | 5/5 | **3/5** | 思考モード: 剰余演算を 17 までハードコード列挙 — 一般化できず部分通過 |
| `reverse_string` | 5/5 | **5/5** | 思考モードでも `s[::-1]` がそのまま通過（thinking は空欄） |
| `factorial` | 5/5 | **5/5** | 再帰 base case、両モード同一 |
| `is_palindrome` | 5/5 | **5/5** | 思考モードは `s.strip()`、通常は `s.lower()` — どちらも通過 |
| `find_max` | 5/5 | **5/5** | 両モードとも組み込み `max()` を活用 |

<details>
<summary><b>✅ THINKING モード通過コード</b></summary>

<br/>

```python
def is_prime(n):  # 3/5 部分通過 — 17 までの剰余演算をハードコード
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    # ... n % 17 == 0 まで繰り返す（一般化された √n 分解ではない）

def reverse_string(s):
    return s[::-1]

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def is_palindrome(s):
    s = s.strip()
    return s == s[::-1]

def find_max(lst):
    max_num = max(lst)
    for num in lst:
        if num > max_num:
            max_num = num
    return max_num
```

</details>

---

### 5.6 残るボトルネック

thinking→answer ハンドオフが解決し、残る問題は **形式ではなく精度** として明確になりました。

| 問題 | 例 | 性質 |
|:-----|:-----|:-----|
| 🇰🇷 韓国語算術 | 5−3 を「3+3=9」（通常）/「5개−3개=3개」（思考）と計算 | 知識・演算エラー — データ補強が必要 |
| 🔁 反復·退化 | en_fact の思考で同じ文を反復、ko_fact の思考で「1km」を反復 | デコーディング品質 — RL/DPO 領域 |
| 🇯🇵→🇬🇧 翻訳 | 原文反復または無関係な英語文の羅列 | 翻訳能力そのものが未形成 |
| コーディング一般化 | 思考モードの `is_prime` が剰余演算 17 個をハードコード（分解ロジックではない） | 部分的理解 — 完全なアルゴリズム学習はまだ |

> 📌 v1 の「閉じない」→ v5 の「閉じるが伝わらない」→ v6 は **「伝わるが内容が時々間違う」** 段階へボトルネックが移動しました。  
> 形式の学習は事実上ほぼ完了段階と見られます。

---

## 6. 以前のバージョン

最新との比較用です。必要なときだけ開いてください。

<details>
<summary><b>📔 sft_base_v5 · 2026-07-14 · pretrain_v2 · greedy</b></summary>

<br/>

📁 [`benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json)

初の greedy（決定的）ベンチ。ベースを `pretrain_base_v2` に交換（SFT ミックスは v4 と同一）。
通常チャットのコーディング初満点、韓国語が回復し始める。THINKING ハンドオフ（タグ後の回答への転写失敗）が新たなボトルネックとして浮上（v6 で解決）。

#### スコア

| モード | 結果 |
|:-------|:-----|
| 通常平均 | **3.21 / 5** |
| 🧠 THINKING 平均 | **0.50 / 5** · 非空 **10/14** |
| 🇰🇷 通常 | **1.67/5** (2 / 0 / 3) |
| 🇯🇵 通常 | **1.33/5** (4 / 0 / 0) |
| 🇺🇸 通常 | **3.67/5** (5 / 3 / 3) |
| 💻 通常 | 完全 **5/5** · テスト 25/25（初満点） |
| 🇰🇷 合計 (思考+通常) | **8/30** |

#### コーディング（通常 · greedy）

| 問題 | テスト | メモ |
|:-----|:------:|:-----|
| is_prime | 5/5 | √n 試し割り |
| reverse_string | 5/5 | `s[::-1]` |
| factorial | 5/5 | base case 再帰 |
| is_palindrome | 5/5 | lower + 反転 |
| find_max | 5/5 | 組み込み `max()` |

#### ハイライト

| | |
|:--|:--|
| ✅ | コーディング完全通過 3/5 → 5/5（初満点 · 決定的 greedy） |
| ✅ | 韓国語合計 1/30 → 8/30（初の要約成功 · 思考 math に正解数字） |
| ⚠️ | 同じ SFT ミックス、違うベース + greedy 切替 → 「SFT だけが良くなった」わけではない |
| ❌ | THINKING コーディングは依然 0/5 · thinking 内の正解が回答欄へ伝わらない（ハンドオフのボトルネックを発見） |

</details>

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
> v2 の効果は現在 **sft_base_v5 · sft_base_v6** の下流だけで観測されています。

</details>

---

## 7. 所感 · 次に

### ✅ バージョンを通じてうまくいったこと

- Pretrain loss 11 → ~2、SFT ループ健全 · パイプライン検証 OK
- v1: 「形式を試す」がベンチに出る
- v2: THINKING **閉じ** がほぼ復旧 (0/14 → 13/14)
- v3: 通常チャット **コーディング 4/5** — データ補強目標達成
- v4: KO SFT 比率↑ · thinking 内に韓国語首都の正解一文が初出現 (伝達は失敗)
- v5: 通常コーディング 5/5 · 平均 3.21 · KO 8/30 · greedy で再現可能
- **v6: THINKING ハンドオフ解決 · 思考平均 0.50→2.93 · 思考コーディング初通過(4/5) · 通常平均も 3.21→3.93 と連動上昇** — 現状最良
- continued pretrain (v2) が **同じ SFT ミックス** でも下流を押し上げる
- prep_sft 前処理 + 推論予算分離だけでハンドオフ問題を解消（ベース・SFT ソースはそのまま）

### 🚧 まだ詰まっていること (優先順)

> ✅ v5 まで最優先だった「THINKING ハンドオフ」と「THINKING コーディングのフォーマット学習」は **v6 で解決**しました（prep_sft 前処理 + 推論予算分離）。以下はその次の課題です。

1. 🇰🇷 韓国語算術·事実の精度 — 形式はできるが計算がよく間違う (思考: 5−3=3、通常: 3+3=9)
2. 🔁 長い生成での反復·退化パターン (thinking·回答とも) — RL/DPO へ持ち越す領域
3. 🇯🇵→🇬🇧 翻訳能力が未形成 — 原文反復のみ
4. コーディング一般化 — 思考モードの `is_prime` が剰余演算のハードコード列挙 (√n 分解ではない)
5. 実使用品質まではまだ遠い

### 🧭 次に

| 優先 | 内容 |
|:----:|:-----|
| 1 | 韓国語 fact·算術の追加補強 (pretrain + SFT 併用) — 唯一未解決の「精度」軸 |
| 2 | 反復·退化の抑制 — DPO / 校正 SFT / RLVR (形式はできたので次は品質を磨く番) |
| 3 | 日本語→英語の翻訳データ補強 |
| 4 | コーディング一般化 — THINKING モードでも √n 分解のような一般ロジックを学習させる |
| 5 | pretrain_v2 単体チャットベンチ追加 (ベース効果の分離、依然未測定) |
| 6 | **同じ 14 問** でバージョン比較を維持 |

### 📐 解釈ガイド (もう一度)

```text
❌  「SFT を回すほど v6 > v5 > v4 > v3」
✅  「ベース分岐 + プロトコル変更 + 前処理変更を分けて読む」
    · v3→v4: 同じベース · 同じ temp0.7 · SFT ミックス変化 (横ばい + ノイズ)
    · v4→v5: 同じ SFT ミックス · 違うベース · greedy 切替 (最高点)
    · v5→v6: 同じベース · 同じ SFT ミックス · prep_sft 改善+推論予算分離 (ハンドオフ解決)
```

---

### 📁 原文ファイル

| パス | 内容 |
|:-----|:-----|
| [`ckpt/benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) | ⭐ **最新** SFT v6 (ハンドオフ解決 · greedy · pretrain_v2) |
| [`ckpt/benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) | SFT v5 (greedy · pretrain_v2) |
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
