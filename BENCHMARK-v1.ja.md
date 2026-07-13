[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.ja.md)

---

# 📊 BENCHMARK · base ~327M

> **学習日記**に近い文書です。きれいな順位表より、  
> *どう学習したか · 何を使ったか · 何と答えたか* を残します。

| | |
|:--|:--|
| 🆕 **最新** | `sft_base_v3` · 2026-07-13 |
| 📦 **モデル** | Decoder-only · 約 **326.7M** (`base`) |
| 🧪 **セット** | 同一 **14 問 × THINKING on/off**（バージョン比較用） |
| 📁 **原文** | [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) |

> ⚠️ まだ **実使用段階ではありません。**  
> パイプラインが回るか、SFT を繰り返すと何が変わるかを見るスナップショットです。

### 📑 目次

1. [一目で](#1-一目で)
2. [学習の過程](#2-学習の過程)
3. [データセット](#3-データセット)
4. [採点方法](#4-採点方法)
5. [sft_base_v3 · 最新結果](#5-sft_base_v3--最新結果)
6. [以前のバージョン](#6-以前のバージョン)
7. [所感 · 次に](#7-所感--次に)

---

## 1. 一目で

太字の列が **いま見ている最新** です。

| | Pretrain | SFT v1 | SFT v2 | ⭐ **SFT v3** |
|:--|:--------:|:------:|:------:|:------------:|
| 通常チャット感 | ~0 | 低い | 低い | **2.57 / 5** |
| コーディング完全通過 | 0/5 | 1/5 | 1/5 | **4/5** |
| THINKING 回答生成 | — | **0/14** | **13/14** | **12/14** |
| 印象 | チャット不可 | 形式を試す | 終了タグ復旧 | コーディング↑ · 韓国語↓ |

```text
pretrain ──► sft_v1 ──► sft_v2 ──► sft_v3 ★
  チャット0     形式を試す   THINKING終了     コーディング4/5
```

| Ver | うまくいったこと | 詰まっていること |
|:---:|:-----------------|:-----------------|
| v1 | 指示形式を追おうとする | THINKING の答が全部空 |
| v2 | `</THINKING>` 終了がほぼ復旧 | コーディング·品質はまだ弱い |
| **v3** | **コーディング 4/5**, 英語 fact/math 回復 | **韓国語崩壊**, THINKING コーディング 0/5 |

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
② Pretrain · 18k step · lr 6e-4
        ▼  pretrain_base_v1
③ SFT v1 · 5k step · lr 3e-5
        ▼  sft_base_v1
④ SFT v2 · THINKING 終了パターン強化
        ▼  sft_base_v2
⑤ SFT v3 · コーディングデータ補強
        ▼  sft_base_v3  ★ 本文
```

| 段階 | チェックポイント | steps | メモ |
|:-----|:-----------------|------:|:-----|
| Pretrain | `pretrain_base_v1` | 18,000 | ゼロから · `53f815d47abc4887` |
| SFT v1 | `sft_base_v1` | 5,000 | pretrain から · `bbc2211091309d3c` |
| SFT v2 | `sft_base_v2` | ~5.6k–5.7k | 目標: THINKING 終了復旧 |
| ⭐ SFT v3 | `sft_base_v3` | データ補強 SFT | コーディング完全通過 **4/5** |

共通: **AdamW** (β 0.9 / 0.95, wd 0.1) · grad clip 1.0 · warmup + cosine · CUDA

### 📉 Loss の推移 (base · v1 時点)

<details>
<summary><b>Pretrain · SFT v1 の loss 表</b></summary>

<br/>

**Pretrain**

| step | train | val |
|----:|------:|----:|
| 0 | 11.29 | — |
| 500 | 3.42 | 4.35 |
| 1,000 | 2.76 | 4.08 |
| 1,500 | 3.00 | 3.25 |
| 16,500 | | 2.72 |
| 17,500 | | 2.81 |
| 終盤 | ~2.1–2.5 | |

**SFT v1**

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

</details>

> 💡 Loss が下がる = *対話形式に適応中* という信号です。  
> ベンチの点数がすぐ上がる意味では **ありません。**

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

### 📚 Pretrain · 約 19.1 億トークン

`train 1,893,646,391` · `val 19,127,741`

<details>
<summary><b>🌍 自然言語 (EN / KO / JA)</b></summary>

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
<summary><b>💻 コード</b></summary>

<br/>

| 内容 | 出典 | メモ |
|:-----|:-----|:-----|
| python, c, cpp, java, js, ts, go, rust, … | codeparrot/github-code-clean | 言語あたり ~1,200 ファイル |
| 関数単位 | Fsoft-AIC/the-vault-function | ~4 万 rows |
| コミット + diff | bigcode/commitpack | ~8 千 rows |

</details>

<details>
<summary><b>🗣️ SFT · 321,367 例 (v1 ミックス)</b></summary>

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
  "assistant": "答えは14です。"
}
```

user / system 区間は loss から外します。**答側** だけを学ぶためです。

v2·v3 はこのミックスを引き継ぎ、  
→ v2: THINKING 終了パターン · v3: コーディング品質 を補強しました。

</details>

### 比率の感覚 (v1)

```text
Pretrain トークン
  英語 web・ストーリー  ████████
  韓・日 wiki・web     ████████
  コード               ██

SFT 例
  英語対話             ██████████  ~30%
  コード指示           ████████    ~23%
  日本語               ██████      ~20%
  思考・数学           ██████      ~20%
  韓国語               ██          ~7%
```

---

## 4. 採点方法

| 項目 | 内容 |
|:-----|:-----|
| 設問 | **14 問 × 2 モード = 28 生成**（バージョンごとに同一） |
| モード | 🧠 THINKING オン / 💬 通常チャット |
| QA · 記述 | 0–5 点（回答全文を採点） |
| コーディング | ユニットテスト実行の通過数 |
| v3 生成 | temp `0.7` · top_p `0.9` · max_new `256` · seed `0` · 1 サンプル |
| 日時 | v1 `07-09/10` · v2 `07-10` · **v3 `07-13`** (2026) |

> 📌 コーディング分母: v1/v2 合計 **17** · v3 合計 **25**（問題あたり 5 ケース）。  
> バージョン比較は **完全通過 N/5** を見る方が安全です。

---

## 5. sft_base_v3 · 最新結果

| | |
|:--|:--|
| 🏁 チェックポイント | `ckpt/sft_base_v3.pt` |
| 📁 JSON | [`benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) |

### 5.1 スコアカード

#### 💬 通常チャット (THINKING オフ) — 全体平均 **2.57 / 5**

| 領域 | スコア | 状態 |
|:-----|:----:|:----:|
| 🇰🇷 韓国語 | **0.00 / 5** | 🔴 |
| 🇯🇵 日本語 | **1.33 / 5** | 🟡 |
| 🇺🇸 英語 | **4.00 / 5** | 🟢 |
| 💻 コーディング | **20/25** テスト · **4/5** 完全通過 | 🟢 |

#### 🧠 THINKING オン

| 指標 | 値 | 状態 |
|:-----|:--:|:----:|
| 平均スコア | **0.21 / 5** | 🔴 |
| 非空回答 | **12 / 14** | 🟢（終了はおおむね OK） |
| コーディング完全通過 | **0 / 5** | 🔴 散文のみ |

### 5.2 v2 → v3 ハイライト

| | 変化 |
|:--|:-----|
| ✅ | コーディング完全通過 **0/5 → 4/5**（`is_prime`, `factorial`, `is_palindrome`, `find_max`） |
| ✅ | 英語 Paris **5/5**, 120 km **5/5**（v2 の Burgundy 幻覚から回復） |
| ⚠️ | `reverse_string` 本体 `s[::-1]` は正解 · top-level `print` → NameError → **0/5** |
| ❌ | 韓国語 6 問（思考+通常）**すべて 0 点** — カテゴリ崩壊 |
| ❌ | THINKING コーディングは依然 **コードなし 0/5** |
| 🔍 | 途中で正しい式 (7−2=5) が出て最終が外れるパターン増 |

---

### 5.3 質問と回答 · 通常チャット

スコアはすべて **sft_base_v3 · THINKING オフ** 基準です。

#### 🇰🇷 韓国語 — 平均 0.00

| 設問 | スコア | 期待 | モデルがしたこと | コメント |
|:-----|:----:|:----:|:-----------------|:---------|
| 首都 | **0** | ソウル | GDP·面積の幻覚、「서울」なし | v1(2 点)より退歩 |
| りんご 5−3 | **0** | 2 | 「3 個食べた」言い直し | 引き算なし |
| 要約 | **0** | 一文 | 同じ句の無限ループ | degenerate |

> 🧠 THINKING 首都: thinking 内では「서울」まで到達するが、タグ後の **答が空**。

#### 🇯🇵 日本語 — 平均 1.33

| 設問 | スコア | 期待 | モデルがしたこと | コメント |
|:-----|:----:|:----:|:-----------------|:---------|
| 首都 | **4** | 東京 | 「首都は東京です」+ ランドマーク重複 | 正解、やや減点 |
| りんご 7−2 | **0** | 5 | `7−8=2` 式崩壊 | 正解なし |
| 翻訳 | **0** | 英訳 | “Hopper's Park is a park for kids.” | 無関係 |

> 🧠 THINKING: 首都は thinking に東京があるが答欄空（v2 思考モード 5 点から退歩）。  
> 演算は途中 7−2=5 のあと ×7=35 へ脱線。

#### 🇺🇸 英語 — 平均 4.00

| 設問 | スコア | 期待 | モデルがしたこと | コメント |
|:-----|:----:|:----:|:-----------------|:---------|
| 首都 | **5** | Paris | “The capital of France is Paris.” | v2 幻覚から回復 🟢 |
| 列車 60×2 | **5** | 120 | Distance = Speed × Time → 120 km | 模範解答 🟢 |
| 要約 | **2** | 一文 | “The weather was nice today.” | 情報が大きく欠落 |

---

### 5.4 コーディング · 通常チャット

関数シグネチャだけ渡し、取り出したコードを **ユニットテストで実行**。  
🧠 THINKING モードのコーディング 5 問 → すべて散文のみ → **0/5**。

| 問題 | テスト | スコア | 一行 |
|:-----|:------:|:----:|:-----|
| `is_prime` | **5/5** | **5** | ✅ √n 試し割り |
| `reverse_string` | 0/5 | 0 | ⚠️ 本体 OK · `print` NameError |
| `factorial` | **5/5** | **5** | ✅ base case 再帰（v2 無限再帰を修正） |
| `is_palindrome` | **5/5** | **5** | ✅ lower + 反転（常に True を修正） |
| `find_max` | **5/5** | **5** | ✅ ループ直書き（builtin `max` 依存から脱却） |

<details>
<summary><b>✅ 通過したコード</b></summary>

<br/>

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]

def find_max(lst):
    max_num = lst[0]
    for num in lst:
        if num > max_num:
            max_num = num
    return max_num
```

</details>

<details>
<summary><b>⚠️ 惜しい失敗 · reverse_string</b></summary>

<br/>

```python
def reverse_string(s):
    return s[::-1]

print(reverse_string(s))  # NameError: s is not defined → 実行 0/5
```

関数自体は正しいが、実行ハーネスを壊す top-level `print` で 0 点になります。

</details>

---

## 6. 以前のバージョン

比較用アーカイブです。必要なときだけ展開してください。

<details>
<summary><b>📘 sft_base_v2 · 2026-07-10</b></summary>

<br/>

📁 [`benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json)

#### スコア

| モード | 結果 |
|:-----|:-----|
| 🧠 THINKING | 回答 **13/14** · 平均 **0.71/5** |
| 🇰🇷 通常 | **0.33/5** |
| 🇯🇵 通常 | **2.00/5** |
| 🇺🇸 通常 | **1.00/5** |
| 💻 通常 | テスト **9/17** · 完全 **1/5**（`find_max` → 組み込み `max()`） |

#### v1 → v2

| | |
|:--|:--|
| ✅ | THINKING 終了バグの大部分を解決（0/14 → 13/14） |
| ✅ | 思考モード初の満点: `ja_fact` **5/5** |
| ⚠️ | 英語 fact = Burgundy 幻覚（temp 0.7 · 単一サンプル） |
| ❌ | コーディングは本質的に弱い · thinking でコードなし |

#### 通常チャット一覧

| 設問 | スコア | メモ |
|:-----|:----:|:-----|
| ko_fact / math / summary | 0 / 0 / 1 | 韓国語が弱い |
| ja_fact / math / translate | **5** / 1 / 0 | 東京正解 |
| en_fact / math / summary | 0 / 2 / 1 | Burgundy |
| coding 完全通過 | 1/5 | `find_max` only |

#### コーディング（通常）

| 問題 | テスト | メモ |
|:-----|:------:|:-----|
| is_prime | 2/5 | 部分 |
| reverse_string | 2/3 | 部分 |
| factorial | 0/3 | 無限再帰など |
| is_palindrome | 2/3 | 常に True 系 |
| **find_max** | **3/3** | `max()` に委譲 |

</details>

<details>
<summary><b>📗 sft_base_v1 · 2026-07-09/10</b></summary>

<br/>

📁 [`benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json)

#### スコア

| モード | 結果 |
|:-----|:-----|
| 🧠 THINKING | 回答 **0/14** · 平均 0.0 🔴 |
| 🇰🇷 通常 | **1.33/5** |
| 🇯🇵 通常 | **0.33/5** |
| 🇺🇸 通常 | **2.67/5** |
| 💻 通常 | テスト **4/17** · 完全 **1/5** |

> 🧠 閉じタグの前に生成が終わる → 推論が答を空にする。  
> このチェックポイントは **通常チャットモード** で使う方が正しいです。

#### Pretrain vs SFT v1

| | Pretrain | SFT v1 |
|:--|:---------|:-------|
| スコア | 事実上ほぼ全部 0 | 通常モードでスコア発生 |
| コーディング | 0/34 | 4/17 |
| 振る舞い | 書き写し · 繰り返し | 形式は試す（中身はよく間違う） |

SFT 5,000 step だけで **「全く従えない → 形式は試す」** へ一段上がります。

#### Q&A · 通常チャット

| 言語 | 設問 | スコア | メモ |
|:----:|:-----|:----:|:-----|
| 🇰🇷 | 首都 / りんご / 要約 | 2 / 1 / 1 | 「서울」は出るが文が矛盾 |
| 🇯🇵 | 首都 / りんご / 翻訳 | 0 / 1 / 0 | 人口の話 · 指示無視 |
| 🇺🇸 | 首都 / 列車 / 要約 | **4** / 1 / 3 | 当時いちばん強い信号は Paris |

Pretrain only (en_fact): 質問をドイツの首都の問いにすり替えて聞き返す → 0 点。

#### コーディング（通常）

| 問題 | テスト | スコア | メモ |
|:-----|:------:|:----:|:-----|
| is_prime | 0/5 | 0 | 偶数判定に変質 |
| reverse_string | 0/3 | 0 | `.lower()` だけ |
| factorial | 0/3 | 1 | 再帰の骨格、0 処理なし |
| **is_palindrome** | **3/3** | **5** | ✅ 唯一きれいな正解 |
| find_max | 1/3 | 1 | 運の良い 1 ケース |

```python
# ✅ v1 通過
def is_palindrome(s):
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]

# ❌ v1 is_prime → 事実上偶数検査
def is_prime(n):
    if n <= 1:
        return False
    return n % 2 == 0
```

</details>

<details>
<summary><b>📙 pretrain_base_v1 · チャット ~0 点</b></summary>

<br/>

📁 [`benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json)

| | 結果 |
|:--|:-----|
| チャット反応 | ほぼなし · ~0 / 28 |
| コーディング | 0 / 34 |
| 振る舞い | 書き写し · 同じ言葉の繰り返し · 質問のすり替え |

SFT 前は **指示形式そのものが未学習** です。  
v1 SFT 後に「形式を試す」変化がベンチに初めて載ります。

</details>

---

## 7. 所感 · 次に

### ✅ バージョンを通じてうまくいったこと

- Pretrain loss 11 → ~2、SFT v1 2.67 → ~1.05 → **学習ループは正常**
- v1: 「形式を試す」がベンチに載った
- v2: THINKING **閉じ** がほぼ復旧（0/14 → 13/14）
- v3: 通常チャット **コーディング 4/5** — データ補強の目標達成
- 英語 fact/math は v3 で再び安定信号

### 🚧 まだ詰まっていること

1. 🧠 THINKING モードのコーディング → 依然として散文のみ (0/5)
2. 🇰🇷 韓国語 → v3 でカテゴリ崩壊
3. 算術·指示追従の不安定（途中は合うが最終が外れる）
4. `reverse_string` のように **正解本体 + 危険な top-level**
5. 実使用品質まではまだ遠い

### 🧭 次に

| 優先 | 内容 |
|:----:|:-----|
| 1 | THINKING でも **コードを出しタグを閉じる** SFT / 制約デコード |
| 2 | 韓国語の事実·算術·要約の比重を再調整（崩壊の修復） |
| 3 | コーディング出力形式の整理（ハーネスを壊す side effect 抑制） |
| 4 | DPO / 訂正 SFT / RLVR |
| 5 | **同じ 14 問** でバージョン比較を維持 |

---

### 📁 元ファイル

| パス | 内容 |
|:-----|:-----|
| [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) | ⭐ **最新** SFT v3 |
| [`ckpt/benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json) | SFT v2 |
| [`ckpt/benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json) | SFT v1 |
| [`ckpt/benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json) | Pretrain |

> 重み（`.pt`）と元コーパスは公開リポジトリに含めていません。

---

<div align="center">

[README](README.ja.md) · [ARCHITECTURE](ARCHITECTURE.ja.md) · [POST-TRAINING](POST-TRAINING.ja.md)

</div>
