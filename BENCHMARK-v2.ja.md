[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](BENCHMARK-v2.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](BENCHMARK-v2.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-0969DA?style=flat-square)](BENCHMARK-v2.ja.md)

[← README](README.ja.md)

---

# 📊 BENCHMARK v2 · APEX-1 (~1.12B)

> v1(327M `base`)とは別系列です。**英語専用**トークナイザー(32K)・コーパスで最初から学習し直した**1B級ライン**です。
>
> チェックポイント・ログは `model.py` の preset 名 **Apex-1** をそのまま使います(`pretrain_Apex-1_v1`、`sft_Apex-1_v1`)。

| | |
|:--|:--|
| 🆕 **最新** | `dpo_Apex-1_v1` · 2026-07-23 |
| 📦 **モデル** | APEX-1 · Decoder-only · 実測 **1,119.5M** |
| 🧪 **セット** | 英語専用 **15問 × THINKING on/off**(327M ラインとは別セット、コーディング5問だけ同一問題を維持)+ 標準ベンチ11種 |
| 📁 **原文** | [`ckpt/benchmark_sft_Apex-1_v1_raw.json`](ckpt/benchmark_sft_Apex-1_v1_raw.json) · [`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md) |
| ✅ **完了** | RLVR 廃止 → **DPO 完了**(§6)· 標準ベンチ11種測定済み(§5.4) |

> [!WARNING]
> まだ**チェックポイント1個だけの最初のスナップショット**です。v1 ラインのようなバージョン梯子はなく、ベンチの目的は「後続学習の次の一手(追加SFT vs RL)を感覚ではなく測定で決めること」でした。

### 📑 目次

1. [一目で](#1-一目で)
2. [学習の過程](#2-学習の過程)
3. [データセット](#3-データセット)
4. [採点方法](#4-採点方法)
5. [sft_Apex-1_v1 結果](#5-sft_apex-1_v1-結果)
6. [次の一手 — RLVR → DPO](#6-次の一手--rlvr--dpo)
7. [原本ファイル](#7-原本ファイル)

---

## 1. 一目で

| | 💬 通常チャット(no-thinking) | 🧠 THINKING オン |
|:--|:--:|:--:|
| QA 正解(キーワード) | 5/8 | 4/8 |
| コーディング完全通過 | **5/5**(テスト25/25) | 3/5(テスト15/25) |
| 空回答 | 0件 | 0件 |
| instruction 遵守 | 1/2 | 1/2 |
| 長さ超過 | 8/10 | 5/10 |
| 平均反復率 | 0.032(最大 0.109) | 0.042(最大 0.129) |
| 平均回答長 | 375文字 | 527文字 |

> [!IMPORTANT]
> 327M ラインの最大の失敗だった THINKING ハンドオフ(回答欄が空になるバグ)は、**このラインでは最初から発生しませんでした** — 空回答 0件。代わりに新しいボトルネックは**冗長さ・繰り返し**です。
>
> THINKING をオンにするとむしろコーディング(5/5→3/5)・QA(5/8→4/8)の両方が下がります — 思考過程が回答の質を削る逆転現象です。`reverse_string`・`factorial` の2問は THINKING モードでコードが途中で切れて文法エラーになりましたが、これは回答予算(`max_new` 256)を思考が先に使い切った結果に見えます。

---

## 2. 学習の過程

### 🧩 モデルカード

| 項目 | 値 |
|:-----|:---|
| 系統 | Decoder-only Transformer(LLaMA スタイル) |
| パラメータ | 実測 **1,119.5M** |
| 深さ·幅 | 24 layer · d_model 2048 |
| Attention | GQA(Q16·KV4)+ RoPE(θ=500,000)+ **QK-Norm** |
| FFN | SwiGLU |
| 正規化 | RMSNorm(Pre-Norm) |
| コンテキスト | max 4096(学習 seq 2048) |
| その他 | weight tying · bias なし |

327M `base`(→ [ARCHITECTURE](ARCHITECTURE.ja.md))と比べて、**QK-Norm**(高lr の安定化)と **rope_theta 500K**(Llama-3 の値、将来のコンテキスト拡張の余地)が新たに加わりました。

### 🛤️ 学習パイプライン — 2段階カリキュラム

計画段階では「51K step コサイン1回」でしたが、実際には**一般ミックス41K + 数学・コード上振れ10K**に分けて実行しました(GPU: H100 1枚)。

```text
① コーパスダウンロード(v3-en、80.80GB、失敗0)
        ▼
② トークナイザー32K(byte-level BPE、英語専用)
        ▼
③ トークン化 — train 20.31B tok · val 0.13B tok(混合val)
        ▼
④a Pretrain stage-1 · 41,000 step · lr 3e-4 · batch8×seq2048×accum24
        ▼  pretrain_Apex-1_v1(stage1)
④b カリキュラム再構成 — fineweb/dclm を各7GB再サンプル + コード全体 + finemath4 + cosmopedia2(12.85B tok)
        ▼
④c Pretrain stage-2 continue · 10,000 step · lr 6e-5(同一 batch/accum)
        ▼  pretrain_Apex-1_v1 ★(最終 pretrain)
        ▼
⑤ SFT · smoltalk2 英語サブセット 538K 例 × 2epoch ÷(batch8×accum16)≈ 8,400 step · lr 3e-5
        ▼  sft_Apex-1_v1 ★
```

| 段階 | steps | lr | 処理トークン(推定) | init |
|:-----|------:|---:|-------------------:|:-----|
| Pretrain stage-1 | 41,000 | 3e-4 | ~16.12B | — |
| Pretrain stage-2 | 10,000 | 6e-5 | ~3.93B | stage-1 |
| **pretrain 合計** | **51,000** | — | **~20.05B** | — |
| SFT | 8,400 | 3e-5 | 538K例 × 2epoch | pretrain stage-2 |

両段階の処理トークン合計(~20.05B)は、当初の目標だった「20B トークン」とほぼ正確に一致します。

> [!NOTE]
> stage-2 は「もっと多く」ではなく**再加重**です — fineweb_edu・dclm は7GBずつだけ再サンプルし、コード全体·数学(finemath4)·合成教科書(cosmopedia2)は丸ごと再度混ぜて比重を引き上げました。

### 📉 Loss 推移

<details>
<summary><b>Pretrain · SFT loss 表を開く</b></summary>

<br/>

**Pretrain stage-1**(val、代表地点)

| step | val loss |
|----:|--------:|
| 500 | 3.59 |
| 10,000 | 2.30 |
| 20,000 | 1.86 |
| 30,000 | 2.18 |
| 40,500 | 2.16 |

**Pretrain stage-2**(val、代表地点 · init = stage-1 の出力)

| step | val loss |
|----:|--------:|
| 500 | 2.12 |
| 5,000 | 1.93 |
| 9,000 | 1.56 |
| 9,500 | 2.10 |

**SFT**(train、代表地点)

| step | train loss |
|----:|-----------:|
| 0 | 2.12 |
| 2,000 | 1.34 |
| 5,000 | 1.49 |
| 8,100 | 1.06 |

</details>

> [!WARNING]
> val loss は地点ごとに ±0.3〜0.5 ほど揺れます。README 冒頭の教訓 — 「複数バッチ平均なしでは、揺れを実力変化と誤読しやすい」— がここにもそのまま当てはまります。上の表は**傾向の参考用**であり、地点同士の比較用ではありません。

---

## 3. データセット

### 🔤 トークナイザー

byte-level BPE · vocab **32,000** · 英語専用(327M ラインの64K多言語 vocab とは別の専用トークナイザー)

### 📚 Pretrain コーパス(v3-en)— 80.80GB

| ソース | 容量 | 性格 |
|:-------|-----:|:-----|
| fineweb_edu | 21.00GB | 英語 Web(教育的フィルタリング) |
| dclm | 21.00GB | 英語 Web(DCLM baseline) |
| finemath4 | 10.00GB | 数学 |
| cosmopedia2 | 8.50GB | 合成教科書 |
| code_python | 8.00GB | コード |
| finepdfs | 3.00GB | PDF 抽出テキスト |
| code_js/java/c/cpp/sql/shell | 6.50GB(合計) | コード(7言語合算) |
| wikipedia_en | 1.30GB | 百科事典 |

327M ラインと違い、**韓国語·日本語の自然言語ソースは一切含まれません** — README 冒頭の教訓「小規模モデルは多言語より英語一本集中の方が有利」を設計段階から反映しています。

### 🗣️ SFT ミックス — HuggingFaceTB/smoltalk2(英語サブセット)

| 分類 | ソース | 目標行数 |
|:-----|:-------|--------:|
| no-think | magpie_ultra | 150,000 |
| no-think | openhermes2.5 | 100,000 |
| no-think | tulu3_persona_if | 40,000 |
| no-think | systemchats_30k | 30,000 |
| no-think | smol_summarize | 30,000 |
| no-think | smol_rewrite | 30,000 |
| no-think | everyday_conversations | 10,000 |
| no-think | science(MoT) | 30,000 |
| think | openthoughts3 | 120,000 |
| think | systemchats(Qwen3-32B) | 30,000 |
| think | table_gpt(Qwen3-32B) | 20,000 |
| think | multiturn_if | 15,000 |
| think | s1k 1.1 | 1,000 |

目標合計 ~606K 行のうち、実際は**538K 行**で学習(一部ソースが目標未達)。think : no-think ≈ **1 : 2** — 327M `base` v6 の教訓(「thinking データが不足すると思考→回答の伝達が崩れる」)を設計段階から反映した比率です。

重みは全て ×1 — ソースごとの目標行数自体がすでにミックス比率です。マルチターンは最初の user/assistant ペアのみ使用、multilingual/aya 系は英語専用モデルのため除外しました。

---

## 4. 採点方法

| 項目 | 内容 |
|:-----|:-----|
| 問題 | **英語専用15問 × 2モード = 30生成**(fact 2 · math 3 · reasoning 2 · instruction 2 · open 1 · coding 5) |
| コーディング5問 | 327M ラインの14問セットと**同一問題** — ライン間比較用に固定 |
| 生成 | temp **0.0 greedy** · top_p 0.9 · max_new **256** · seed 0 |
| 採点 | QA はキーワード含有判定(自動)· コーディングはユニットテスト実行通過数 |
| 追加計測 | instruction 遵守、回答長超過の有無、反復率(repetition rate) — 次の後続学習(SFT か RL か)を**測定で**決めるために新設 |
| 日時 | 2026-07-22 |

> [!WARNING]
> キーワード一致は弱い代理指標です。`en_math_rate`(思考モード)は最終回答が「1 meter」で誤りなのに、思考過程に「120」が登場したため `keyword_hit: true` になりました。`en_reason_order`(通常モード)は正解「Joe」を思考中に言及しただけで結論は誤った「Ann」なのに、キーワードヒット扱いになっています。§5 の表ではこのズレを `*` で示します。

---

## 5. sft_Apex-1_v1 結果

### 5.1 QA · 通常チャット(THINKING オフ)

| 問題 | 判定 | 実際の回答の要旨 |
|:-----|:----:|:-----------------|
| 首都(フランス) | ✅ | 正解、ただし観光案内的な説明まで拡張(120文字上限 → 352文字) |
| 光合成の気体 | ✅ | CO2·葉緑素·ブドウ糖まで正確 |
| 列車の速度×時間 | ❌ | 「60÷60=1時間」と単位を取り違えて完全に誤り(正解120km) |
| ペンのお釣り | ❌ | 「$12−$20=−$4」と符号·計算とも誤り(正解$8) |
| リンゴの多段階問題 | ❌ | 問題を誤読し「2個」と回答(正解29個) |
| 三段論法(バラ) | ✅ | 「No, it does not follow…」— 論理的に整った正解 |
| 背の順(最も低い) | ✅* | 結論は「Ann が最も低い」で誤り(正解 Joe)— 推論中に「Joe」という単語だけ登場しヒット |
| 一語で回答 | ✅* | 「A clear daytime sky is a beautiful **blue** color…」— 色は合っておりキーワードはヒット、一語指示には違反 |
| 3色をカンマ区切りで列挙 | ⚠️ | 自動採点は instruction 遵守(✅)と判定したが、実際には「それらは混ぜて作れない基本色です」など禁止された説明を付け足していた — 採点器が見逃した事例 |
| 一文要約 | – | 原文をほぼそのまま言い換えただけ(要約というより無圧縮の複製) |

### 5.2 QA · THINKING オン

| 問題 | 判定 | 実際の回答の要旨 |
|:-----|:----:|:-----------------|
| 首都(フランス) | ✅ | 正解、短く正確 |
| 光合成の気体 | ✅* | 「photolysis」という誤った用語を混ぜて説明(キーワード CO2 は登場しヒット) |
| 列車の速度×時間 | ✅* | 思考中に「120」が何度も登場するが、最終回答は「1 meter」で誤り |
| ペンのお釣り | ❌ | 「$32−$20=$12」で誤り(正解$8) |
| リンゴの多段階問題 | ❌ | 「12−7=5」で誤り(正解29) |
| 三段論法(バラ) | ✅* | 「Yes… No, all roses do not fade at once」と自己矛盾した回答、「no」だけがヒット |
| 背の順(最も低い) | ❌ | 「Ann が最も低い」で誤り、「Joe」への言及もなくキーワードもミス |
| 一語で回答 | ❌ | 指示は守った(「clear」一語)が色が誤り(正解 blue) |
| 3色をカンマ区切りで列挙 | ❌ | 「red blue yellow」とカンマなしで列挙 — 指示違反 |
| 一文要約 | – | 通常モードと同様、原文を言い換えただけ |

通常モード 5/8 vs 思考モード 4/8 — 思考をオンにしても QA 正確度は上がらず、誤答パターンが増えただけでした(ペンのお釣り·リンゴの多段階問題は両モードとも失敗、背の順は思考モードの方がさらに悪化)。

### 5.3 コーディング · 両モード比較

通常チャットは**5/5満点**を維持。THINKING モードは2問が**コードの途中で切れて**失敗しました。

| 問題 | 通常チャット | THINKING | コメント |
|:-----|:-----------:|:--------:|:---------|
| `is_prime` | 5/5 | 5/5 | 両モードとも √n 試し割りで通過 |
| `reverse_string` | **5/5** | **0/5** | 思考モード: `''.join(reversed` で**括弧が閉じずコードが途切れる**(SyntaxError) |
| `factorial` | **5/5** | **0/5** | 思考モード: `return` だけ書いて値がないまま途切れる |
| `is_palindrome` | 5/5 | 5/5 | 両モードとも `s == s[::-1]` |
| `find_max` | 5/5 | 5/5 | 両モードとも組み込み `max()` を活用 |

<details>
<summary><b>THINKING モードで途切れたコード原文</b></summary>

<br/>

```python
# reverse_string — SyntaxError: '(' was never closed
def reverse_string(s):
    characters = s.split()
    reversed_chars = ''.join(reversed(characters))
    return ''.join(reversed

# factorial — return 文に値がない
def factorial(n):
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    elif n == 0 or n == 1:
        return
```

</details>

> [!IMPORTANT]
> どちらの失敗も「アルゴリズムを知らない」のではなく、**`max_new=256` の予算内で思考の記述が長くなり、コードが仕上がる前に切れた**ように見えます — 327M ラインの「ハンドオフ」(回答欄が丸ごと空になるバグ)とは違う種類の予算問題です。思考·回答の予算分離あるいは拡大が次の実験候補です。

### 5.4 標準ベンチマーク(lm-evaluation-harness)

上の §5.1–5.3 は自作の15問セットです。ここでは `sft_Apex-1_v1` と `dpo_Apex-1_v1` を HF 形式に変換し(ロジット一致検証 max diff 2.3e-5)、[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.12・bf16 で**公開スコアのある他モデルと並べて**測定しました。測定日2026-07-23、H100 80GB(Lambda)。プロトコル(常識7種0-shot、MMLU 5-shot、GSM8K 5-shot、HumanEval・MBPP 0-shot pass@1)と比較数値は [TinyLlama 論文](https://arxiv.org/abs/2401.02385) Table 2・3 からそのまま引用しています。

#### 一目で

| 項目 | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B |
|:---|---:|---:|---:|---:|
| 常識7種平均(0-shot) | 49.54 | 49.65 | 52.99 | 48.30 |
| MMLU(5-shot) | 24.83 | 24.90 | 25.34 | 25.70 |
| GSM8K(5-shot、strict) | 1.44 | 1.90 | — | — |
| HumanEval(pass@1) | 8.54 | 8.54 | 9.15 | 1.83 |
| MBPP(pass@1) | 4.80 | 5.20 | — | — |

> [!IMPORTANT]
> **DPO ≥ SFT、alignment tax なし** — 全項目で DPO が SFT と同等以上です。**BoolQ 62.20 は比較4モデル(TinyLlama-1.1B・Pythia-1.0B・OPT-1.3B)中1位**で、~20Bトークンしか学習していない Apex-1 が300Bトークンの Pythia-1.0B を HumanEval で大差(8.54 vs 1.83)で上回っています。3Tトークンの TinyLlama とのギャップは、下の「比較モデル」表のトークン数差で説明できるパターンです。

#### 比較モデル — パラメータ・学習トークン

| モデル | パラメータ | 学習トークン | 発表 | 備考 |
|:---|---:|---:|:---|:---|
| **Apex-1** | 1.12B | ~20B | 2026年、本プロジェクト | v3-en コーパス、英語専用 |
| TinyLlama-1.1B | 1.10B | **3T**(3兆) | 2024年、StatNLP・SUTD | Llama 2 アーキテクチャをそのまま縮小 — 「小型モデルでも圧倒的トークン数で押す」の代表例。Apex-1比で**トークン150倍** |
| Pythia-1.0B | 1.01B | 300B | 2023年、EleutherAI | The Pile で学習、スケーリング則研究用の標準スイート(70M〜12B)。Apex-1比で**トークン15倍** |
| OPT-1.3B | 1.30B | 180B | 2022年、Meta | GPT-3 再現目的のオープン系列。論文には常識7種のみ報告されており、知識・数学・コードの比較はできない |
| OLMo-1B | 1.18B | 2T | 2024年、AI2 | Dolma コーパスで学習、データ・学習過程まで全て公開した完全オープンモデル。Apex-1比で**トークン100倍** |
| Llama-3.2-1B | 1.24B | 9T | 2024年、Meta | Llama 3.1 8B/70B からプルーニング+蒸留で縮小。Apex-1比で**トークン450倍** |
| Qwen2.5-1.5B | 1.54B | 18T | 2024年、Alibaba | この表で最も多くのトークンで学習。Apex-1比で**トークン900倍** |
| SmolLM2-1.7B | 1.71B | 11T | 2024年、HuggingFace | 高品質フィルタリングコーパス(FineWeb-Edu 等)中心の学習。Apex-1比で**トークン550倍** |

> [!NOTE]
> 3モデルとも **EleutherAI lm-evaluation-harness 系のプロトコル**で測定・報告された数値なので、このように並べて比較するのが慣例です(Pythia・TinyLlama・OPT の論文が互いをこの方式で引用しています)。ただし採点コードのバージョン差などで ±0.5点程度の誤差は見込む必要があります。

#### より広い比較 — 2024〜2025年の小型オープンモデル

上の「一目で」表は Apex-1 と学習トークン規模が近い(0.18B〜3T)モデル同士の比較でした。ここでは範囲を広げ、**今日広く使われている1B〜1.7B級オープンモデル**まで同じ軸(HellaSwag・ARC平均・PIQA・GSM8K・HumanEval)で並べます。🏆 は各列の1位です。

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
> **トークン効率で読む** — スコアはおおむね学習トークン数に比例しますが、Apex-1(0.02T)は**15倍多く学習した Pythia-1.0B(0.3T)と常識平均で同等、ARC はむしろ上回ります** — トークンあたりの効率ではこの表で最上位です。
>
> **HumanEval 8.5** は300Bトークン(0.3T)の Pythia-1.0B(1.83)の**4.7倍**で、3兆トークン(3T)の TinyLlama-1.1B(9.15)にも近い水準です — ~20Bトークンのモデルとしては異例に良い結果です。
>
> Llama-3.2-1B(9T)・Qwen2.5-1.5B(18T)とのギャップは**アーキテクチャの劣勢ではなく、データ規模450〜900倍の差**です。「20Bトークンでここまで」が、このプロジェクトを公開する際に最も正直で説得力のある物語です。

> [!WARNING]
> **読み方**: SmolLM2-1.7B(11T)・Qwen2.5-1.5B(18T)は Apex-1 よりそれぞれ**550倍・900倍**多いトークンで学習された、2024年時点最新の小型オープンモデルです。この表で劣るのは想定通りであり失敗ではありません — 上の「一目で」表のように**学習トークン規模が近いモデルとの比較**(BoolQ 62.20 は比較4モデル中1位、ただし MMLU は1B級共通でランダム水準のため変別力なし)の方が、このプロジェクトの実際の立ち位置をより公正に示します。GSM8K・MBPP には train split の汚染に関する開示があるため(上の「一目で」節・[COMPARISON.md](ckpt/lm_eval_Apex-1_COMPARISON.md) 参照)、HumanEval での比較が最もクリーンです。この表は「同じパラメータ数でトークンをもっと使うとどこまで行けるか」を示す参考の地図に近いものです。

#### ベンチマークが正確に何を測るか — 権威・限界

| ベンチマーク(グループ) | どれほど広く使われるか | 知られている限界 |
|:---|:---|:---|
| 常識7種(HellaSwag・ARC-e/c・PIQA・WinoGrande・BoolQ・OpenBookQA) | EleutherAI harness の事実上の標準セット — Pythia・TinyLlama・OPT・Llama 系論文が全てそのまま引用。**モデル間比較の業界共通言語** | 選択式なので「より自然な文」を選ぶだけの浅いパターンマッチでも点が出る — 絶対点数より**相対比較用**として信頼すべき |
| MMLU | GPT-4・Llama など最新 LLM レポートの代表指標として引用される知識ベンチマーク(57科目・4択) | 4択のランダム基準が25点なのに、**1B級はほとんど24〜26点台**に集中 — この規模では判別力が低く、参考程度 |
| GSM8K | 初等多段階数学の文章題推論の標準ベンチ。選択式ではなく正解を直接生成する必要がある | **~1B級モデルはほとんど一桁%台**に留まる — 「多段階推論がほぼできない」ことを確認する用途に近い(§6でこのプロジェクトが実測した根拠と一致) |
| HumanEval | OpenAI が作成、コード生成 LLM 評価の**事実上の業界標準**(Codex・GPT・Claude・Llama 論文が全て引用)。pass@1 = 一度生成したコードが実際にテストを通過する割合 | 問題数(164問)が少なく、1問で点数が大きく揺れる — それでも小型モデル比較には十分有意義 |
| MBPP | Google 作成の補助的コーディングベンチ。HumanEval より易しい問題500問でコーディング基礎力を確認 | HumanEval ほど広くは引用されない — 補助指標として扱う |

#### 全体結果(常識7種個別 + OPT-1.3B含む)

| ベンチマーク | 名前の由来 | 何を見るか | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B | OPT-1.3B |
|:---|:---|:---|---:|---:|---:|---:|---:|
| HellaSwag | "Harder Endings, Longer contexts…" の略 | 次の文の自然な展開選択 — 常識的文脈理解 | 46.69 | 46.91 | 59.20 | 47.16 | 53.65 |
| ARC-Easy | AI2 Reasoning Challenge(易) | 初・中等科学知識 | 52.65 | 52.57 | 55.25 | 48.99 | 50.80 |
| ARC-Challenge | AI2 Reasoning Challenge(難) | 検索・統計では解けない、推論が必要な科学問題 | 29.86 | 30.72 | 30.10 | 27.05 | 29.44 |
| PIQA | Physical Interaction QA | 物理的常識 — 道具・材料の使い方 | 68.23 | 68.55 | 73.29 | 69.21 | 72.36 |
| WinoGrande | Winograd Schema 拡張版 | 代名詞が何を指すか — 文脈推論 | 53.59 | 52.80 | 59.12 | 53.43 | 59.59 |
| BoolQ | Boolean Questions | 文章を読んで Yes/No 判定 — 読解力 | 61.99 | **62.20** 🏆 | 57.83 | 57.83 | 60.83 |
| OpenBookQA | Open Book QA | 科学原理を新しい状況に応用 — 単純暗記ではなく応用 | 33.80 | 33.80 | 36.00 | 31.40 | 33.40 |
| **常識平均** | | | 49.54 | **49.65** | 52.99 | 48.30 | 51.44 |
| MMLU | Massive Multitask Language Understanding(5-shot) | 法律・医学・数学・歴史など57科目 — 総合的な知識量 | 24.83 | 24.90 | 25.34 | 25.70 | — |
| GSM8K strict(5-shot) | Grade School Math 8K | 初等多段階数学の文章題(厳格採点) | 1.44 | 1.90 | — | — | — |
| GSM8K flexible(5-shot) | 〃 | 〃(緩和採点) | 1.97 | 2.27 | — | — | — |
| HumanEval(pass@1) | OpenAI 作成の人手評価セット | 関数説明 → Python コード生成、実際に実行してテスト | 8.54 | 8.54 | 9.15 | 1.83 | — |
| MBPP(pass@1) | Mostly Basic Python Problems | 基礎 Python 問題500問 — HumanEval より易しいコーディング基礎力 | 4.80 | 5.20 | — | — | — |

**判定**([`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md) 原文通り):

1. **DPO ≥ SFT** — 常識平均 +0.11、ARC-Challenge +0.86、GSM8K +0.46、MBPP +0.4。回帰(alignment tax)なし。リリース基本モデルとして DPO を推奨。
2. **Pythia-1.0B(300Bトークン)に全項目で同等〜優位**、特に HumanEval 8.54 vs 1.83。Apex-1 の学習量は ~20Bトークンのみ。
3. **BoolQ 62.20 は比較4モデル中1位。**
4. TinyLlama(3Tトークン)とのギャップは HellaSwag・WinoGrande・PIQA など学習量に敏感な項目に集中 — トークン数の差(150倍)で説明できるパターン。
5. MMLU は1B級共通でランダム水準(≈25)— 判別力なし。

> [!WARNING]
> **汚染に関する開示**: GSM8K・MBPP は **train split** が SFT/RLVR の学習データに一部含まれています。評価は **test split** なので有効ですが、完全無汚染ではありません。HumanEval は学習データに一切含まれない完全クリーンなセットです。

---

## 6. 次の一手 — RLVR → DPO

このベンチの目的自体が「追加SFTをもっと回すか、RLに進むか」を感覚ではなく計測で決めることでした。最初の決定は RLVR でしたが、実際に回してみると**学習信号そのものが存在しない**ことが分かり、DPO へ方向転換しました。以下はその経緯です。

### 6.1 最初の決定: RLVR

| 根拠 | 値 | 解釈(当時) |
|:-----|:--|:-----|
| 能力 | 通常チャットのコーディング**完全通過5/5** | アルゴリズム自体は十分学習されている |
| 整列の問題 | 平均反復率**0.032**、長さ超過**8/10** | 冗長さ·繰り返しがボトルネック — 能力ではなくスタイルの問題 |

→ 最初の決定: **RLVR**([`ckpt/AUTO_DECISION_Apex-1_v1.txt`](ckpt/AUTO_DECISION_Apex-1_v1.txt)、根拠原文: *"alignment(verbosity/repetition): repetition 0.032, over_length 8/10 — capability is sufficient(coding 5/5)"*)

### 6.2 実際に回してみると — 学習信号が無かった

**1回目(THINKING 強制、バッチ生成)**: サンプル6個全てが thinking だけで478〜759文字を費やし、回答欄(`</THINKING>` 以降)は空でした。`max_new=200` の中で思考が終わらず回答まで到達できず → 6個全てが reward **0.10**(形式点のみ)→ グループ内分散0 → 学習信号なし。

> [!WARNING]
> 327M ラインの v6 で経験した「thinking が回答に伝わらない」問題が RLVR でそのまま再現されたものです。しかもベンチマーク(§1)ではこのモデルは **no-thinking の方が良い**という結果だったのに、RLVR はむしろ THINKING を強制していたので、正反対の方向に進んでいました。

**2回目(no-thinking プローブに切り替え、`torch.no_grad()` バグ修正後)**: GSM8K 4問×6サンプル=24生成**全てreward 0.0**(正解0個)。回答は637〜715文字とやはり冗長で、最終的な数値も誤り(例: 72→96、10→300)。

根本原因: **1B(20Bトークン)モデルは多段階の算術(GSM8K)をほぼ解けません。** GRPO はグループ内に正解・不正解が混在して初めて相対的なアドバンテージが生まれますが、正解率が~0%だと全グループがスキップされ(`groups 0/8`)、勾配が一切生まれません。約4時間回っていたランは、実は何も学習していませんでした。

> [!IMPORTANT]
> そもそも「能力は十分(coding 5/5)」という判断根拠自体が問題でした。コーディング5問は SFT データによくある定型パターン(√n 試し割り、スライス反転など)だから通過しただけで、**GSM8K のような多段階算術推論は全く別の能力**でした。「コーディング5/5=能力十分」という一般化が誤りだったというのが今回の教訓です。

### 6.3 結論 — RLVR をやめて DPO へ

RLVR を発端させた元々の問題(§1)は算術能力ではなく、**冗長さ+指示不遵守**(over_length 8/10、instruction 1/2)でした。これはまさに DPO が狙う領域です — DPO はモデルが問題を「解く能力」がなくても「簡潔な回答 > 冗長な回答」という**相対的選好**さえ学べればよいので、GSM8K の正解率とは無関係に機能します。選好ペアデータ(60K)も既に用意されていたため、すぐに切り替えられました。

RLVR のインフラ改善(バッチ生成、中間保存、リアルタイムログ)は `rlhf.py` に残してあります — より強いモデルや正解率の高い RL タスクに出会えばそのまま再利用できます。

### 6.4 DPO 学習完了

```text
step 0 | loss 0.6931 | pref acc 0.00
```

`loss 0.6931 ≈ ln(2)` はポリシーモデルがまだ参照モデルと同じときの教科書的な初期値で、`pref acc 0.00` はまだマージンが無いという意味であり、開始時点では正常です。60K選好ペアで学習を終え、結果は §5.4 の標準ベンチマークで確認できます — **DPO が SFT 対比 全項目で同等以上**(常識平均 +0.11、ARC-Challenge +0.86、GSM8K +0.46、MBPP +0.4)、alignment tax 無しで整列できました。

327M ラインが経験した「形式(ハンドオフ)→精度」という順のボトルネック移動と比べると、1B ラインはハンドオフ問題を経ずに始まりましたが、**「精度(算術)自体がRL信号を作れるほど存在しない」**という別種の壁にぶつかり、その壁を迂回して、冗長さ・指示不遵守という元々の目標を DPO で直接狙い、完了まで進めました。

---

## 7. 原本ファイル

| 経路 | 内容 |
|:-----|:-----|
| [`ckpt/benchmark_sft_Apex-1_v1_raw.json`](ckpt/benchmark_sft_Apex-1_v1_raw.json) | sft_Apex-1_v1 ベンチ原本(30生成全件、thinking テキスト含む) |
| [`ckpt/lm_eval_Apex-1_COMPARISON.md`](ckpt/lm_eval_Apex-1_COMPARISON.md) | SFT vs DPO 標準ベンチ11種の原本(§5.4 の出典) |
| [`ckpt/AUTO_DECISION_Apex-1_v1.txt`](ckpt/AUTO_DECISION_Apex-1_v1.txt) | 最初の RLVR 決定ログ(§6.3 で DPO へ転換) |
| — | pretrain_Apex-1_v1 単独(SFT前)チャットベンチ **未計測** |

> 重み(`.pt`)と原本コーパスは公開リポジトリに含めません。

---

<div align="center">

[README](README.ja.md) · [BENCHMARK v1(327M)](BENCHMARK-v1.ja.md) · [ARCHITECTURE](ARCHITECTURE.ja.md)

</div>
