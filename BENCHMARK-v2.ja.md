[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](BENCHMARK-v2.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](BENCHMARK-v2.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-0969DA?style=flat-square)](BENCHMARK-v2.ja.md)

[← README](README.ja.md)

---

# 📊 BENCHMARK v2 · APEX-1 (xl, ~1.12B)

> v1(327M `base`)とは別系列です。**英語専用**トークナイザー(32K)・コーパスで最初から学習し直した**1B級ライン**です。

| | |
|:--|:--|
| 🆕 **最新** | `sft_xl_v1` · 2026-07-22 |
| 📦 **モデル** | APEX-1 · Decoder-only · 実測 **1,119.5M**(`xl`) |
| 🧪 **セット** | 英語専用 **15問 × THINKING on/off**(327M ラインとは別セット、コーディング5問だけ同一問題を維持) |
| 📁 **原文** | [`ckpt/benchmark_sft_xl_v1_raw.json`](ckpt/benchmark_sft_xl_v1_raw.json) |

> [!WARNING]
> まだ**チェックポイント1個だけの最初のスナップショット**です。v1 ラインのようなバージョン梯子はなく、ベンチの目的は「後続学習の次の一手(追加SFT vs RL)を感覚ではなく測定で決めること」でした。

### 📑 目次

1. [一目で](#1-一目で)
2. [学習の過程](#2-学習の過程)
3. [データセット](#3-データセット)
4. [採点方法](#4-採点方法)
5. [sft_xl_v1 結果](#5-sft_xl_v1-結果)
6. [次の一手 — RLVR](#6-次の一手--rlvr)
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
        ▼  pretrain_xl_v1(stage1)
④b カリキュラム再構成 — fineweb/dclm を各7GB再サンプル + コード全体 + finemath4 + cosmopedia2(12.85B tok)
        ▼
④c Pretrain stage-2 continue · 10,000 step · lr 6e-5(同一 batch/accum)
        ▼  pretrain_xl_v1 ★(最終 pretrain)
        ▼
⑤ SFT · smoltalk2 英語サブセット 538K 例 × 2epoch ÷(batch8×accum16)≈ 8,400 step · lr 3e-5
        ▼  sft_xl_v1 ★
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

## 5. sft_xl_v1 結果

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

---

## 6. 次の一手 — RLVR

このベンチの目的自体が「追加SFTをもっと回すか、RLに進むか」を感覚ではなく計測で決めることでした。

| 根拠 | 値 | 解釈 |
|:-----|:--|:-----|
| 能力 | 通常チャットのコーディング**完全通過5/5** | アルゴリズム自体は十分学習されている |
| 整列の問題 | 平均反復率**0.032**、長さ超過**8/10** | 冗長さ·繰り返しがボトルネック — 能力ではなくスタイルの問題 |

**→ 決定: RLVR**([`ckpt/AUTO_DECISION_xl_v1.txt`](ckpt/AUTO_DECISION_xl_v1.txt))

> [!NOTE]
> 根拠原文: *"alignment(verbosity/repetition): repetition 0.032, over_length 8/10 — capability is sufficient(coding 5/5)"*
>
> 能力の問題であれば SFT データ·ステップを増やす方が正しいですが、いまのボトルネックは「何を知っているか」ではなく「どう答えるか」なので、次の後続学習は DPO·追加SFT より **RLVR(検証可能な報酬に基づく RL)** で進めることにしました。

327M ラインが経験した「形式(ハンドオフ)→精度」という順のボトルネック移動と比べると、1B ラインはハンドオフ問題を経ずに最初から**「精度+冗長さ」**の段階にいます。

---

## 7. 原本ファイル

| 経路 | 内容 |
|:-----|:-----|
| [`ckpt/benchmark_sft_xl_v1_raw.json`](ckpt/benchmark_sft_xl_v1_raw.json) | sft_xl_v1 ベンチ原本(30生成全件、thinking テキスト含む) |
| [`ckpt/AUTO_DECISION_xl_v1.txt`](ckpt/AUTO_DECISION_xl_v1.txt) | RLVR 決定ログ |
| — | pretrain_xl_v1 単独(SFT前)チャットベンチ **未計測** |

> 重み(`.pt`)と原本コーパスは公開リポジトリに含めません。

---

<div align="center">

[README](README.ja.md) · [BENCHMARK v1(327M)](BENCHMARK-v1.ja.md) · [ARCHITECTURE](ARCHITECTURE.ja.md)

</div>
