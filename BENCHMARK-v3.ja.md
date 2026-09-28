[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](BENCHMARK-v3.md) [![English](https://img.shields.io/badge/English-8B949E?style=flat-square)](BENCHMARK-v3.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-0969DA?style=flat-square)](BENCHMARK-v3.ja.md)

[← README](README.ja.md)

---

# 📊 BENCHMARK v3 · APEX-2（MoE 3.87B · アクティブ 1.45B）

> APEX-1(1.1B dense)とは別系列です。**MoE(Mixture-of-Experts)** 構造でゼロから事前学習したモデルで、コードは別リポジトリ(GeneralLLM)で書きました。

| | |
|:--|:--|
| 🆕 **最終** | `sft2_chat`(SFT 第2段階 step 536)· 2026-09-28 |
| 📦 **モデル** | APEX-2 · Decoder-only MoE · 実測 **3,869.1M**(トークンあたりアクティブ **1,453.2M**) |
| 🤗 **公開** | [huggingface.co/YOON1v/Apex-2](https://huggingface.co/YOON1v/Apex-2) — `transformers`・vLLM でそのままロード(Qwen3MoeForCausalLM マッピング) |
| 🧪 **評価** | コード5種 · 数学2種 · IFEval · MMLU · 常識6種(vLLM greedy、チャットテンプレート) |
| 📁 **原本** | [`ckpt/benchmark_sft_Apex-2_v1.json`](ckpt/benchmark_sft_Apex-2_v1.json) · [`ckpt/benchmark_dpo_Apex-2_v1.json`](ckpt/benchmark_dpo_Apex-2_v1.json) |
| ✅ **結論** | Pretrain → SFT 完了。DPO は性能が下がったため廃止(§7) |

### 目次

1. [ひと目で](#1-ひと目で)
2. [モデルカード](#2-モデルカード)
3. [学習過程](#3-学習過程)
4. [データセット](#4-データセット)
5. [採点方法](#5-採点方法)
6. [同規模モデルとの比較](#6-同規模モデルとの比較)
7. [DPO · RLVR 記録](#7-dpo--rlvr-記録)
8. [制限](#8-制限)

---

## 1. ひと目で

| 分野 | ベンチマーク | base | **SFT(最終)** | DPO(廃止) |
|:--|:--|--:|--:|--:|
| コード | HumanEval / HumanEval+ | 36.6 / 32.9 | **43.9 / 41.5** | 36.0 / 32.3 |
| コード | MBPP / MBPP+ | 54.8 / 46.3 | **56.3 / 48.9** | 45.2 / 37.3 |
| コード | MultiPL-E HumanEval C++ / MBPP C++ | — | **36.0 / 41.6** | 8.1 / 3.8 |
| コード | LiveCodeBench v5–v6(easy / medium / hard) | — | 3.2 (11.9 / 1.0 / 0.0) | 3.5 |
| コード | CRUXEval-O / -I | — | 8.5 / 3.2 | 12.1 / 8.9 |
| 数学 | GSM8K | 15.1 (8-shot) | **32.4** (0-shot CoT) | 14.3 |
| 数学 | MATH-500(0-shot CoT) | — | **21.0** | 14.8 |
| 指示追従 | IFEval prompt / instruction strict | — | **44.7 / 56.6** | 35.7 / 49.2 |
| 知識 | MMLU(5-shot) | 28.2 | 28.6 | 28.3 |
| 常識 | HellaSwag / ARC-e / ARC-c(acc_norm) | 60.8 / 66.7 / 39.7 | 62.2 / 64.1 / 38.4 | 63.1 / 64.6 / 39.2 |
| 常識 | PIQA(acc_norm)/ WinoGrande / LAMBADA | 73.9 / 60.1 / 54.5 | 74.8 / 61.8 / 52.8 | 74.4 / 63.3 / 51.2 |

- SFT でコード・数学・指示追従が上がり、知識・常識は事前学習の水準のままです(±2 以内)。
- base の GSM8K は 8-shot なので、SFT の 0-shot CoT とは方式が異なります。

## 2. モデルカード

| 項目 | 値 |
|:--|:--|
| 構造 | Decoder-only Transformer、pre-norm、**全層が MoE**(dense 層なし) |
| 層 · d_model | 32 · 2,048 |
| アテンション | GQA 16Q / 4KV、head_dim 128、QK-RMSNorm、bias なし |
| 位置 | RoPE θ = 10,000 · コンテキスト 4,096(文書境界マスキング) |
| FFN | SwiGLU MoE: expert 16個、トークンあたり **top-4**、expert 幅 1,024(トークンあたり FFN 幅 4,096) |
| ルーター | Linear 2048→16、fp32 softmax → top-4 → 確率を再正規化 |
| 負荷分散 | aux loss 0.01(グローバルバッチ基準)+ router z-loss 1e-3 |
| 語彙 | 151,936(Qwen3 トークナイザー)、埋め込み–lm_head 共有 |
| HF 形式 | `Qwen3MoeForCausalLM` に 1:1 マッピング(export 検証: logits 相対差 1.0e-6、次トークン予測 100% 一致) |

| 構成 | パラメータ |
|:--|--:|
| アテンション | 335.6M |
| expert(32 × 16 × 3 × 2048 × 1024) | 3,221.2M |
| ルーター · norm | 1.2M |
| 埋め込み(= lm_head) | 311.2M |
| **合計** | **3,869,124,608** |
| **トークンあたりアクティブ** | **1,453,205,504**(非埋め込み 1,142,040,576) |

## 3. 学習過程

| 段階 | トークン(累積) | 設定 |
|:--|:--|:--|
| 事前学習 1 | 20.0B (20.0B) | GH200 1枚、LR 4e-4(warmup 1,000 step) |
| 事前学習 2(CPT) | 46.1B (66.1B) | GH200 1枚 → **GH200 2枚 DiLoCo**(step 26,000 から、250 step ごとに重み平均)、LR 4e-4 維持 |
| 事前学習 3(decay) | 20.4B (**86.5B**) | GH200 2枚 DiLoCo、LR 4e-4 → 0 線形、9,750 step |
| SFT 第1段階 | 2.21B(1 epoch) | H100 1枚、LR 2e-5 → 0、2,130 step |
| SFT 第2段階 | 0.28B × 2 epoch | LR 1e-5 → 0、536 step |

- global batch 1,048,576 トークン(256 × 4,096)、8-bit AdamW(β 0.9/0.95)、weight decay 0.1、grad clip 1.0、bf16 演算 + fp32 master。
- 事前学習の合計は 54,250 step · 86,507,520,000 トークン。
- SFT 検証 loss: 第1段階 0.627 → 0.577、第2段階 0.642 → 0.635(2 epoch 目でも過学習なし)。

## 4. データセット

**事前学習**(重複除去済みの保有コーパス 40 ソース)

| 区間 | ミックス |
|:--|:--|
| 事前学習 1 | Web 62 / コード 22 / 数学 8 / キュレーション 8(FineWeb-Edu、DCLM、StarCoder 系、FineMath、Wikipedia、StackExchange など) |
| 事前学習 2 | 保有コーパス全量を 1 回(Web · コード 11 言語 · 数学 · 論文 · StackExchange · 合成) |
| decay | Web 30 / コード 35 / 数学 15 / キュレーション 20 |

**SFT**(ChatML、assistant の回答にのみ loss、評価セットと 13-gram で汚染除去)

| 段階 | 構成 |
|:--|:--|
| 第1段階 2.21B(413万会話) | 汎用 29%(Dolci-Instruct)/ コード 44%(OpenCoder SFT、OpenCodeInstruct 通過率 ≥ 0.9、self-oss、Code-Feedback)/ 数学 27%(OpenMathInstruct-2、NuminaMath) |
| 第2段階 0.28B | 実行検証済みコード 59% / NuminaMath 21% / 精密な指示追従 20% |

## 5. 採点方法

- 生成: vLLM greedy、チャットテンプレート、プロンプト + 回答 ≤ 4,096 トークン。
- モデルが書いたコードは bwrap サンドボックス(ネットワーク遮断、`/` 読み取り専用)で実行して採点。正解を入れた自己検査で採点器を先に検証。

| ベンチマーク | 方式 |
|:--|:--|
| HumanEval(+) · MBPP(+) | EvalPlus 0.3.1、チャットプロンプト |
| LiveCodeBench | code_generation_lite release v5·v6 の追加分(2024-08 以降の 342 問)、公式プロンプト、pass@1 |
| CRUXEval-O / -I | CoT の後 `[ANSWER]assert …[/ANSWER]`、assert を実行 |
| MultiPL-E C++ | 関数全体を受け取りテストと合わせて `g++ -std=c++17` |
| GSM8K · MATH-500 | 0-shot CoT、`\boxed{}`(MATH は math-verify) |
| IFEval | lm-eval 0.4.13、チャットテンプレート |
| MMLU · 常識 6種 | lm-eval loglikelihood(チャットテンプレートなし)、base と同じ方式 |

## 6. 同規模モデルとの比較

他モデルのスコアは公式モデルカード・技術報告の値です。few-shot 数、CoT、IFEval 指標、LiveCodeBench の期間がモデルごとに異なるため、**おおよその比較**としてご覧ください。

| モデル | 事前学習トークン | HE+ | MBPP+ | GSM8K | IFEval | MMLU |
|:--|--:|--:|--:|--:|--:|--:|
| **Apex-2 SFT(3.87B · アクティブ 1.45B)** | **0.087T** | **41.5** | **48.9** | **32.4** | **44.7** | **28.6** |
| Qwen2.5-1.5B-Instruct | 18T | (HE 61.6) | (MBPP 63.2) | 73.2 | 42.5 | 50.7 |
| Qwen2.5-Coder-1.5B-Instruct | 5.5T | 66.5 | 59.4 | — | — | — |
| Qwen3-1.7B(non-thinking) | 36T | — | — | — | 68.2 | 64.4 |
| Llama-3.2-1B-Instruct | 9T | — | — | 44.4 | 59.5* | 49.3 |
| Gemma-3-1B-it | 2T | (HE 41.5) | (MBPP 35.2) | 62.8 | 80.2* | 38.8 |
| OLMoE-1B-7B(アクティブ 1.3B) | 5.1T | 54.4 | — | 72.4 | 66.4* | 55.1 |
| DeepSeek-Coder-1.3B | 2T | 60.4 | 54.8 | — | — | — |

<sub>括弧内は HumanEval/MBPP。\* IFEval の指標が異なる(Apex-2・Qwen: prompt-level strict)。Qwen2.5・Qwen3 の MMLU は MMLU-Redux、OLMoE の MMLU は CoT。</sub>

**base 同士の比較**(事前学習直後)

| モデル | 事前学習トークン | HE / HE+ | MBPP / MBPP+ | MMLU | HellaSwag | PIQA | WinoGrande |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Apex-2 base** | **0.087T** | 36.6 / 32.9 | 54.8 / 46.3 | 28.2 | 60.8 | 73.9 | 60.1 |
| Qwen2.5-1.5B | 18T | 37.2 / 32.9 | 60.2 / 49.6 | 60.9 | 67.9 | — | 65.0 |
| Qwen2.5-Coder-1.5B | 5.5T | 43.9 / 36.6 | 69.2 / 58.6 | 53.6 | 61.8 | — | 60.7 |
| Gemma-2-2B | 2T | 17.7 / — | 29.6 / — | 51.3 | 73.0 | 77.8 | 70.9 |
| OLMoE-1B-7B | 5.1T | — | — | 56.3 | 81.7 | 78.7 | 70.6 |

- 事前学習データは比較対象の 1/23〜1/400 ですが、base のコードスコアは Qwen2.5-1.5B(18T)と同水準です(HE+ 32.9)。
- 差が大きいのは知識(MMLU)と数学で、どちらも事前学習トークン数に左右されます。

<sub>出典: Qwen2.5 [blog](https://qwenlm.github.io/blog/qwen2.5-llm/) · [report](https://arxiv.org/abs/2412.15115) · Qwen2.5-Coder [report](https://arxiv.org/abs/2409.12186) · Qwen3 [report](https://arxiv.org/abs/2505.09388) · [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) · Gemma 3 [report](https://arxiv.org/abs/2503.19786) · [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b-it) · [OLMoE Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) · [OLMoE base](https://huggingface.co/allenai/OLMoE-1B-7B-0125) · DeepSeek-Coder [paper](https://arxiv.org/abs/2401.14196)</sub>

## 7. DPO · RLVR 記録

**DPO — 廃止**
- データ: allenai/Dolci-Instruct-DPO 22万ペア(off-policy)、長さ正規化 DPO(β 5)、LR 1e-6。
- 検証精度は step 100 で 81% とほぼ飽和したのに、chosen NLL が 1.023 → 1.161 と上がり続けました(chosen まで確率が下がる likelihood displacement)。step 300 で停止しました。
- 評価では平均回答長が 320 → 733 トークン(2.3 倍)に伸び、長さ上限に達する回答が大きく増え、コード・数学・指示追従がすべて下がりました(§1 の表)。

**RLVR(GSPO)— 中断**
- コーパス: OpenCodeInstruct(単体テスト)· GSM8K · MATH · DeepMath · IFBench 学習セット(すべて自由ライセンス)、報酬はサンドボックス実行 · math-verify · IFEval 制約。
- DPO モデルから開始しましたが、DPO の結果を見て step 33 で止めました。

## 8. 制限

- 英語中心。韓国語などほかの言語はほとんどできません(SFT で多言語を除外)。
- 知識が乏しく、事実誤り(ハルシネーション)が多いです(MMLU 28.6)。
- 競技プログラミング(LiveCodeBench medium 1%)とコード実行推論(CRUXEval)が弱いです。
- コンテキスト 4,096 トークン。
