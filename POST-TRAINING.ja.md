<div align="center">

# 後続学習(Post-Training) — 人間フィードバックによるアライメント

<img src="https://img.shields.io/badge/RLHF-DPO%20%7C%20GRPO%20%7C%20RLVR-76B900?style=flat" alt="RLHF" />

[한국어](POST-TRAINING.md) &nbsp;·&nbsp; [English](POST-TRAINING.en.md) &nbsp;·&nbsp; **[日本語](POST-TRAINING.ja.md)**

[← README.md](README.ja.md) &nbsp;|&nbsp; 用語に馴染みがなければ[GLOSSARY.md](GLOSSARY.ja.md)から

</div>

---

デプロイ済みモデルの回答を人間がチェックして、「この質問にはこう答えるべきだ」というフィードバックを
集め、それを元にモデルを直し続ける循環パイプラインの話だ。ChatGPTが使うRLHF、オープンソース系でよく
使われるRLAIF・DPOと同じ構造を、チュートリアル規模に縮めて実装した。

## 目次

1. [全体循環構造 (Feedback Loop)](#1-全体循環構造-feedback-loop)
2. [フィードバック収集設計](#2-フィードバック収集設計)
3. [後続学習 3トラック (難易度順)](#3-後続学習-3トラック-難易度順)
4. [トラック選択ガイド](#4-トラック選択ガイド)
5. [安全装置と評価](#5-安全装置と評価)
6. [1サイクルの実行フロー](#6-1サイクルの実行フロー)

---

## 1. 全体循環構造 (Feedback Loop)

```
                ┌─────────────────────────────────────────────┐
                │                                             │
   [デプロイ/推論]  →  [フィードバック収集]  →  [フィードバックデータセット]  →  [後続学習]
   infer.py        人が回答を評価          feedback.jsonl              3つのトラック
   (ログ保存)      ① 修正 ② 選好         preference.jsonl            (下記§3)
                   ③ スコア                                            │
                │                                             ↓
                └────────────── 新チェックポイントを配布 ←──────────┘
                                (1サイクル = 1 iteration、繰り返す)
```

ここで大事なのは、これを一回で終わらせないということだ。収集→学習→デプロイ→再収集をひたすら繰り返し、
サイクルが回るたびに前サイクルのデータを積み上げてまた学習し直す。

## 2. フィードバック収集設計

### 2.1 推論ログ (収集の出発点)

`infer.py`があらゆる会話を自動で記録する。

```json
// logs/chat_log.jsonl — 1行 = 1ターン
{"ts": "2026-07-07T12:00:00", "user": "2+3*4=?", "thinking": "...", "answer": "165.", "ckpt": "sft_nano.pt"}
```

### 2.2 人間フィードバック3種 (ラベリング形式)

| 種類 | 人がやること | データ形式 | 使いどころ |
|---|---|---|---|
| **①修正 (Correction)** | 間違った回答を見て「本当はこう答えるべきだ」という正解を直接書く | `{user, bad_answer, corrected_answer, corrected_thinking}` | 修正SFT |
| **②選好 (Preference)** | 同じ質問に対する回答2つのうち、良い方を選ぶ | `{user, chosen, rejected}` | DPO / 報酬モデル |
| **③スコア (Rating)** | 回答に1~5点をつける(👍/👎でもOK) | `{user, answer, score}` | 報酬モデル、フィルタリング |

②は同じプロンプトをtemperatureだけ変えて2回生成させれば、ペアを作るのは簡単だ。①の修正回答は
`<THINKING>`の思考過程まで一緒に書いてあげると、思考の質も一緒に良くなる。

```json
// data/feedback.jsonl (修正)
{"user": "2 + 3 * 4 = ?", "bad_answer": "165.",
 "corrected_thinking": "掛け算が先。3*4=12、2+12=14。", "corrected_answer": "答えは14です。"}

// data/preference.jsonl (選好ペア)
{"user": "Pythonでクイックソートを実装して。",
 "chosen":   "def quicksort(a): ...(正しい実装)",
 "rejected": "sort()を使ってください。"}
```

### 2.3 ラベリングUI (最小構成)

- チュートリアル: **CLI** — `infer.py`の回答直後に`[g]ood / [b]ad / [c]orrect / [p]refer`キーを入力 → jsonlにappend
- 拡張: 簡単なウェブUI(Gradio/Streamlit)でA/B比較画面を用意
- 実サービス構成: ユーザーの👍/👎(暗黙のフィードバック) + 専門ラベラーによる修正(明示的フィードバック)を併用する

## 3. 後続学習 3トラック (難易度順)

### トラックA — 修正SFT (Rejection-sampling SFT、最もシンプルで最初の一歩)

人間が書いた修正回答をそのままSFTデータに変換して再学習する。

```
feedback.jsonl → {user, thinking: corrected_thinking, assistant: corrected_answer}
              → data.py sftと同じ形式 → train.py --mode sft --init ckpt/sft_nano.pt
```

- 既存のパイプラインをそのまま再利用する。新しいコードは要らない。
- 既存のSFTデータと修正データを**1 : 3**の比率で混ぜる(修正側に重みをかける) — 既存の能力を忘れてしまう
  現象(catastrophic forgetting)を防ぐためだ。
- ③スコアのデータは4~5点の回答だけを選んでSFTデータに昇格させる(rejection sampling)。

### トラックB — DPO (Direct Preference Optimization、この設計のメイン)

選好ペア②を使って、報酬モデルなしで直接アライメントする。RLHFより実装も安定性も圧倒的に楽なので、
チュートリアルには一番向いている。

原理はシンプルだ。chosenの確率は上げてrejectedの確率は下げつつ、元の(参照)モデルからあまり
離れすぎないようにする。

```
L_DPO = -log σ( β·[ (logπ_θ(chosen) - logπ_ref(chosen))
                  - (logπ_θ(rejected) - logπ_ref(rejected)) ] )
```

```python
# train.pyに--mode dpoを追加(スケルトン)
def seq_logprob(model, ids, labels):
    """シーケンスの応答部分のログ確率の合計 (labels=-100の位置は除く)."""
    logits, _ = model(ids[:, :-1])                      # (B,T,V) — 全logitsが必要
    logp = F.log_softmax(logits.float(), -1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    picked = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)                      # (B,)

def dpo_loss(policy, ref, batch, beta=0.1):
    # batch: chosen/rejectedそれぞれ (ids, labels) — SFTと同じマスキング規則
    pc = seq_logprob(policy, *batch["chosen"])
    pr = seq_logprob(policy, *batch["rejected"])
    with torch.no_grad():                               # 参照モデルは固定(frozen)
        rc = seq_logprob(ref, *batch["chosen"])
        rr = seq_logprob(ref, *batch["rejected"])
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
```

| ハイパーパラメータ | 推奨値 |
|---|---|
| β (KL強度) | 0.1 (0.05~0.5で探索) |
| LR | 5e-7 ~ 1e-6 (SFTよりずっと低く) |
| Epoch | 1~3 (過学習注意 — chosen確率だけが上がっていないか監視) |
| 参照モデル | 学習開始時点のチェックポイントのコピー (VRAM不足時は事前にref logprobをキャッシュ) |

### トラックC — 報酬モデル + RL (RLHF/GRPO、拡張フェーズ)

**報酬モデル(RM)**: 既存のGPTバックボーンをそのまま使い、LM headだけをスカラーheadに置き換える。

```python
class RewardModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = GPT(cfg)                        # SFTチェックポイントで初期化
        self.backbone.lm_head = nn.Identity()
        self.v_head = nn.Linear(cfg.d_model, 1, bias=False)
    def forward(self, ids):                             # 最後のトークンのhidden → スコア
        h = self.backbone.hidden(ids)                   # (B,T,D)
        return self.v_head(h[:, -1]).squeeze(-1)        # (B,)

# 学習: 選好ペアのランキング損失
loss = -F.logsigmoid(rm(chosen) - rm(rejected)).mean()
```

**RL段階 — GRPOを推奨** (PPOよりシンプルで、valueモデルが要らない):

```
プロンプトごとにG個の回答をサンプリング → RMスコア → グループ平均に対するadvantage
A_i = (r_i - mean(r)) / std(r)
→ advantage加重policy gradient + KL(π_θ ‖ π_ref)ペナルティ
```

- 数学・コーディングのように**正解を自動で採点できる領域**では、RMの代わりにルールベースの報酬
  (正解一致+1、コードはテスト通過+1)を使う → RLVR。`<THINKING>`の思考品質が実際に良くなるのは
  まさにこの区間だ。
- 報酬ハッキング(reward hacking)の監視が必要だ — RMのスコアは上がっているのに人が見ると悪化して
  いる場合は即中断。

## 4. トラック選択ガイド

```
フィードバックが「正解を書く」形式        → トラックA (修正SFT)         ← まずここから
フィードバックが「2択から選ぶ」形式       → トラックB (DPO)             ← メイン
自動採点可能(数学・コーディング) / 大規模  → トラックC (RM+GRPO / RLVR)   ← 拡張
実践的な推奨コンボ: Aで土台を固める → Bを繰り返す → (余力があれば) C
```

## 5. 安全装置と評価

| 項目 | 設計 |
|---|---|
| **忘却防止** | 後続学習のたびに固定回帰セット(既存SFT検証セット)のlossを測定 — 5%以上悪化したらロールバック |
| **KLガード** | DPOのβ / GRPOのKLペナルティで参照モデルとの距離を制限 |
| **昇格ゲート** | 新チェックポイントは (1) 回帰セット通過 (2) ホールドアウト選好ペアのwin-rate > 55% のときだけデプロイ |
| **データ衛生** | 同一プロンプトの重複除去、修正↔選好ラベルの矛盾チェック、ラベラー間の一致度確認 |
| **バージョン管理** | `ckpt/{mode}_{preset}_v{n}.pt` + 学習に使ったデータのスナップショットを一緒に保管 |

> **実装状況**: トラックA(混合SFT)・B(DPO)・C(RM+GRPO/RLVR)は全部実装済みだ。GRPO損失は
> `mean_token(-A·logπ_θ + β·KL_k3(π_ref‖π_θ))`、advantageは`A=(r-mean)/std`、RLVR報酬は
> 形式報酬 + 正解一致で計算する。`python rlhf.py --mode {rm,rlvr}`で実行する。

## 6. 1サイクルの実行フロー

トラックA・Bを例に、後続学習の1サイクルを最初はこんな感じでスケッチしてみた。実際の実装で使う正確な
コマンド・フラグは[README.mdの§5 実行方法](README.ja.md#5-実行方法)に従う。

```bash
python infer.py --ckpt ckpt/sft_nano.pt --log          # 1. デプロイ・会話ログ収集
python feedback.py review                              # 2. 人が回答をレビュー・ラベリング
python data.py sft --input data/feedback.jsonl         # 3a. 修正 → SFTデータ
python train.py --mode sft --init ckpt/sft_nano.pt     # 4a. トラックA学習
python train.py --mode dpo --init ckpt/sft_nano.pt \
                --data data/preference.jsonl           # 4b. トラックB学習
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano.pt  # 5. 昇格ゲート
# 通過したら新チェックポイントで1番から繰り返す
```

<div align="center">

---

[← ARCHITECTURE.md](ARCHITECTURE.ja.md) &nbsp;|&nbsp; [README.md](README.ja.md) &nbsp;|&nbsp; [GLOSSARY.md](GLOSSARY.ja.md)

</div>
