# POST-TRAINING

[![KO](https://img.shields.io/badge/KO-lightgrey)](POST-TRAINING.md) [![EN](https://img.shields.io/badge/EN-lightgrey)](POST-TRAINING.en.md) [![JA](https://img.shields.io/badge/JA-0969da)](POST-TRAINING.ja.md)

[← README](README.ja.md) · 用語で詰まったら [GLOSSARY](GLOSSARY.ja.md)

---

デプロイだけで終わると、モデルはそこで止まります。
人が「この質問にはこう答えるべきだ」と教えてくれれば、それを集めてまた学習できます。

この文書が扱うのは、その循環です。
ChatGPT 側の RLHF、オープンソース側の RLAIF・DPO のようなものを **小さな規模で** 移した版だと思ってください。

---

## 1. 全体の循環構造

```
                ┌─────────────────────────────────────────────┐
                │                                             │
   [デプロイ/推論]  →  [フィードバック収集]  →  [フィードバックデータセット]  →  [後続学習]
   infer.py        人が回答を評価          feedback.jsonl              3 つのトラック
   (ログ保存)      ① 訂正  ② 選好         preference.jsonl            (下記 §3)
                   ③ スコア                                            │
                │                                             ↓
                └────────────── 新チェックポイントを配布 ←──────────┘
                                (1 サイクル = 1 iteration、繰り返し)
```

一回で終わりの構造ではありません。
収集 → 学習 → デプロイ → 再収集を繰り返し、前サイクルのデータも積み上げてまた学習します。

---

## 2. フィードバック収集の設計

### 2.1 推論ログ

`infer.py` が会話を自動で記録します。

```json
// logs/chat_log.jsonl — 1 行 = 1 ターン
{"ts": "2026-07-07T12:00:00", "user": "2+3*4=?", "thinking": "...", "answer": "165.", "ckpt": "sft_nano.pt"}
```

### 2.2 人間フィードバック 3 種

| 種類 | 人がやること | データ形式 | 使いどころ |
|---|---|---|---|
| **① 訂正 (Correction)** | 間違った答を見て「こう答えるべきだ」と正解を直接書く | `{user, bad_answer, corrected_answer, corrected_thinking}` | 訂正 SFT |
| **② 選好 (Preference)** | 同じ質問への答 2 つから良い方を選ぶ | `{user, chosen, rejected}` | DPO / 報酬モデル |
| **③ スコア (Rating)** | 答に 1~5 点 (👍/👎 でも可) | `{user, answer, score}` | 報酬モデル、フィルタリング |

② は temperature だけ変えて 2 回生成すれば、ペアを作りやすいです。
① の訂正に `<THINKING>` 思考過程まで一緒に書くと、思考品質も一緒に上がります。

```json
// data/feedback.jsonl (訂正)
{"user": "2 + 3 * 4 = ?", "bad_answer": "165.",
 "corrected_thinking": "掛け算が先。3*4=12、2+12=14。", "corrected_answer": "答えは14です。"}

// data/preference.jsonl (選好ペア)
{"user": "クイックソートをPythonで実装して。",
 "chosen":   "def quicksort(a): ...(正しい実装)",
 "rejected": "sort()を使ってください。"}
```

### 2.3 ラベリング UI

- チュートリアル: **CLI** — 答の直後に `[g]ood / [b]ad / [c]orrect / [p]refer` → jsonl append
- 拡張: Gradio/Streamlit で A/B 画面
- 実サービス: ユーザー 👍/👎 (暗黙) + ラベラーの訂正 (明示) を併用

---

## 3. 後続学習 3 トラック (易しい順)

### トラック A — 訂正 SFT (いちばん単純、ここから)

人間が書いた訂正答をそのまま SFT データに変えて再学習します。

```
feedback.jsonl → {user, thinking: corrected_thinking, assistant: corrected_answer}
              → data.py sft 形式と同じ → train.py --mode sft --init ckpt/sft_nano.pt
```

- 既存パイプラインそのまま。新しいコードは要りません。
- 既存 SFT : 訂正 = **1 : 3** で混ぜます (訂正側に重み)。

> [!WARNING]
> そうしないと既存能力を忘れる catastrophic forgetting が起きます。

- スコア ③ のうち 4~5 点だけを選んで SFT に昇格 (rejection sampling)。

### トラック B — DPO (この設計のメイン)

選好ペア ② で、報酬モデルなしにそのまま整列します。
RLHF より実装・安定性がずっと楽なので、チュートリアルに向いています。

原理は単純です。chosen の確率は上げ、rejected は下げつつ、参照モデルから遠く離れすぎないようにします。

```
L_DPO = -log σ( β·[ (logπ_θ(chosen) - logπ_ref(chosen))
                  - (logπ_θ(rejected) - logπ_ref(rejected)) ] )
```

```python
# train.py に --mode dpo を追加 (スケルトン)
def seq_logprob(model, ids, labels):
    """シーケンスの応答部分のログ確率の合計 (labels=-100 位置は除く)。"""
    logits, _ = model(ids[:, :-1])                      # (B,T,V)
    logp = F.log_softmax(logits.float(), -1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    picked = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)                      # (B,)

def dpo_loss(policy, ref, batch, beta=0.1):
    pc = seq_logprob(policy, *batch["chosen"])
    pr = seq_logprob(policy, *batch["rejected"])
    with torch.no_grad():                               # 参照モデルは固定
        rc = seq_logprob(ref, *batch["chosen"])
        rr = seq_logprob(ref, *batch["rejected"])
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
```

| ハイパーパラメータ | 推奨値 |
|---|---|
| β (KL 強度) | 0.1 (0.05~0.5 で探索) |
| LR | 5e-7 ~ 1e-6 (SFT よりずっと低く) |
| Epoch | 1~3 (過学習注意 — chosen 確率だけ上がっていないか監視) |
| 参照モデル | 学習開始時点のチェックポイントのコピー (VRAM 不足時は ref logprob を事前キャッシュ) |

### トラック C — 報酬モデル + RL (拡張)

**報酬モデル (RM)**: GPT バックボーンを使い、LM head だけをスカラー head に差し替えます。

```python
class RewardModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = GPT(cfg)                        # SFT チェックポイントで初期化
        self.backbone.lm_head = nn.Identity()
        self.v_head = nn.Linear(cfg.d_model, 1, bias=False)
    def forward(self, ids):                             # 最後のトークン hidden → スコア
        h = self.backbone.hidden(ids)                   # (B,T,D)
        return self.v_head(h[:, -1]).squeeze(-1)        # (B,)

# 学習: 選好ペアのランキング損失
loss = -F.logsigmoid(rm(chosen) - rm(rejected)).mean()
```

**RL 段階 — GRPO 推奨** (PPO より単純、value モデル不要):

```
プロンプトごとに G 個の回答をサンプリング → RM スコア → グループ平均に対する advantage
A_i = (r_i - mean(r)) / std(r)
→ advantage 加重 policy gradient + KL(π_θ ‖ π_ref) ペナルティ
```

- 数学・コーディングのように **正解を自動採点** できるなら、RM の代わりにルール報酬  
  (正解一致 +1、テスト通過 +1) → **RLVR**。`<THINKING>` 品質が本当に上がる区間はここです。
- 報酬ハッキング監視: RM スコアは上がるのに人が見ると悪くなっていたら即中断。

---

## 4. トラックの選び方

```
フィードバックが「正解を書く」形          → トラック A (訂正 SFT)          ← ここから
フィードバックが「2 択から選ぶ」形         → トラック B (DPO)               ← メイン
自動採点可能 (数学・コーディング) / 大規模  → トラック C (RM+GRPO / RLVR)    ← 拡張
実戦の勧め: A で土台 → B を繰り返す → (余力があれば) C
```

---

## 5. 安全装置と評価

| 項目 | 設計 |
|---|---|
| **忘却防止** | 後続学習のたびに固定回帰セットの loss を測定 — 5% 以上悪化したらロールバック |
| **KL ガード** | DPO の β / GRPO の KL ペナルティで参照モデルとの距離を制限 |
| **昇格ゲート** | (1) 回帰セット通過 (2) ホールドアウト選好ペア win-rate > 55% のときだけデプロイ |
| **データ衛生** | 同一プロンプトの重複除去、訂正↔選好ラベルの衝突検収、ラベラー一致度 |
| **バージョン管理** | `ckpt/{mode}_{preset}_v{n}.pt` + 学習データのスナップショットを一緒に保管 |

トラック A (混合 SFT)・B (DPO)・C (RM+GRPO/RLVR) は実装済みです。
GRPO 損失は `mean_token(-A·logπ_θ + β·KL_k3(π_ref‖π_θ))`、advantage は `A=(r-mean)/std`、  
RLVR は形式報酬 + 正解一致。実行: `python rlhf.py --mode {rm,rlvr}`。

---

## 6. 1 サイクルの実行の流れ

トラック A+B 基準のスケッチです。正確なフラグは [README §5](README.ja.md#5-実行方法) に従います。

```bash
python infer.py --ckpt ckpt/sft_nano.pt --log          # 1. デプロイ・会話ログ
python feedback.py review                              # 2. 人がレビュー・ラベリング
python data.py sft --input data/feedback.jsonl         # 3a. 訂正 → SFT データ
python train.py --mode sft --init ckpt/sft_nano.pt     # 4a. トラック A
python train.py --mode dpo --init ckpt/sft_nano.pt \
                --data data/preference.jsonl           # 4b. トラック B
python eval_gate.py --old ckpt/sft_nano.pt --new ckpt/dpo_nano.pt  # 5. 昇格ゲート
# 通過したら新チェックポイントで 1 から繰り返し
```

---

[ARCHITECTURE](ARCHITECTURE.ja.md) · [README](README.ja.md) · [GLOSSARY](GLOSSARY.ja.md) · [BENCHMARK](BENCHMARK-v1.ja.md)
