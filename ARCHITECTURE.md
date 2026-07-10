<div align="center">

# 모델 아키텍처 · 학습 · 추론

<img src="https://img.shields.io/badge/CUDA-enabled-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />

**[한국어](ARCHITECTURE.md)** &nbsp;·&nbsp; [English](ARCHITECTURE.en.md) &nbsp;·&nbsp; [日本語](ARCHITECTURE.ja.md)

[← README.md](README.md) &nbsp;|&nbsp; 용어가 낯설면 [GLOSSARY.md](GLOSSARY.md)부터

</div>

---

GPT-2(초기 GPT) 이상급의 소형 범용 LLM을 직접 구현·학습·추론하고, `<THINKING>` 태그 기반 사고
모드까지 붙이는 과정을 다룬다.

## 목차

1. [전체 로드맵](#1-전체-로드맵)
2. [모델 아키텍처 설계](#2-모델-아키텍처-설계)
3. [토크나이저](#3-토크나이저)
4. [데이터셋 전략](#4-데이터셋-전략)
5. [학습 파이프라인](#5-학습-파이프라인)
6. [추론 엔진](#6-추론-엔진)
7. [평가](#7-평가)
8. [현실적 기대치와 확장 경로](#8-현실적-기대치와-확장-경로)

---

## 1. 전체 로드맵

| 단계 | 내용 | 산출물 |
|---|---|---|
| 0 | 환경 구축 (PyTorch + CUDA) | 개발 환경 |
| 1 | 토크나이저 (BPE) | `tokenizer.json` |
| 2 | 모델 아키텍처 (Decoder-only Transformer) | `model.py` |
| 3 | 사전학습 (Pretraining) | base 체크포인트 |
| 4 | 지도 미세조정 (SFT, 대화 형식) | chat 체크포인트 |
| 5 | 사고 모드 SFT (`<THINKING>` 데이터) | thinking 체크포인트 |
| 6 | 추론 엔진 (KV 캐시 + 샘플링 + 사고 모드 파싱) | `inference.py` |
| 7 | (배포 후) 인간 피드백 기반 후속 학습 | 정렬된 체크포인트 — [POST-TRAINING.md](POST-TRAINING.md) |

---

## 2. 모델 아키텍처 설계

### 2.1 기본 구조 — Decoder-only Transformer (모던 레시피)

GPT-2 원본보다 개선된 현대적 구성 요소를 쓴다 (LLaMA 계열 레시피). 각 구성 요소 이름이 낯설다면
[GLOSSARY.md](GLOSSARY.md#모델-구조-용어)에서 하나씩 찾아보면 된다.

```
Input Tokens
   │
Token Embedding (weight tying으로 출력층과 공유)
   │
[ Transformer Block ] × N
   ├─ RMSNorm (Pre-Norm)
   ├─ Self-Attention (Causal, GQA + RoPE + Flash Attention)
   ├─ Residual 연결
   ├─ RMSNorm
   ├─ FFN (SwiGLU)
   └─ Residual 연결
   │
RMSNorm (최종)
   │
LM Head (Linear → vocab logits)
```

| 구성요소 | GPT-2 (2019) | 본 설계 (권장) | 이유 |
|---|---|---|---|
| 정규화 | LayerNorm | **RMSNorm** | 더 단순, 학습 안정 |
| 위치 정보 | 학습형 절대 위치 | **RoPE** | 길이 확장이 비교적 용이 |
| 활성함수 | GELU | **SwiGLU** | 성능 향상 |
| 어텐션 | MHA | **GQA** (Grouped-Query) | KV 캐시 메모리 절감 |
| 바이어스 | 있음 | **제거** | 파라미터 절약, 안정 |

### 2.2 모델 크기 (GPU 예산별 3단계)

| 프리셋 | 파라미터 | layers | d_model | heads (Q/KV) | ctx | 필요 VRAM(학습) |
|---|---|---|---|---|---|---|
| **nano** (튜토리얼) | ~30M | 6 | 384 | 6 / 2 | 1024 | ~4GB |
| **small** (GPT-2급) | ~124M | 12 | 768 | 12 / 4 | 2048 | ~12GB |
| **base** (GPT-2 이상) | ~350M | 24 | 1024 | 16 / 4 | 4096 | ~24GB (또는 A100 클라우드) |

- vocab: **32,000** (BPE) — 한국어 포함 시 48k~64k 권장
- FFN hidden: `d_model × 8/3` (SwiGLU 기준, 예: 768 → 2048)

> 이 저장소의 실제 구현([llm/](llm/))은 **nano** 프리셋으로 학습·추론·피드백 루프까지 전부
> 동작을 확인했다. small·base로 키우는 전략은 [README.md의 §2](README.md#2-파라미터의-개수)를
> 그대로 따른다.

### 2.3 핵심 코드 스켈레톤 (PyTorch)

<details>
<summary><b>model.py 전체 코드 보기</b> (클릭해서 펼치기)</summary>

```python
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12          # Query heads
    n_kv_head: int = 4        # KV heads (GQA)
    d_model: int = 768
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def apply_rope(x, cos, sin):          # x: (B, H, T, Dh)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.dh = cfg.d_model // cfg.n_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.dh, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.dh, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.dh, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:                      # 추론용 KV 캐시
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[:] = [k, v]
        k = k.repeat_interleave(self.n_head // self.n_kv, dim=1)   # GQA 확장
        v = v.repeat_interleave(self.n_head // self.n_kv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.d_model * 8 / 3 / 64) * 64   # 64 배수 정렬
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)  # down
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.ffn  = RMSNorm(cfg.d_model), SwiGLU(cfg)
    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, kv_cache)
        return x + self.ffn(self.norm2(x))

class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying
        # RoPE 사전 계산 (cos/sin buffer) 생략 — register_buffer 사용

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        cos, sin = self.rope_cache(idx.shape[1])            # 구현 생략
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-100)
        return logits, loss
```

</details>

---

## 3. 토크나이저

- **BPE (Byte-level)** — HuggingFace `tokenizers` 라이브러리 사용
- 특수 토큰 설계 (사고 모드 핵심):

```
<|bos|>  <|eos|>  <|pad|>
<|user|>  <|assistant|>  <|system|>       ← 대화 역할
<THINKING>  </THINKING>                    ← 사고 모드 구간
```

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=[
    "<|bos|>","<|eos|>","<|pad|>","<|user|>","<|assistant|>","<|system|>",
    "<THINKING>","</THINKING>"])
tok.train(files=["corpus.txt"], trainer=trainer)
tok.save("tokenizer.json")
```

---

## 4. 데이터셋 전략

> **구현 상태**: 아래 전략에 맞춰 실제 한국어·영어·일본어 자연어 + 11개 언어 코드 데이터셋을
> 비-게이트(인증 불필요) 공개 소스에서 다운로드해 [llm/datasets/](llm/datasets/)에 정리해 두었다 —
> 총 11GB, 사전학습 텍스트 + SFT/THINKING 32만여 행. 전체 목록·출처·용량은
> [llm/datasets/README.md](llm/datasets/README.md) 참고.

### 4.1 사전학습 (Pretraining)

| 프리셋 | 데이터셋 | 토큰 수 | 비고 |
|---|---|---|---|
| nano | **TinyStories** | ~500M | 하루 안에 완주 가능, 문법적 생성 확인용 |
| small | **FineWeb-Edu (10B 샘플)** | 5~10B | GPT-2 재현급 품질 |
| base | FineWeb-Edu + **The Stack (smol)** | 10~30B | **코딩 성능**은 코드 데이터 비율 15~25%가 핵심 |
| 한국어 | + AI Hub / 모두의말뭉치 / ko-wikipedia | +α | 한국어 능력 필요 시 |

> **Chinchilla 법칙**([README.md의 §2](README.md#2-파라미터의-개수) 참고): 최적 토큰 수 ≈ 파라미터 × 20 (124M 모델 → 최소 2.5B 토큰).

### 4.2 SFT (대화 형식)

- **OpenHermes-2.5**, **UltraChat**, **KoAlpaca**(한국어) 등 공개 instruction 데이터
- 형식:

```
<|system|>You are a helpful assistant.<|eos|>
<|user|>파이썬으로 퀵소트 구현해줘<|eos|>
<|assistant|>...코드...<|eos|>
```

### 4.3 사고 모드 데이터 (핵심)

- **OpenThoughts / OpenR1-Math / Raiden-DeepSeek-R1** 등 공개 reasoning 트레이스 데이터를 `<THINKING>` 형식으로 변환
- 형식:

```
<|user|>3자리 소수 중 가장 큰 것은?<|eos|>
<|assistant|><THINKING>
3자리 최대 수는 999. 999=3×333 합성수. 998 짝수. 997을 확인:
√997≈31.6, 2~31 소수로 나눠보면... 나누어떨어지지 않음 → 소수.
</THINKING>
가장 큰 3자리 소수는 **997**입니다.<|eos|>
```

- **손실 마스킹**: user 토큰은 `-100` 처리(손실 제외), `<THINKING>` 내부를 포함한 assistant 토큰
  전체에 손실 적용 → 모델이 사고 과정을 스스로 생성하도록 학습

---

## 5. 학습 파이프라인

### 5.1 하이퍼파라미터 (small 기준)

| 항목 | 값 |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| LR 스케줄 | Warmup 2k step → Cosine decay |
| Peak LR | 6e-4 (pretrain) / 2e-5 (SFT) |
| Batch | 유효 배치 ~0.5M 토큰 (grad accumulation 활용) |
| 정밀도 | bfloat16 (mixed precision) |
| Grad clip | 1.0 |
| 기타 | `torch.compile`, Flash Attention (SDPA) |

### 5.2 학습 루프 스켈레톤

```python
model = GPT(cfg).cuda()
model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4,
                        betas=(0.9, 0.95), weight_decay=0.1)

for step, (x, y) in enumerate(loader):        # x,y: (B, T) uint16 memmap 권장
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x.cuda(), y.cuda())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    lr_scheduler.step()
    if step % 1000 == 0:
        save_checkpoint(model, opt, step)      # + val loss 로깅
```

### 5.3 단계별 순서

```
[1] Pretraining  : 원시 텍스트+코드, next-token prediction     (수일~수주)
[2] SFT          : 대화 형식, user 마스킹                        (수시간)
[3] Thinking SFT : <THINKING> 데이터 혼합 (일반:사고 = 3:7)      (수시간)
    ※ 2·3을 합쳐 한 번에 해도 됨. 사고 모드 on/off는
      system 프롬프트("think step by step in <THINKING>")로 제어 가능
[4] (선택) DPO/GRPO : 정답 검증 가능한 수학·코딩 문제로 강화학습 → 사고 품질 향상 (POST-TRAINING.md 참고)
```

---

## 6. 추론 엔진

### 6.1 생성 루프 (KV 캐시 + 사고 모드)

```python
@torch.no_grad()
def generate(model, tok, prompt, thinking=True,
             max_new=2048, temperature=0.7, top_p=0.9):
    sys = "You are a helpful assistant."
    if thinking:
        sys += " Reason step-by-step inside <THINKING> tags before answering."
    ids = tok.encode(f"<|system|>{sys}<|eos|><|user|>{prompt}<|eos|><|assistant|>")
    if thinking:
        ids += tok.encode("<THINKING>")        # 사고 모드 강제 시작(프리필)

    kv_caches = [[] for _ in model.blocks]     # 레이어별 KV 캐시
    x = torch.tensor([ids]).cuda()
    out = []
    for _ in range(max_new):
        logits = model.forward_with_cache(x, kv_caches)[:, -1]
        logits /= temperature
        probs = top_p_filter(F.softmax(logits, -1), top_p)
        next_id = torch.multinomial(probs, 1)
        if next_id.item() == tok.token_to_id("<|eos|>"):
            break
        out.append(next_id.item())
        x = next_id                             # 캐시 덕에 새 토큰 1개만 forward

    text = tok.decode(out)
    # 사고 구간 분리 → UI에서 접기/숨기기 처리
    thought, _, answer = text.partition("</THINKING>")
    return {"thinking": thought.strip(), "answer": answer.strip()}
```

### 6.2 사고 모드 동작 원리 정리

1. **학습**: `<THINKING>...</THINKING>` 이 포함된 assistant 응답으로 SFT → 모델이 "먼저 사고, 그다음 답변" 패턴을 내재화
2. **추론**: assistant 턴 시작 직후 `<THINKING>` 토큰을 프리필해 사고를 강제 시작 (off 모드는 프리필 생략 + system 프롬프트 변경)
3. **후처리**: `</THINKING>` 기준으로 사고/답변 분리, 멀티턴 히스토리에는 **답변만** 저장 (컨텍스트 절약)
4. **안전장치**: 사고 토큰 상한(예: 1024) 초과 시 `</THINKING>` 강제 삽입

---

## 7. 평가

| 능력 | 벤치마크 | 도구 |
|---|---|---|
| 언어 이해 | HellaSwag, ARC, MMLU(일부) | `lm-evaluation-harness` |
| 코딩 | **HumanEval, MBPP** | pass@1 채점 스크립트 |
| 수학·사고 | GSM8K | 사고 모드 on/off 비교 → 사고 모드 효과 검증 |
| 기본 | validation loss / perplexity | 자체 |

---

## 8. 현실적 기대치와 확장 경로

- **nano/small 자체 사전학습**: GPT-2 원본 수준의 유창한 생성 + 간단한 지시 수행까지는 된다. 본격
  코딩 능력은 기대하지 않는 게 좋다.
- **코딩 성능을 실제로 올리려면**: ① 코드 데이터 비율을 늘리고(The Stack), ② 모델을 350M~1B+로
  키우고, ③ 사고 모드 + 코딩 SFT 데이터(예: Magicoder-OSS-Instruct)를 더한다.
- **지름길(권장 트랙)**: 공개 소형 베이스 모델(Qwen2.5-0.5B/1.5B 등)을 받아
  [POST-TRAINING.md](POST-TRAINING.md)의 SFT→Thinking→추론 과정만 직접 수행하면, 같은 아키텍처
  지식을 배우면서도 GPT-2를 훨씬 뛰어넘는 실사용 품질을 확보할 수 있다.
- 참고 구현: `karpathy/nanoGPT`, `karpathy/build-nanogpt`, HuggingFace `smol-course`.

<div align="center">

---

[← GLOSSARY.md](GLOSSARY.md) &nbsp;|&nbsp; [README.md](README.md) &nbsp;|&nbsp; [POST-TRAINING.md →](POST-TRAINING.md)

</div>
