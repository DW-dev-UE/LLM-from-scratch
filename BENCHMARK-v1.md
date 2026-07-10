<div align="center">

# Benchmark Report · Base V1

**학습은 어떻게 했고, 데이터는 뭘 썼으며, 질문에 뭐라고 답했는가**

<br/>

![ckpt](https://img.shields.io/badge/sft__base__v1-326.7M-3b82f6?style=flat-square)
![pretrain](https://img.shields.io/badge/pretrain-18k%20steps-22c55e?style=flat-square)
![sft](https://img.shields.io/badge/SFT-5k%20steps-22c55e?style=flat-square)
![status](https://img.shields.io/badge/status-early%20stage-f59e0b?style=flat-square)

<br/>

[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.md)

</div>

---

### 먼저 한 줄로

| | 결과 |
|:--|:-----|
| **Pretrain only** | 채팅 프롬프트에 거의 반응 못 함 (28문항 0점 수준) |
| **SFT 이후** | 지시 형식은 따라 하려 함. 내용은 자주 틀림 |
| **THINKING 모드** | 답변이 비는 버그 (14문항 중 답 0개) → 일반 채팅 모드로 평가 |

아직 실사용 단계는 아닙니다.  
다만 **파이프라인이 실제로 돌아가는지**, SFT가 무엇을 바꾸는지 보여 주는 첫 스냅샷입니다.

---

## 목차

1. [학습 과정](#1-학습-과정)  
2. [데이터셋](#2-데이터셋)  
3. [어떻게 채점했나](#3-어떻게-채점했나)  
4. [점수 요약](#4-점수-요약)  
5. [Pretrain 과 SFT 비교](#5-pretrain-과-sft-비교)  
6. [질문과 답변](#6-질문과-답변)  
7. [코딩 문항](#7-코딩-문항)  
8. [느낀 점 · 다음에 할 일](#8-느낀-점--다음에-할-일)

---

## 1. 학습 과정

### 모델 한 장 요약

| 항목 | 값 |
|:-----|:---|
| 계열 | Decoder-only Transformer (LLaMA 스타일 구성) |
| 파라미터 | 약 **326.7M** (`base` 프리셋) |
| 깊이 · 폭 | 24 layer · d_model 1024 |
| Attention | GQA (Q 16 · KV 4) + RoPE |
| FFN | SwiGLU |
| 정규화 | RMSNorm (Pre-Norm) |
| 기타 | weight tying, bias 없음 |

```text
입력 토큰
   │
임베딩 ──────────────────────────────┐
   │                                 │ (weight tying)
[ Block × 24 ]                       │
  RMSNorm → Attention → +            │
  RMSNorm → SwiGLU    → +            │
   │                                 │
RMSNorm                              │
   │                                 │
LM Head ◄────────────────────────────┘
   │
다음 토큰 확률
```

### 학습 순서

```text
  ① BPE 토크나이저 (vocab 64k)
           │
           ▼
  ② Pretrain · 18,000 step · lr 6e-4
           │
           ▼  pretrain_base_v1
  ③ SFT · 5,000 step · lr 3e-5
           │
           ▼  sft_base_v1
  ④ 벤치마크 (이 문서)
```

| 단계 | 체크포인트 | steps | 시작점 | 데이터 지문 |
|:-----|:-----------|------:|:-------|:------------|
| Pretrain | `pretrain_base_v1` | 18,000 | 처음부터 | `train.bin` · `53f815d47abc4887` |
| SFT | `sft_base_v1` | 5,000 | pretrain 이어서 | `sft.pt` · `bbc2211091309d3c` |

공통: AdamW (β 0.9 / 0.95, weight decay 0.1), grad clip 1.0, warmup + cosine, CUDA.

### Loss 가 어떻게 내려갔나

**Pretrain**

| step | train | val |
|----:|------:|----:|
| 0 | 11.29 | — |
| 500 | 3.42 | 4.35 |
| 1,000 | 2.76 | 4.08 |
| 1,500 | 3.00 | 3.25 |
| 16,500 | | 2.72 |
| 17,500 | | 2.81 |
| 끝 무렵 | 약 2.1 ~ 2.5 | |

**SFT** (pretrain 체크포인트에서 이어감)

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

Loss 가 내려간 것은 “대화 형식에 적응 중”이라는 신호입니다.  
벤치 점수가 바로 좋아진다는 뜻은 아닙니다. 아래 Q&A 를 보면 그 갭이 드러납니다.

---

## 2. 데이터셋

원본 파일(수십 GB)은 GitHub 에 넣지 않았습니다.  
공개 HuggingFace 등에서 받아 `data.py` 로 전처리했습니다.

### 토크나이저

| | |
|:--|:--|
| 방식 | Byte-level BPE |
| 크기 | **64,000** vocab |
| 샘플 | fineweb, wiki(ko/ja), tinystories 등 약 900MB |
| 특수 토큰 | 역할 토큰, `THINKING` 구간 토큰 |

### Pretrain · 약 19.1억 토큰

`train 1,893,646,391` · `val 19,127,741`

<details>
<summary><b>자연어 (영어 · 한국어 · 일본어)</b> · 펼치기</summary>

<br/>

| 언어 | 파일 | 출처 | 규모 |
|:----:|:-----|:-----|:-----|
| EN | fineweb_edu | HuggingFaceFW/fineweb-edu | ~1.3GB |
| EN | tinystories | roneneldan/TinyStories | ~2.1GB |
| KO | fineweb2_ko | HuggingFaceFW/fineweb-2 | ~1.1GB |
| KO | wikipedia_ko | wikimedia 20231101.ko | ~1.3GB |
| JA | fineweb2_ja | HuggingFaceFW/fineweb-2 | ~1.1GB |
| JA | wikipedia_ja | wikimedia 20231101.ja | ~1.6GB |

</details>

<details>
<summary><b>코드</b> · 펼치기</summary>

<br/>

| 내용 | 출처 | 메모 |
|:-----|:-----|:-----|
| python, c, cpp, java, js, ts, go, rust, … | codeparrot/github-code-clean | 언어당 약 1,200 파일 |
| 함수 단위 | Fsoft-AIC/the-vault-function | 약 4만 rows |
| 커밋 + diff | bigcode/commitpack | 약 8천 rows |

</details>

<details>
<summary><b>SFT · 321,367 예제</b> · 펼치기</summary>

<br/>

| 영역 | 출처 | rows |
|:-----|:-----|----:|
| 코드 지시 | Magicoder-OSS-Instruct | 75,197 |
| 영어 일반 | OpenHermes-2.5 | 60,000 |
| 영어 멀티턴 | UltraChat 200k | 35,000 |
| 일본어 | shisa / llm-jp / dolly-ja 혼합 | 65,015 |
| 한국어 | KoAlpaca v1.1a | 21,155 |
| 수학 CoT | OpenR1-Math | 25,000 |
| 사고 CoT | OpenThoughts3 | 40,000 |

학습 때 쓰는 한 줄 형식 예시:

```json
{
  "user": "2 + 3 * 4 = ?",
  "thinking": "곱셈이 먼저다. 3*4=12, 2+12=14.",
  "assistant": "답은 14입니다."
}
```

user / system 구간은 loss 에서 빼 둡니다. 모델이 “질문 따라 쓰기”가 아니라 **답 쪽**을 배우게 하려는 것입니다.

</details>

### 구성 비율 감각

```text
Pretrain 토큰
  영어 웹·스토리  ████████
  한·일 wiki·웹   ████████
  코드            ██

SFT 예제 (대략)
  영어 대화       ██████████  ~30%
  코드 지시       ████████    ~23%
  일본어          ██████      ~20%
  사고·수학       ██████      ~20%
  한국어          ██          ~7%
```

---

## 3. 어떻게 채점했나

| 항목 | 내용 |
|:-----|:-----|
| 문항 수 | 14개 질문 × 2모드 = 28 생성 |
| 모드 | THINKING 켬 / 끔 |
| QA · 서술 | 0~5점 (답변 전문을 읽고 채점) |
| 코딩 | 유닛테스트 통과 개수 |
| 일시 | 생성 2026-07-09 · 채점 2026-07-10 |

---

## 4. 점수 요약

### sft_base_v1

| 모드 | 결과 |
|:-----|:-----|
| THINKING **켬** | 답 생성 **0 / 14** · 평균 0.0 |
| 일반 채팅 · 한국어 | 평균 **1.33 / 5** |
| 일반 채팅 · 일본어 | 평균 **0.33 / 5** |
| 일반 채팅 · 영어 | 평균 **2.67 / 5** |
| 일반 채팅 · 코딩 | 테스트 **4 / 17** · 문제 완전 통과 **1 / 5** |

THINKING 모드에서는 닫는 태그를 쓰기 전에 생성이 끝나 버립니다.  
추론 코드가 그 경우 답을 비워 두므로, **이 체크포인트는 일반 채팅 모드로 쓰는 편이 맞습니다.**

---

## 5. Pretrain 과 SFT 비교

| | Pretrain | SFT 이후 |
|:--|:---------|:---------|
| 전체 점수 | 사실상 전부 0 | 일반 모드에서 점수 발생 |
| 코딩 테스트 | 0 / 34 | 4 / 17 (일반 모드) |
| 행동 | 질문 따라 쓰기, 같은 말 반복 | 답 형식을 맞추려 함 (내용은 자주 틀림) |

SFT 5,000 step 만으로도  
**“전혀 못 따름 → 형식은 시도함”** 으로 한 단계 올라갑니다.  
그 위 품질은 아직 멀었습니다.

---

## 6. 질문과 답변

점수는 모두 **sft_base_v1 · 일반 채팅 모드** 기준입니다.  
THINKING 모드는 대부분 답이 비어 있어, 필요할 때만 짧게 적습니다.

---

### 한국어

#### 사실 · 수도

| | |
|:--|:--|
| **점수** | 2 / 5 |
| **질문** | 대한민국의 수도는 어디인가요? |
| **기대** | 서울 |
| **모델 답** | 서울은 수도로서, 수도의 역할을 하는 도시인 수도는 서울이 아닌 다른 지역의 수도입니다… 서울은 수도이며, 수도는 따로 없습니다. |
| **코멘트** | “서울”은 나오지만 문장이 앞뒤가 안 맞음 |

THINKING 모드: 답 칸 비어 있음. (사고 조각에는 “서울”이 잠깐 보임)

---

#### 연산 · 사과

| | |
|:--|:--|
| **점수** | 1 / 5 |
| **질문** | 철수는 사과를 5개 가지고 있었는데, 3개를 먹었습니다. 남은 사과는 몇 개인가요? |
| **기대** | 2 |
| **모델 답** | 사과는 3개의 사과를 가지고 있습니다. |
| **코멘트** | 뺄셈을 하지 않음 |

---

#### 요약

| | |
|:--|:--|
| **점수** | 1 / 5 |
| **질문** | 다음 문장을 한 문장으로 요약해 주세요. *「오늘 날씨가 매우 좋아서 공원에 산책을 나갔다. 사람들이 많이 나와 있었고, 아이들은 뛰어놀고 있었다.」* |
| **모델 답** | 원문을 거의 그대로 다시 씀 |
| **코멘트** | 요약이 아니라 따라 쓰기 |

---

### 일본어

#### 사실 · 수도

| | |
|:--|:--|
| **점수** | 0 / 5 |
| **질문** | 日本の首都はどこですか? |
| **기대** | 東京 |
| **모델 답** | 日本は世界でも最も人口の多い国であり、人口が多い国の一つです… |
| **코멘트** | 수도가 아니라 인구 이야기 |

---

#### 연산 · 사과

| | |
|:--|:--|
| **점수** | 1 / 5 |
| **질문** | 太郎はりんごを7個持っていて、2個食べました。残りは何個ですか? |
| **기대** | 5 |
| **모델 답** | 太郎は2個食べています。 |
| **코멘트** | 문제 다시 말하기만 함 |

---

#### 번역

| | |
|:--|:--|
| **점수** | 0 / 5 |
| **질문** | 「今日は天気が良いので公園に散歩に行った。」を英語に翻訳してください。 |
| **모델 답** | 「公園で散歩をするのは素晴らしいね。今日は天気が良いので…」(일본어 유지) |
| **코멘트** | 영어 번역 지시를 무시 |

---

### 영어

#### 사실 · 수도

| | |
|:--|:--|
| **점수** | **4 / 5** |
| **질문** | What is the capital of France? |
| **기대** | Paris |
| **모델 답** | The capital of France is Paris. The capital of France is also located in the heart of France, which is part of the Loire Valley… |
| **코멘트** | 핵심은 맞음. 뒤에 지리가 조금 이상해짐. **이번 벤치에서 가장 괜찮은 신호** |

Pretrain 만 쓰면: 질문을 독일 수도 물음으로 바꿔 되묻기 → 0점.

---

#### 연산 · 기차

| | |
|:--|:--|
| **점수** | 1 / 5 |
| **질문** | If a train travels at 60 km/h for 2 hours, how far does it go? |
| **기대** | 120 |
| **모델 답** | … 60 km/h * 0.5 hours = 60 km … |
| **코멘트** | 시간·거리 관계가 틀림 |

---

#### 요약

| | |
|:--|:--|
| **점수** | 3 / 5 |
| **질문** | Summarize in one sentence: *“The weather was nice today so I went for a walk in the park, and there were many people and children playing.”* |
| **모델 답** | 네 문장으로 늘려 씀. swings, flowers 등 없는 내용 추가 |
| **코멘트** | 분위기는 비슷, “한 문장” 지시는 실패 |

---

## 7. 코딩 문항

함수 시그니처만 주고, 뽑힌 코드를 테스트로 돌렸습니다.  
아래는 **일반 채팅 모드** (THINKING 모드 코딩은 전부 실패).

| 문제 | 테스트 | 점수 | 한 줄 평 |
|:-----|:------:|:----:|:---------|
| is_prime | 0 / 5 | 0 | 짝수 판별로 변질 |
| reverse_string | 0 / 3 | 0 | lower 만 하고 끝 |
| factorial | 0 / 3 | 1 | 재귀 뼈대는 있으나 0 처리 없음 |
| **is_palindrome** | **3 / 3** | **5** | **유일하게 깔끔한 정답** |
| find_max | 1 / 3 | 1 | 변수 덮어쓰기, 운 좋게 한 케이스 |

### 잘 된 답 (is_palindrome)

```python
def is_palindrome(s):
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]
```

### 실패한 예 (is_prime → 사실상 짝수 검사)

```python
def is_prime(n):
    if n <= 1:
        return False
    return n % 2 == 0
```

### 실패한 예 (reverse_string)

```python
def reverse_string(s):
    return s.lower()
```

---

## 8. 느낀 점 · 다음에 할 일

### 잘 된 것

- Pretrain loss 11 → 약 2, SFT 2.67 → 약 1.05 → **학습 루프는 정상**
- SFT 후 “형식 시도”가 벤치에 잡힘
- 영어 사실 1문항, 회문 코드 1문항은 분명한 신호

### 아직 막힌 것

1. THINKING 닫기 실패  
2. 한·일 사실 · 산술 약함  
3. 요약 · 번역 지시 무시  
4. 코딩 완전 통과 1 / 5

### 다음에

- THINKING 종료 패턴을 더 넣은 SFT (v2 방향)  
- 코딩 · 산술 비중 확대, 필요 시 RLVR  
- DPO / 교정 SFT  
- **같은 14문항**으로 버전 비교 유지

---

### 원본 파일 (로컬)

| 경로 | 내용 |
|:-----|:-----|
| `llm/ckpt/benchmark_sft_base_v1.json` | SFT 전체 문항 |
| `llm/ckpt/benchmark_pretrain_base_v1.json` | Pretrain 비교 |
| `llm/DATA_SOURCES.md` | 데이터 출처 목록 |

가중치와 원본 코퍼스는 공개 저장소에 포함하지 않습니다.

---

<div align="center">

[README](README.md) · [ARCHITECTURE](ARCHITECTURE.md) · [POST-TRAINING](POST-TRAINING.md)

</div>
