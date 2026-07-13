[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.md)

---

# 📊 BENCHMARK · base ~327M

> **학습 일기**에 가깝습니다. 예쁜 점수표보다  
> *어떻게 학습했고 · 뭘 썼고 · 뭐라고 답했는지* 를 남기는 문서입니다.

| | |
|:--|:--|
| 🆕 **최신** | `sft_base_v3` · 2026-07-13 |
| 📦 **모델** | Decoder-only · 약 **326.7M** (`base`) |
| 🧪 **세트** | 동일 **14문항 × THINKING on/off** (버전 비교용) |
| 📁 **원본** | [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) |

> ⚠️ 아직 **실사용 단계가 아닙니다.**  
> 파이프라인이 도는지, SFT를 반복하면 뭐가 바뀌는지를 보는 스냅샷입니다.

### 📑 목차

1. [한눈에 보기](#1-한눈에-보기)
2. [학습 과정](#2-학습-과정)
3. [데이터셋](#3-데이터셋)
4. [채점 방법](#4-채점-방법)
5. [sft_base_v3 · 최신 결과](#5-sft_base_v3--최신-결과)
6. [이전 버전 아카이브](#6-이전-버전-아카이브)
7. [느낀 점 · 다음에](#7-느낀-점--다음에)

---

## 1. 한눈에 보기

버전을 한 줄로 비교합니다. 굵은 칸이 **지금 보는 최신**입니다.

| | Pretrain | SFT v1 | SFT v2 | ⭐ **SFT v3** |
|:--|:--------:|:------:|:------:|:------------:|
| 일반 채팅 감 | ~0 | 낮음 | 낮음 | **2.57 / 5** |
| 코딩 완전 통과 | 0/5 | 1/5 | 1/5 | **4/5** |
| THINKING 답 생성 | — | **0/14** | **13/14** | **12/14** |
| 인상 | 채팅 불가 | 형식 시도 | 태그 복구 | 코딩↑ · 한국어↓ |

```text
pretrain ──► sft_v1 ──► sft_v2 ──► sft_v3 ★
  채팅 0점     형식 시도    THINKING 종료    코딩 4/5
```

| 버전 | 잘된 것 | 막힌 것 |
|:----:|:--------|:--------|
| v1 | 지시 형식을 따라 보려 함 | THINKING 답 전부 비어 있음 |
| v2 | `</THINKING>` 종료 대부분 복구 | 코딩·품질은 여전히 약함 |
| **v3** | **코딩 4/5**, 영어 fact/math 회복 | **한국어 붕괴**, THINKING 코딩 0/5 |

---

## 2. 학습 과정

### 🧩 모델 카드

| 항목 | 값 |
|:-----|:---|
| 계열 | Decoder-only Transformer (LLaMA 스타일) |
| 파라미터 | 약 **326.7M** |
| 깊이 · 폭 | 24 layer · `d_model` 1024 |
| Attention | GQA (Q 16 · KV 4) + RoPE |
| FFN | SwiGLU |
| 정규화 | RMSNorm (Pre-Norm) |
| 기타 | weight tying · bias 없음 |

```text
입력 토큰
   │
임베딩 ──────────────────────────────┐
   │                                 │ weight tying
[ Block × 24 ]
  RMSNorm → Attention → +
  RMSNorm → SwiGLU    → +
   │
RMSNorm
   │
LM Head ◄────────────────────────────┘
   │
다음 토큰 확률
```

### 🛤️ 학습 순서

```text
① 토크나이저 (BPE · vocab 64k)
        ▼
② Pretrain · 18k step · lr 6e-4
        ▼  pretrain_base_v1
③ SFT v1 · 5k step · lr 3e-5
        ▼  sft_base_v1
④ SFT v2 · THINKING 종료 패턴 보강
        ▼  sft_base_v2
⑤ SFT v3 · 코딩 데이터 보강
        ▼  sft_base_v3  ★ 본문
```

| 단계 | 체크포인트 | steps | 메모 |
|:-----|:-----------|------:|:-----|
| Pretrain | `pretrain_base_v1` | 18,000 | 처음부터 · `53f815d47abc4887` |
| SFT v1 | `sft_base_v1` | 5,000 | pretrain 이어서 · `bbc2211091309d3c` |
| SFT v2 | `sft_base_v2` | ~5.6k–5.7k | THINKING 종료 복구가 목표 |
| ⭐ SFT v3 | `sft_base_v3` | 데이터 보강 SFT | 코딩 완전 통과 **4/5** |

공통 설정: **AdamW** (β 0.9 / 0.95, wd 0.1) · grad clip 1.0 · warmup + cosine · CUDA

### 📉 Loss 추이 (base · v1 시점)

<details>
<summary><b>Pretrain · SFT v1 loss 표 펼치기</b></summary>

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
| 끝 무렵 | ~2.1–2.5 | |

**SFT v1**

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

</details>

> 💡 Loss가 내려간다 = *대화 형식에 적응 중* 이라는 신호입니다.  
> 벤치 점수가 바로 오른다는 뜻은 **아닙니다.**

---

## 3. 데이터셋

원본 코퍼스(수십 GB)는 저장소에 없습니다.  
공개 HuggingFace 등을 받아 `data.py`로 전처리했습니다.

### 🔤 토크나이저

| | |
|:--|:--|
| 방식 | Byte-level BPE |
| 크기 | **64,000** vocab |
| 샘플 | fineweb, wiki(ko/ja), tinystories 등 ~900MB |
| 특수 토큰 | 역할 토큰 · `THINKING` 구간 토큰 |

### 📚 Pretrain · 약 19.1억 토큰

`train 1,893,646,391` · `val 19,127,741`

<details>
<summary><b>🌍 자연어 (EN / KO / JA)</b></summary>

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
<summary><b>💻 코드</b></summary>

<br/>

| 내용 | 출처 | 메모 |
|:-----|:-----|:-----|
| python, c, cpp, java, js, ts, go, rust, … | codeparrot/github-code-clean | 언어당 ~1,200 파일 |
| 함수 단위 | Fsoft-AIC/the-vault-function | ~4만 rows |
| 커밋 + diff | bigcode/commitpack | ~8천 rows |

</details>

<details>
<summary><b>🗣️ SFT · 321,367 예제 (v1 믹스)</b></summary>

<br/>

| 영역 | 출처 | rows |
|:-----|:-----|----:|
| 코드 지시 | Magicoder-OSS-Instruct | 75,197 |
| 영어 일반 | OpenHermes-2.5 | 60,000 |
| 영어 멀티턴 | UltraChat 200k | 35,000 |
| 일본어 | shisa / llm-jp / dolly-ja | 65,015 |
| 한국어 | KoAlpaca v1.1a | 21,155 |
| 수학 CoT | OpenR1-Math | 25,000 |
| 사고 CoT | OpenThoughts3 | 40,000 |

```json
{
  "user": "2 + 3 * 4 = ?",
  "thinking": "곱셈이 먼저다. 3*4=12, 2+12=14.",
  "assistant": "답은 14입니다."
}
```

user / system 구간은 loss에서 제외합니다. **답 쪽**만 배우게 하려는 설정입니다.

v2·v3는 이 믹스를 이어 가며  
→ v2: THINKING 종료 패턴 · v3: 코딩 품질 을 보강했습니다.

</details>

### 비율 감각 (v1)

```text
Pretrain 토큰
  영어 웹·스토리   ████████
  한·일 wiki·웹    ████████
  코드             ██

SFT 예제
  영어 대화        ██████████  ~30%
  코드 지시        ████████    ~23%
  일본어           ██████      ~20%
  사고·수학        ██████      ~20%
  한국어           ██          ~7%
```

---

## 4. 채점 방법

| 항목 | 내용 |
|:-----|:-----|
| 문항 | **14질문 × 2모드 = 28 생성** (버전마다 동일) |
| 모드 | 🧠 THINKING 켬 / 💬 일반 채팅 |
| QA · 서술 | 0–5점 (답변 전문 채점) |
| 코딩 | 유닛테스트 실행 통과 수 |
| v3 생성 | temp `0.7` · top_p `0.9` · max_new `256` · seed `0` · 1샘플 |
| 일시 | v1 `07-09/10` · v2 `07-10` · **v3 `07-13`** (2026) |

> 📌 코딩 테스트 분모: v1/v2 합 **17** · v3 합 **25** (문제당 5케이스).  
> 버전 비교는 **완전 통과 N/5** 를 보는 편이 안전합니다.

---

## 5. sft_base_v3 · 최신 결과

| | |
|:--|:--|
| 🏁 체크포인트 | `ckpt/sft_base_v3.pt` |
| 📁 JSON | [`benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) |

### 5.1 점수 카드

#### 💬 일반 채팅 (THINKING 끔) — 전체 평균 **2.57 / 5**

| 영역 | 점수 | 상태 |
|:-----|:----:|:----:|
| 🇰🇷 한국어 | **0.00 / 5** | 🔴 |
| 🇯🇵 일본어 | **1.33 / 5** | 🟡 |
| 🇺🇸 영어 | **4.00 / 5** | 🟢 |
| 💻 코딩 | **20/25** 테스트 · **4/5** 완전 통과 | 🟢 |

#### 🧠 THINKING 켬

| 지표 | 값 | 상태 |
|:-----|:--:|:----:|
| 평균 점수 | **0.21 / 5** | 🔴 |
| 비어 있지 않은 답 | **12 / 14** | 🟢 (종료는 대체로 OK) |
| 코딩 완전 통과 | **0 / 5** | 🔴 산문만 출력 |

### 5.2 v2 → v3 하이라이트

| | 변화 |
|:--|:-----|
| ✅ | 코딩 완전 통과 **0/5 → 4/5** (`is_prime`, `factorial`, `is_palindrome`, `find_max`) |
| ✅ | 영어 Paris **5/5**, 120 km **5/5** (v2 Burgundy 환각에서 회복) |
| ⚠️ | `reverse_string` 본문은 `s[::-1]` 정답 · top-level `print` → NameError → **0/5** |
| ❌ | 한국어 6문항(사고+일반) **전부 0점** — 카테고리 붕괴 |
| ❌ | THINKING 코딩은 여전히 **코드 없음 0/5** |
| 🔍 | 추론 중간에 맞는 식(7−2=5)이 보이다 최종이 틀어지는 패턴 증가 |

---

### 5.3 질문과 답변 · 일반 채팅

점수는 모두 **sft_base_v3 · THINKING 끔** 기준입니다.

#### 🇰🇷 한국어 — 평균 0.00

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **0** | 서울 | GDP·면적 환각, “서울” 없음 | v1(2점)보다 퇴보 |
| 사과 5−3 | **0** | 2 | “3개를 먹었습니다” 재진술 | 뺄셈 없음 |
| 요약 | **0** | 한 문장 | 같은 구문 무한 반복 | degenerate loop |

> 🧠 THINKING 수도 문항: thinking 안에는 “서울”까지 도달했지만, 태그 뒤 **답이 비어 있음**.

#### 🇯🇵 일본어 — 평균 1.33

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **4** | 東京 | 「首都は東京です」+ 랜드마크 중복 | 정답, 약간 감점 |
| 사과 7−2 | **0** | 5 | `7−8=2` 식 붕괴 | 정답 없음 |
| 번역 | **0** | 영문 | “Hopper's Park is a park for kids.” | 무관 문장 |

> 🧠 THINKING: 수도는 thinking에 東京이 있으나 답 공란 (v2 사고 모드 5점에서 퇴보).  
> 연산은 중간에 7−2=5 후 ×7=35로 탈선.

#### 🇺🇸 영어 — 평균 4.00

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **5** | Paris | “The capital of France is Paris.” | v2 환각에서 회복 🟢 |
| 기차 60×2 | **5** | 120 | Distance = Speed × Time → 120 km | 모범 답안 🟢 |
| 요약 | **2** | 한 문장 | “The weather was nice today.” | 정보는 많이 빠짐 |

---

### 5.4 코딩 · 일반 채팅

함수 시그니처만 주고, 뽑힌 코드를 **유닛테스트로 실행**했습니다.  
🧠 THINKING 모드 코딩 5문항 → 전부 산문만 → **0/5**.

| 문제 | 테스트 | 점수 | 한 줄 |
|:-----|:------:|:----:|:------|
| `is_prime` | **5/5** | **5** | ✅ √n 시도분해 |
| `reverse_string` | 0/5 | 0 | ⚠️ 본문 OK · `print` NameError |
| `factorial` | **5/5** | **5** | ✅ base case 재귀 (v2 무한재귀 수정) |
| `is_palindrome` | **5/5** | **5** | ✅ lower + 뒤집기 (v2 항상 True 수정) |
| `find_max` | **5/5** | **5** | ✅ 루프 직접 구현 (v2 `max()` 의존 탈피) |

<details>
<summary><b>✅ 통과한 코드 펼치기</b></summary>

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
<summary><b>⚠️ 아쉬운 실패 · reverse_string</b></summary>

<br/>

```python
def reverse_string(s):
    return s[::-1]

print(reverse_string(s))  # NameError: s is not defined → 실행 0/5
```

함수 자체는 맞지만, 실행 하네스를 깨는 top-level `print` 때문에 점수가 0이 됩니다.

</details>

---

## 6. 이전 버전 아카이브

최신과 비교용입니다. 필요할 때만 펼치면 됩니다.

<details>
<summary><b>📘 sft_base_v2 · 2026-07-10</b></summary>

<br/>

📁 [`benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json)

#### 점수

| 모드 | 결과 |
|:-----|:-----|
| 🧠 THINKING | 답 **13/14** · 평균 **0.71/5** |
| 🇰🇷 일반 | **0.33/5** |
| 🇯🇵 일반 | **2.00/5** |
| 🇺🇸 일반 | **1.00/5** |
| 💻 일반 | 테스트 **9/17** · 완전 **1/5** (`find_max` → 내장 `max()`) |

#### v1 → v2

| | |
|:--|:--|
| ✅ | THINKING 종료 버그 대부분 해결 (0/14 → 13/14) |
| ✅ | 첫 사고 모드 만점: `ja_fact` **5/5** |
| ⚠️ | 영어 fact = Burgundy 환각 (temp 0.7 · 단일 샘플) |
| ❌ | 코딩 본질적으로 약함 · thinking 코드 없음 |

#### 일반 채팅 한눈에

| 문항 | 점수 | 메모 |
|:-----|:----:|:-----|
| ko_fact / math / summary | 0 / 0 / 1 | 한국어 약함 |
| ja_fact / math / translate | **5** / 1 / 0 | 東京 정답 |
| en_fact / math / summary | 0 / 2 / 1 | Burgundy 환각 |
| coding 완전 통과 | 1/5 | `find_max` only |

#### 코딩 (일반)

| 문제 | 테스트 | 메모 |
|:-----|:------:|:-----|
| is_prime | 2/5 | 부분 |
| reverse_string | 2/3 | 부분 |
| factorial | 0/3 | 무한 재귀 등 |
| is_palindrome | 2/3 | 항상 True 계열 |
| **find_max** | **3/3** | `max()` 위임 |

</details>

<details>
<summary><b>📗 sft_base_v1 · 2026-07-09/10</b></summary>

<br/>

📁 [`benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json)

#### 점수

| 모드 | 결과 |
|:-----|:-----|
| 🧠 THINKING | 답 **0/14** · 평균 0.0 🔴 |
| 🇰🇷 일반 | **1.33/5** |
| 🇯🇵 일반 | **0.33/5** |
| 🇺🇸 일반 | **2.67/5** |
| 💻 일반 | 테스트 **4/17** · 완전 **1/5** |

> 🧠 닫는 태그 전에 생성이 끝남 → 추론 코드가 답을 비움.  
> 이 체크포인트는 **일반 채팅 모드**로 쓰는 편이 맞습니다.

#### Pretrain vs SFT v1

| | Pretrain | SFT v1 |
|:--|:---------|:-------|
| 점수 | 사실상 전부 0 | 일반 모드에서 점수 발생 |
| 코딩 | 0/34 | 4/17 |
| 행동 | 따라 쓰기 · 반복 | 형식은 시도 (내용은 자주 틀림) |

SFT 5,000 step만으로 **「전혀 못 따름 → 형식은 시도」** 로 한 단계 올라갑니다.

#### Q&A · 일반 채팅

| 언어 | 문항 | 점수 | 메모 |
|:----:|:-----|:----:|:-----|
| 🇰🇷 | 수도 / 사과 / 요약 | 2 / 1 / 1 | “서울”은 나오나 문장 모순 |
| 🇯🇵 | 수도 / 사과 / 번역 | 0 / 1 / 0 | 인구 이야기 · 지시 무시 |
| 🇺🇸 | 수도 / 기차 / 요약 | **4** / 1 / 3 | Paris가 당시 가장 강한 신호 |

Pretrain only (en_fact): 질문을 독일 수도로 바꿔 되묻기 → 0점.

#### 코딩 (일반)

| 문제 | 테스트 | 점수 | 메모 |
|:-----|:------:|:----:|:-----|
| is_prime | 0/5 | 0 | 짝수 판별로 변질 |
| reverse_string | 0/3 | 0 | `.lower()` 만 |
| factorial | 0/3 | 1 | 재귀 뼈대, 0 처리 없음 |
| **is_palindrome** | **3/3** | **5** | ✅ 유일한 깔끔한 정답 |
| find_max | 1/3 | 1 | 운 좋은 1케이스 |

```python
# ✅ v1 통과
def is_palindrome(s):
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]

# ❌ v1 is_prime → 사실상 짝수 검사
def is_prime(n):
    if n <= 1:
        return False
    return n % 2 == 0
```

</details>

<details>
<summary><b>📙 pretrain_base_v1 · 채팅 ~0점</b></summary>

<br/>

📁 [`benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json)

| | 결과 |
|:--|:-----|
| 채팅 반응 | 거의 없음 · ~0 / 28 |
| 코딩 | 0 / 34 |
| 행동 | 따라 쓰기 · 같은 말 반복 · 질문 바꿔 되묻기 |

SFT 전에는 **지시 형식 자체가 학습되지 않은 상태**입니다.  
v1 SFT 이후 “형식을 시도한다”는 변화가 벤치에 처음 잡힙니다.

</details>

---

## 7. 느낀 점 · 다음에

### ✅ 버전에 걸쳐 잘 된 것

- Pretrain loss 11 → ~2, SFT v1 2.67 → ~1.05 → **학습 루프는 정상**
- v1: “형식 시도”가 벤치에 잡힘
- v2: THINKING **닫기** 대부분 복구 (0/14 → 13/14)
- v3: 일반 채팅 **코딩 4/5** — 데이터 보강 목표 달성
- 영어 fact/math 는 v3에서 다시 안정 신호

### 🚧 아직 막힌 것

1. 🧠 THINKING 모드 코딩 → 여전히 산문만 (0/5)
2. 🇰🇷 한국어 → v3에서 카테고리 붕괴
3. 산술·지시 따르기 불안정 (중간은 맞다 최종이 틀림)
4. `reverse_string`처럼 **정답 본문 + 위험한 top-level 코드**
5. 실사용 품질까지는 아직 멀다

### 🧭 다음에

| 우선 | 내용 |
|:----:|:-----|
| 1 | THINKING에서도 **코드를 내고 태그를 닫는** SFT / 제약 디코딩 |
| 2 | 한국어 사실·산술·요약 비중 재균형 (붕괴 복구) |
| 3 | 코딩 출력 형식 정리 (실행 하네스 깨는 side effect 억제) |
| 4 | DPO / 교정 SFT / RLVR |
| 5 | **같은 14문항**으로 버전 비교 유지 |

---

### 📁 원본 파일

| 경로 | 내용 |
|:-----|:-----|
| [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) | ⭐ **최신** SFT v3 |
| [`ckpt/benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json) | SFT v2 |
| [`ckpt/benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json) | SFT v1 |
| [`ckpt/benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json) | Pretrain |

> 가중치(`.pt`)와 원본 코퍼스는 공개 저장소에 포함하지 않습니다.

---

<div align="center">

[README](README.md) · [ARCHITECTURE](ARCHITECTURE.md) · [POST-TRAINING](POST-TRAINING.md)

</div>
