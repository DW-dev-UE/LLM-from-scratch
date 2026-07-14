[한국어](BENCHMARK-v1.md) · [English](BENCHMARK-v1.en.md) · [日本語](BENCHMARK-v1.ja.md)

[← README](README.md)

---

# 📊 BENCHMARK · base ~327M

> **학습 일기**에 가깝습니다. 예쁜 점수표보다  
> *어떻게 학습했고 · 뭘 썼고 · 뭐라고 답했는지* 를 남기는 문서입니다.

| | |
|:--|:--|
| 🆕 **최신** | `sft_base_v5` · 2026-07-14 |
| 📦 **모델** | Decoder-only · 약 **326.7M** (`base`) |
| 🧪 **세트** | 동일 **14문항 × THINKING on/off** (버전 비교용) |
| 📁 **원본** | [`ckpt/benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) |

> ⚠️ 아직 **실사용 단계가 아닙니다.**  
> 파이프라인이 도는지, 체크포인트·프로토콜이 바뀌면 뭐가 달라지는지를 보는 스냅샷입니다.

### 📑 목차

1. [한눈에 보기](#1-한눈에-보기)
2. [학습 과정](#2-학습-과정)
3. [데이터셋](#3-데이터셋)
4. [채점 방법](#4-채점-방법)
5. [sft_base_v5 · 최신 결과](#5-sft_base_v5--최신-결과)
6. [이전 버전 아카이브](#6-이전-버전-아카이브)
7. [느낀 점 · 다음에](#7-느낀-점--다음에)

---

## 1. 한눈에 보기

버전을 한 줄로 비교합니다. 굵은 칸이 **지금 보는 최신**입니다.

> ⚠️ **비교 주의**  
> · **프로토콜**: v1–v4는 temp `0.7` 단일 샘플 · **v5는 greedy (temp `0.0`)**  
> · **베이스 분기**: v1–v4는 `pretrain_base_v1` · **v5는 `pretrain_base_v2` 위 SFT**  
> · 따라서 **v5 > v4 > v3 를 “순수 SFT 사다리”로 읽으면 안 됩니다.**  
> · 버전 간 코딩 비교는 테스트 분모 차이 때문에 **완전 통과 N/5** 가 안전합니다.

| | Pretrain v1 | SFT v1 | SFT v2 | SFT v3 | SFT v4 | ⭐ **SFT v5** |
|:--|:-----------:|:------:|:------:|:------:|:------:|:------------:|
| 일반 채팅 평균 | ~0 | 낮음† | 낮음† | 2.57 | 2.07 | **3.21 / 5** |
| 코딩 완전 통과 (일반) | 0/5 | 1/5 | 1/5 | 4/5 | 3/5 | **5/5** |
| 코딩 완전 통과 (사고) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **0/5** |
| THINKING 비공란 답 | — | **0/14** | **13/14** | 12/14 | 11/14 | **10/14** |
| THINKING 평균 | ~0 | 0.0 | 0.71 | 0.21 | 0.50 | **0.50** |
| 한국어 합 (사고+일반) | 0/30 | — | 2/30 | 0/30 | 1/30 | **8/30** |
| 베이스 | v1 | v1 | v1 | v1 | v1 | **v2** |
| 샘플링 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | **0.0 greedy** |
| 인상 | 채팅 불가 | 형식 시도 | 태그 복구 | 코딩↑ · KO↓ | 횡보 · 노이즈 | 코딩 만점 · KO↑ |

† v1/v2 일반 채팅은 언어별 평균만 기록 (전체 평균 미집계):  
v1 KO **1.33** / JA **0.33** / EN **2.67** · v2 KO **0.33** / JA **2.0** / EN **1.0**

```text
                    ┌─► sft_v1 ─► sft_v2 ─► sft_v3 ─► sft_v4
pretrain_v1 ────────┤     형식      태그닫기    코딩4/5     KO보강·횡보
   18k · lr 6e-4    │
                    └─► pretrain_v2 (25k · lr 1e-4 · corpus v2)
                              │
                              └─► sft_v5 ★  (동일 sft.pt as v4 · greedy 벤치)
                                    코딩 5/5 · 일반 3.21 · KO 8/30
```

| 버전 | 잘된 것 | 막힌 것 |
|:----:|:--------|:--------|
| v1 | 지시 형식을 따라 보려 함 | THINKING 답 전부 비어 있음 |
| v2 | `</THINKING>` 종료 대부분 복구 | 코딩·품질은 여전히 약함 |
| v3 | 코딩 **4/5**, 영어 fact/math 회복 | **한국어 붕괴**, THINKING 코딩 0/5 |
| v4 | KO SFT 비중↑ · thinking 안 서울 정답 등장 | 벤치 KO는 거의 회복 안 됨 · item flip 노이즈 |
| **v5** | **코딩 5/5**, 일반 **3.21**, KO **8/30** | **thinking→answer 핸드오프** · thinking 코딩 0/5 |

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
② Pretrain v1 · 18k step · lr 6e-4
        ▼  pretrain_base_v1
        ├─► ③ SFT v1 · 5k · lr 3e-5 ──► sft_base_v1
        ├─► ④ SFT v2 · ~5.7k · THINKING 종료 보강 ──► sft_base_v2
        ├─► ⑤ SFT v3 · 8k · 코딩 데이터 보강 ──► sft_base_v3
        ├─► ⑥ SFT v4 · 9.2k · 한국어 SFT 보강 ──► sft_base_v4
        │
        └─► ②′ Pretrain v2 · 25k · lr 1e-4 · corpus v2
                ▼  pretrain_base_v2
                └─► ⑦ SFT v5 · 9.2k · (v4와 동일 sft.pt) ──► sft_base_v5 ★
```

| 단계 | 체크포인트 | steps | init | 데이터 해시 · 메모 |
|:-----|:-----------|------:|:-----|:-------------------|
| Pretrain v1 | `pretrain_base_v1` | 18,000 | — | `53f815d47abc4887` · lr 6e-4 |
| SFT v1 | `sft_base_v1` | 5,000 | pretrain_v1 | `bbc2211091309d3c` |
| SFT v2 | `sft_base_v2` | 5,700 | pretrain_v1 | `fc177a36965df49a` · THINKING 종료 |
| SFT v3 | `sft_base_v3` | 8,000 | pretrain_v1 | `db63d09ddfa41388` · 코딩 보강 |
| SFT v4 | `sft_base_v4` | 9,200 | pretrain_v1 | `975c1771bfff1919` · KO +70k |
| Pretrain v2 | `pretrain_base_v2` | 25,000 | pretrain_v1 | `4ad58fc7307962c0` · lr 1e-4 |
| ⭐ SFT v5 | `sft_base_v5` | 9,200 | **pretrain_v2** | **`975c1771bfff1919`** (v4와 동일) |

공통 설정: **AdamW** (β 0.9 / 0.95, wd 0.1) · grad clip 1.0 · warmup + cosine · CUDA · SFT lr `3e-5`

> 🔑 **v5의 핵심 변수**는 SFT 믹스가 아니라 **베이스(pretrain_v2)** 입니다.  
> v4와 v5는 같은 `sft.pt` (`975c…`) · 같은 9,200 step · 다른 init 가중치입니다.

### 📉 Loss 추이

<details>
<summary><b>Pretrain · SFT loss 표 펼치기</b></summary>

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
| 끝 무렵 | ~2.1–2.5 | |

**SFT v1** (init = pretrain_v1)

| step | train |
|----:|------:|
| 0 | 2.67 |
| 500 | 1.60 |
| 1,000 | 1.39 |
| 4,950 | **1.05** |

**SFT v5** (init = pretrain_v2 · 동일 sft 믹스)

| step | train |
|----:|------:|
| 0 | **2.35** (v1 계열 init loss ~2.8대보다 낮음) |
| 500 | 1.32 |
| 1,000 | 1.40 |
| 끝 무렵 | ~1.1–1.5 대역 진동 |

</details>

> 💡 Loss가 내려간다 = *대화 형식에 적응 중* 이라는 신호입니다.  
> 벤치 점수가 바로 오른다는 뜻은 **아닙니다.**  
> 다만 v5의 **낮은 SFT 초기 loss**는 corpus-v2 continued pretrain이 SFT 쪽에 유리하게 붙었다는 힌트입니다.

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

### 📚 Pretrain

**v1** · 약 19.1억 토큰 · `train 1,893,646,391` · `val 19,127,741` · hash `53f815…`

<details>
<summary><b>🌍 자연어 (EN / KO / JA) · v1 규모</b></summary>

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
<summary><b>💻 코드 · v1</b></summary>

<br/>

| 내용 | 출처 | 메모 |
|:-----|:-----|:-----|
| python, c, cpp, java, js, ts, go, rust, … | codeparrot/github-code-clean | 언어당 ~1,200 파일 |
| 함수 단위 | Fsoft-AIC/the-vault-function | ~4만 rows |
| 커밋 + diff | bigcode/commitpack | ~8천 rows |

</details>

**v2** · continued pretrain from v1 · **25,000 step · lr 1e-4** · data hash `4ad58f…`  
코퍼스를 확장·재토큰화한 corpus v2 (fineweb_edu 확대, finemath 추가 등).  
일부 소스(vault/commitpack)는 파이프라인 단계에서 실패 기록 있음.

> 참고: ko-wiki val loss는 v1 대비 소폭 악화(보고: 2.504 → 2.605)였지만,  
> SFT 초기 loss·다운스트림 벤치는 **v2 베이스가 더 유리**했습니다.

### 🗣️ SFT 믹스 진화

<details>
<summary><b>v1 믹스 · 321,367 예제</b></summary>

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

</details>

<details>
<summary><b>v3 믹스 · 446,771 예제 (코딩·수학 보강)</b></summary>

<br/>

v1 위에 추가·배수:

| 추가 | rows × 배수 |
|:-----|------------:|
| gsm8k | 7,473 ×4 |
| mbpp | 474 ×8 |
| codealpaca_20k | 20,016 ×2 |
| orca_math_ko | 25,000 ×1 |
| jamard_gsm8k_ja | 6,672 ×4 |

→ 코딩 완전 통과 **1/5 → 4/5** (일반 채팅)

</details>

<details>
<summary><b>v4 / v5 믹스 · 516,771 예제 (한국어 보강) · hash `975c…`</b></summary>

<br/>

v3 위에 한국어 추가:

| 추가 | rows |
|:-----|----:|
| kullm_v2 | 40,000 |
| ko_wikidata_qa | 30,000 |

한국어 비중 약 **10.3% → 22.5%**.  
**v4와 v5는 이 동일 `sft.pt`를 공유**합니다. 차이는 init 베이스뿐입니다.

</details>

### 비율 감각 (v1 기준)

```text
Pretrain 토큰
  영어 웹·스토리   ████████
  한·일 wiki·웹    ████████
  코드             ██

SFT 예제 (v1)
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
| QA · 서술 | 0–5점 (답변 전문 채점 · Claude) |
| 코딩 | 유닛테스트 실행 통과 수 |
| v1–v4 생성 | temp **`0.7`** · top_p `0.9` · max_new `256` · seed `0` · **1샘플** |
| **v5 생성** | temp **`0.0` greedy** · top_p `0.9` · max_new `256` · seed `0` |
| 일시 | v1 `07-09/10` · v2 `07-10` · v3 `07-13` · v4 `07-13` · **v5 `07-14`** (2026) |

> 📌 코딩 테스트 분모: v1/v2 합 **17** · v3+ 합 **25** (문제당 5케이스).  
> 버전 비교는 **완전 통과 N/5** 를 보는 편이 안전합니다.

> 🎲 **샘플링 노이즈 (v4 교훈)**  
> temp 0.7 · 단일 샘플에서 v3↔v4 item flip이 컸습니다  
> (예: en_fact 5→0, code_prime 5→0, code_maxlist 5→0, code_reverse 0→5).  
> **v5부터 greedy** 로 바꿔 버전 내 노이즈를 제거했습니다.  
> 다만 v5 수치를 v3/v4와 나란히 둘 때는 **프로토콜 차이 주석이 필수**입니다.

> 🇰🇷 한국어 합 점수: 사고+일반 각 3문항 × 0–5 = **최대 30점**.

---

## 5. sft_base_v5 · 최신 결과

| | |
|:--|:--|
| 🏁 체크포인트 | `ckpt/sft_base_v5.pt` |
| 🧱 베이스 | `pretrain_base_v2` (25k · lr 1e-4 · corpus v2) |
| 📦 SFT 데이터 | v4와 동일 `sft.pt` · `975c1771bfff1919` · 9,200 step |
| 🎲 생성 | **greedy · temp 0.0** (결정적) |
| 📁 JSON | [`benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) |

### 5.1 점수 카드

#### 💬 일반 채팅 (THINKING 끔) — 전체 평균 **3.21 / 5**

| 영역 | 점수 | 상태 |
|:-----|:----:|:----:|
| 🇰🇷 한국어 | **1.67 / 5** (2 / 0 / 3) | 🟡 회복 중 |
| 🇯🇵 일본어 | **1.33 / 5** (4 / 0 / 0) | 🟡 fact만 |
| 🇺🇸 영어 | **3.67 / 5** (5 / 3 / 3) | 🟢 |
| 💻 코딩 | **25/25** 테스트 · **5/5** 완전 통과 | 🟢 첫 만점 |

#### 🧠 THINKING 켬 — 평균 **0.50 / 5**

| 지표 | 값 | 상태 |
|:-----|:--:|:----:|
| 평균 점수 | **0.50 / 5** | 🔴 |
| 비어 있지 않은 답 | **10 / 14** | 🟡 (v4 11/14 · v3 12/14) |
| 코딩 완전 통과 | **0 / 5** | 🔴 산문만 출력 |
| 한국어 합 (사고) | 3 / 15 | math만 부분 점수 |

### 5.2 v4 → v5 하이라이트 (프로토콜·베이스 주의)

| | 변화 |
|:--|:-----|
| ✅ | 코딩 완전 통과 **3/5 → 5/5** (첫 만점 · 결정적 greedy) |
| ✅ | 일반 평균 **2.07 → 3.21** |
| ✅ | 한국어 합 **1/30 → 8/30** (첫 요약 성공 · 사고 math에 정답 숫자) |
| ✅ | SFT init loss 개선 신호 (pretrain_v2 효과) |
| ⚠️ | **같은 SFT 믹스**, 다른 베이스 + greedy 프로토콜 → “SFT만 더 잘됨”이 아님 |
| ❌ | THINKING 코딩은 여전히 **코드 없음 0/5** |
| ❌ | thinking 안에 정답이 있어도 **답 칸이 비는** 핸드오프 (ko_fact, ja_fact, en_summary) |
| ❌ | 한·일 산술 답변은 여전히 자주 붕괴 |

---

### 5.3 질문과 답변 · 일반 채팅

점수는 모두 **sft_base_v5 · THINKING 끔 · greedy** 기준입니다.

#### 🇰🇷 한국어 — 평균 1.67

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **2** | 서울 | “가장 큰 도시는 서울” 등 · 수도 명시 약함 | v3/v4(0) 대비 개선 |
| 사과 5−3 | **0** | 2 | “남은 사과는 3개” | 여전히 오답 |
| 요약 | **3** | 한 문장 | “오늘 날씨가 매우 좋아서 공원에 산책을 나갔다.” | **한국어 요약 첫 성공** 🟢 |

> 🧠 THINKING 수도: thinking = `대한민국의 수도는 서울특별시입니다.` **완벽** · 답 공란 → 점수 0.  
> 🧠 THINKING 사과: 답에 `2` 등장 (동사 오류) → **3점** · 한국어 math 첫 부분 정답.

#### 🇯🇵 일본어 — 평균 1.33

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **4** | 東京 | 「首都は東京です」+ 순환 반복 | 정답, 약간 감점 |
| 사과 7−2 | **0** | 5 | “2個食べました” 재진술 | 정답 없음 |
| 번역 | **0** | 영문 | 공원 에세이로 탈선 | 영어 없음 |

> 🧠 THINKING 수도: thinking에 東京 있으나 답 공란.

#### 🇺🇸 영어 — 평균 3.67

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **5** | Paris | “The capital of France is Paris.” | 완벽 🟢 |
| 기차 60×2 | **3** | 120 | 선두에 120 km · 이후 reasoning이 자기모순 | 부분 |
| 요약 | **3** | 한 문장 | 아이들/공원 위주 · 날씨 정보 탈락 | 준수 |

---

### 5.4 코딩 · 일반 채팅

함수 시그니처만 주고, 뽑힌 코드를 **유닛테스트로 실행**했습니다.  
🧠 THINKING 모드 코딩 5문항 → 전부 산문만 → **0/5**.

| 문제 | 테스트 | 점수 | 한 줄 |
|:-----|:------:|:----:|:------|
| `is_prime` | **5/5** | **5** | ✅ √n 시도분해 |
| `reverse_string` | **5/5** | **5** | ✅ `s[::-1]` (top-level print 없음) |
| `factorial` | **5/5** | **5** | ✅ base case 재귀 |
| `is_palindrome` | **5/5** | **5** | ✅ lower + 뒤집기 (중복 줄 1 무해) |
| `find_max` | **5/5** | **5** | ✅ 내장 `max()` 위임 · 통과 |

<details>
<summary><b>✅ v5 통과 코드 펼치기</b></summary>

<br/>

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def reverse_string(s):
    return s[::-1]

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def is_palindrome(s):
    s = s.lower()
    s = s.lower()
    return s == s[::-1]

def find_max(lst):
    max_num = max(lst)
    return max_num
```

</details>

---

### 5.5 THINKING 핸드오프 — 지금 가장 선명한 병목

4/14 thinking 실행이 **빈 답**으로 끝났고, 그중 3개는 thinking 안에 쓸 만한 내용이 있었습니다.

| 문항 | thinking 안 | answer 칸 | 점수 |
|:-----|:------------|:----------|:----:|
| ko_fact | `대한민국의 수도는 서울특별시입니다.` | (빈칸) | 0 |
| ja_fact | 東京 정답 포함 | (빈칸) | 0 |
| en_summary | 준수한 요약 | (빈칸) | 0 |
| ko_summary | 원문 복사 | (빈칸) | 0 |

> 지식/요약 **인출은 되는 경우**가 늘었고, **태그 뒤 답으로 옮기는 단계**가 남았습니다.  
> v1의 “닫기 실패”와는 다른 병목입니다 (닫기는 되지만 전달이 안 됨).

---

## 6. 이전 버전 아카이브

최신과 비교용입니다. 필요할 때만 펼치면 됩니다.

<details>
<summary><b>📕 sft_base_v4 · 2026-07-13 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v4.json`](ckpt/benchmark_sft_base_v4.json)

한국어 SFT 보강 (+kullm-v2 40k, +ko_wikidata_QA 30k · ko 비중 10.3%→22.5%).  
전체적으로 **횡보**. 벤치 한국어는 거의 회복되지 않음.

#### 점수

| 모드 | 결과 |
|:-----|:-----|
| 일반 평균 | **2.07 / 5** |
| 🧠 THINKING 평균 | **0.50 / 5** · 비공란 답 **11/14** |
| 🇰🇷 일반 | **0.33/5** (0 / 0 / 1) |
| 🇯🇵 일반 | **1.67/5** (5 / 0 / 0) |
| 🇺🇸 일반 | **2.67/5** (0 / 5 / 3) |
| 💻 일반 | 완전 **3/5** · 테스트 15/25 |
| 🇰🇷 합 (사고+일반) | **1/30** |

#### 코딩 (일반 · temp 0.7)

| 문제 | 테스트 | 메모 |
|:-----|:------:|:-----|
| is_prime | 0/5 | 본문 비어 IndentationError — 샘플링 노이즈 가능 |
| **reverse_string** | **5/5** | `s[::-1]` (v3의 print 부작용 수정) |
| factorial | 5/5 | base case 재귀 |
| is_palindrome | 5/5 | alnum 필터 + lower · 더 견고 |
| find_max | 0/5 | 변수 섀도잉 버그 |

#### 하이라이트

| | |
|:--|:--|
| ✅ | thinking 안 첫 완벽한 한국어 수도 문장 (`서울특별시`) — 답 칸은 비어 있음 |
| ✅ | thinking 모드 첫 math 만점: en_math **5/5** |
| ⚠️ | v3 대비 item flip 큼 → **temp 0.7 단일 샘플 노이즈** 지배 |
| ❌ | 한국어 벤치 1/30 — SFT 비중만으로는 부족 → pretrain 쪽 레버 권고 |

</details>

<details>
<summary><b>📘 sft_base_v3 · 2026-07-13 · pretrain_v1 · temp 0.7</b></summary>

<br/>

📁 [`benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json)

코딩 데이터 보강 목표 달성. 한국어 카테고리 붕괴.

#### 점수

| 모드 | 결과 |
|:-----|:-----|
| 일반 평균 | **2.57 / 5** |
| 🧠 THINKING 평균 | **0.21 / 5** · 비공란 답 **12/14** |
| 🇰🇷 일반 | **0.00/5** |
| 🇯🇵 일반 | **1.33/5** |
| 🇺🇸 일반 | **4.00/5** |
| 💻 일반 | 테스트 **20/25** · 완전 **4/5** |
| 🇰🇷 합 | **0/30** |

#### 코딩 (일반)

| 문제 | 테스트 | 메모 |
|:-----|:------:|:-----|
| is_prime | 5/5 | √n 시도분해 |
| reverse_string | 0/5 | 본문 OK · top-level `print` → NameError |
| factorial | 5/5 | base case 재귀 (v2 무한재귀 수정) |
| is_palindrome | 5/5 | lower + 뒤집기 (v2 항상 True 수정) |
| find_max | 5/5 | 루프 직접 구현 |

#### 일반 채팅 한눈에

| 문항 | 점수 | 메모 |
|:-----|:----:|:-----|
| ko_fact / math / summary | 0 / 0 / 0 | 한국어 붕괴 |
| ja_fact / math / translate | **4** / 0 / 0 | 東京 정답 |
| en_fact / math / summary | **5** / **5** / 2 | Paris·120 회복 |
| coding 완전 통과 | 4/5 | reverse만 실패 |

</details>

<details>
<summary><b>📗 sft_base_v2 · 2026-07-10 · pretrain_v1 · temp 0.7</b></summary>

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
<summary><b>📙 sft_base_v1 · 2026-07-09/10 · pretrain_v1 · temp 0.7</b></summary>

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
<summary><b>📓 pretrain_base_v1 · 채팅 ~0점</b></summary>

<br/>

📁 [`benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json)

| | 결과 |
|:--|:-----|
| 채팅 반응 | 거의 없음 · ~0 / 28 |
| 코딩 | 0 / 34 (양 모드 합) |
| 행동 | 따라 쓰기 · 같은 말 반복 · 질문 바꿔 되묻기 |

SFT 전에는 **지시 형식 자체가 학습되지 않은 상태**입니다.  
v1 SFT 이후 “형식을 시도한다”는 변화가 벤치에 처음 잡힙니다.

> 📭 **pretrain_base_v2 단독 채팅 벤치는 아직 없음.**  
> v2의 효과는 현재 **sft_base_v5** 다운스트림으로만 관측됩니다.

</details>

---

## 7. 느낀 점 · 다음에

### ✅ 버전에 걸쳐 잘 된 것

- Pretrain loss 11 → ~2, SFT 루프 정상 · 파이프라인 검증 OK
- v1: “형식 시도”가 벤치에 잡힘
- v2: THINKING **닫기** 대부분 복구 (0/14 → 13/14)
- v3: 일반 채팅 **코딩 4/5** — 데이터 보강 목표 달성
- v4: KO SFT 비중↑ · thinking 안 한국어 정답 단문 최초 출현 (전달은 실패)
- **v5: 일반 코딩 5/5 · 평균 3.21 · KO 8/30** — 현재 최고 · greedy로 재현 가능
- continued pretrain (v2) 이 **같은 SFT 믹스**에서도 다운스트림을 끌어올림

### 🚧 아직 막힌 것 (우선순위)

1. 🧠 **thinking → answer 핸드오프** — 인출은 되는데 답 칸이 빔
2. 🧠 THINKING 모드 **코딩 0/5** — 여전히 산문만
3. 🇰🇷 한국어 — 개선(8/30)됐지만 **고쳐진 것은 아님** (fact·math 불안정)
4. 한·일 산술·지시 따르기 불안정
5. 실사용 품질까지는 아직 멀다

### 🧭 다음에

| 우선 | 내용 |
|:----:|:-----|
| 1 | THINKING 닫은 뒤 **답을 강제 생성**하는 SFT / 제약 디코딩 |
| 2 | THINKING에서도 **코드를 내도록** 포맷 학습 (코딩 0/5 해소) |
| 3 | 한국어 fact·산술 추가 보강 (pretrain + SFT 병행) |
| 4 | **greedy 또는 multi-sample** 프로토콜로 버전 비교 고정 |
| 5 | pretrain_v2 단독 채팅 벤치 추가 (베이스 효과 분리) |
| 6 | DPO / 교정 SFT / RLVR |
| 7 | **같은 14문항**으로 버전 비교 유지 |

### 📐 해석 가이드 (다시 한 번)

```text
❌  “SFT를 더 돌릴수록 v5 > v4 > v3”
✅  “베이스 분기 + 프로토콜 변경을 분리해서 읽기”
    · v3→v4: 같은 베이스 · 같은 temp0.7 · SFT 믹스 변화 (횡보 + 노이즈)
    · v4→v5: 같은 SFT 믹스 · 다른 베이스 · greedy 전환 (최고 점수)
```

---

### 📁 원본 파일

| 경로 | 내용 |
|:-----|:-----|
| [`ckpt/benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) | ⭐ **최신** SFT v5 (greedy · pretrain_v2) |
| [`ckpt/benchmark_sft_base_v4.json`](ckpt/benchmark_sft_base_v4.json) | SFT v4 |
| [`ckpt/benchmark_sft_base_v3.json`](ckpt/benchmark_sft_base_v3.json) | SFT v3 |
| [`ckpt/benchmark_sft_base_v2.json`](ckpt/benchmark_sft_base_v2.json) | SFT v2 |
| [`ckpt/benchmark_sft_base_v1.json`](ckpt/benchmark_sft_base_v1.json) | SFT v1 |
| [`ckpt/benchmark_pretrain_base_v1.json`](ckpt/benchmark_pretrain_base_v1.json) | Pretrain v1 |
| — | pretrain_v2 채팅 벤치 **미측정** |

> 가중치(`.pt`)와 원본 코퍼스는 공개 저장소에 포함하지 않습니다.

---

<div align="center">

[README](README.md) · [ARCHITECTURE](ARCHITECTURE.md) · [POST-TRAINING](POST-TRAINING.md)

</div>
