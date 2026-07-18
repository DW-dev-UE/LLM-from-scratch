[![KO](https://img.shields.io/badge/KO-0969da)](BENCHMARK-v1.md) [![EN](https://img.shields.io/badge/EN-lightgrey)](BENCHMARK-v1.en.md) [![JA](https://img.shields.io/badge/JA-lightgrey)](BENCHMARK-v1.ja.md)

[← README](README.md)

---

# 📊 BENCHMARK · base ~327M

> **학습 일기**에 가깝습니다. 예쁜 점수표보다  
> *어떻게 학습했고 · 뭘 썼고 · 뭐라고 답했는지* 를 남기는 문서입니다.

| | |
|:--|:--|
| 🆕 **최신** | `sft_base_v6` · 2026-07-15 |
| 📦 **모델** | Decoder-only · 약 **326.7M** (`base`) |
| 🧪 **세트** | 동일 **14문항 × THINKING on/off** (버전 비교용) |
| 📁 **원본** | [`ckpt/benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) |

> [!WARNING]
> 아직 **실사용 단계가 아닙니다.**  
> 파이프라인이 도는지, 체크포인트·프로토콜이 바뀌면 뭐가 달라지는지를 보는 스냅샷입니다.

### 📑 목차

1. [한눈에 보기](#1-한눈에-보기)
2. [학습 과정](#2-학습-과정)
3. [데이터셋](#3-데이터셋)
4. [채점 방법](#4-채점-방법)
5. [sft_base_v6 · 최신 결과](#5-sft_base_v6--최신-결과)
6. [이전 버전 아카이브](#6-이전-버전-아카이브)
7. [느낀 점 · 다음에](#7-느낀-점--다음에)

---

## 1. 한눈에 보기

버전을 한 줄로 비교합니다. 굵은 칸이 **지금 보는 최신**입니다.

> [!WARNING]
> **비교 주의**  
> · **프로토콜**: v1–v4는 temp `0.7` 단일 샘플 · **v5·v6는 greedy (temp `0.0`)**  
> · **베이스 분기**: v1–v4는 `pretrain_base_v1` · **v5·v6는 `pretrain_base_v2` 위 SFT**  
> · **v5→v6**: 베이스·SFT 소스 믹스 모두 동일, **prep_sft 전처리 + 추론 예산 분리만** 변경 — 이 프로젝트에서 가장 깨끗한 단일 변수 비교  
> · 따라서 **v6 > v5 > v4 > v3 를 “순수 SFT 사다리”로 읽으면 안 됩니다.**  
> · 버전 간 코딩 비교는 테스트 분모 차이 때문에 **완전 통과 N/5** 가 안전합니다.

| | Pretrain v1 | SFT v1 | SFT v2 | SFT v3 | SFT v4 | SFT v5 | ⭐ **SFT v6** |
|:--|:-----------:|:------:|:------:|:------:|:------:|:------:|:------------:|
| 일반 채팅 평균 | ~0 | 낮음† | 낮음† | 2.57 | 2.07 | 3.21 | **3.93 / 5** |
| 코딩 완전 통과 (일반) | 0/5 | 1/5 | 1/5 | 4/5 | 3/5 | 5/5 | **5/5** |
| 코딩 완전 통과 (사고) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **4/5** (+prime 3/5) |
| THINKING 비공란 답 | — | **0/14** | **13/14** | 12/14 | 11/14 | 10/14 | **14/14** |
| THINKING 평균 | ~0 | 0.0 | 0.71 | 0.21 | 0.50 | 0.50 | **2.93** |
| 한국어 합 (사고+일반) | 0/30 | — | 2/30 | 0/30 | 1/30 | 8/30 | **10/30** |
| 베이스 | v1 | v1 | v1 | v1 | v1 | v2 | **v2** |
| 샘플링 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.0 greedy | **0.0 greedy** |
| 인상 | 채팅 불가 | 형식 시도 | 태그 복구 | 코딩↑ · KO↓ | 횡보 · 노이즈 | 코딩 만점 · KO↑ | 핸드오프 해결 · 사고 코딩 개통 |

† v1/v2 일반 채팅은 언어별 평균만 기록 (전체 평균 미집계):  
v1 KO **1.33** / JA **0.33** / EN **2.67** · v2 KO **0.33** / JA **2.0** / EN **1.0**

```text
                    ┌─► sft_v1 ─► sft_v2 ─► sft_v3 ─► sft_v4
pretrain_v1 ────────┤     형식      태그닫기    코딩4/5     KO보강·횡보
   18k · lr 6e-4    │
                    └─► pretrain_v2 (25k · lr 1e-4 · corpus v2)
                              │
                              ├─► sft_v5    (동일 sft.pt as v4 · greedy 벤치)
                              │               코딩 5/5 · 일반 3.21 · KO 8/30
                              │
                              └─► sft_v6 ★  (동일 믹스 · prep_sft 개선 + 추론 예산 분리)
                                              사고 평균 0.50→2.93 · 사고 코딩 0/5→4/5
```

| 버전 | 잘된 것 | 막힌 것 |
|:----:|:--------|:--------|
| v1 | 지시 형식을 따라 보려 함 | THINKING 답 전부 비어 있음 |
| v2 | `</THINKING>` 종료 대부분 복구 | 코딩·품질은 여전히 약함 |
| v3 | 코딩 **4/5**, 영어 fact/math 회복 | **한국어 붕괴**, THINKING 코딩 0/5 |
| v4 | KO SFT 비중↑ · thinking 안 서울 정답 등장 | 벤치 KO는 거의 회복 안 됨 · item flip 노이즈 |
| v5 | **코딩 5/5**, 일반 **3.21**, KO **8/30** | **thinking→answer 핸드오프** · thinking 코딩 0/5 |
| **v6** | **핸드오프 해결**(14/14 비공란) · 사고 평균 **2.93** · 사고 코딩 **4/5** | 한국어 산술 여전히 붕괴 · 반복/퇴화 · JA→EN 번역 |

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
                ├─► ⑦ SFT v5 · 9.2k · (v4와 동일 sft.pt) ──► sft_base_v5
                └─► ⑧ SFT v6 · 9.5k · prep_sft 개선 + 추론 예산 분리 ──► sft_base_v6 ★
```

| 단계 | 체크포인트 | steps | init | 데이터 해시 · 메모 |
|:-----|:-----------|------:|:-----|:-------------------|
| Pretrain v1 | `pretrain_base_v1` | 18,000 | — | `53f815d47abc4887` · lr 6e-4 |
| SFT v1 | `sft_base_v1` | 5,000 | pretrain_v1 | `bbc2211091309d3c` |
| SFT v2 | `sft_base_v2` | 5,700 | pretrain_v1 | `fc177a36965df49a` · THINKING 종료 |
| SFT v3 | `sft_base_v3` | 8,000 | pretrain_v1 | `db63d09ddfa41388` · 코딩 보강 |
| SFT v4 | `sft_base_v4` | 9,200 | pretrain_v1 | `975c1771bfff1919` · KO +70k |
| Pretrain v2 | `pretrain_base_v2` | 25,000 | pretrain_v1 | `4ad58fc7307962c0` · lr 1e-4 |
| SFT v5 | `sft_base_v5` | 9,200 | pretrain_v2 | `975c1771bfff1919` (v4와 동일) |
| ⭐ SFT v6 | `sft_base_v6` | 9,500 | **pretrain_v2** | 동일 14파일 믹스 · **`prep_sft` 개선** (thinking tail-preservation: trimmed 47,273 / demoted 17,540) |

공통 설정: **AdamW** (β 0.9 / 0.95, wd 0.1) · grad clip 1.0 · warmup + cosine · CUDA · SFT lr `3e-5` · v6은 A100 40GB · batch 8 × accum 16

> [!IMPORTANT]
> 🔑 **v5의 핵심 변수**는 SFT 믹스가 아니라 **베이스(pretrain_v2)** 입니다.  
> v4와 v5는 같은 `sft.pt` (`975c…`) · 같은 9,200 step · 다른 init 가중치입니다.

> [!IMPORTANT]
> 🔑 **v6의 핵심 변수**는 베이스도 SFT 소스 믹스도 아니라 **데이터 전처리와 추론 코드**입니다.  
> `data.py`의 `prep_sft`가 긴 THINKING 구간을 다루는 방식을 바꿨고 (꼬리 보존 위주로 trim/demote),  
> `infer.py`는 thinking과 answer가 하나의 `max_new` 예산을 공유하던 버그를 고쳐 **둘을 분리된 예산**으로 생성합니다.  
> 베이스·SFT 원본 파일이 v5와 완전히 같기 때문에, v5→v6 델타는 **이 두 가지 수정만의 순수 효과**로 읽을 수 있습니다.

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

> [!TIP]
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

> [!NOTE]
> ko-wiki val loss는 v1 대비 소폭 악화(보고: 2.504 → 2.605)였지만,  
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
| v5 생성 | temp **`0.0` greedy** · top_p `0.9` · max_new `256` · seed `0` (thinking·answer 예산 공유) |
| **v6 생성** | temp **`0.0` greedy** · top_p `0.9` · max_new `256` · seed `0` · **thinking/answer 분리 예산** |
| 일시 | v1 `07-09/10` · v2 `07-10` · v3 `07-13` · v4 `07-13` · v5 `07-14` · **v6 `07-15`** (2026) |

> [!NOTE]
> 📌 코딩 테스트 분모: v1/v2 합 **17** · v3+ 합 **25** (문제당 5케이스).  
> 버전 비교는 **완전 통과 N/5** 를 보는 편이 안전합니다.

> [!WARNING]
> 🎲 **샘플링 노이즈 (v4 교훈)**  
> temp 0.7 · 단일 샘플에서 v3↔v4 item flip이 컸습니다  
> (예: en_fact 5→0, code_prime 5→0, code_maxlist 5→0, code_reverse 0→5).  
> **v5부터 greedy** 로 바꿔 버전 내 노이즈를 제거했습니다.  
> 다만 v5 수치를 v3/v4와 나란히 둘 때는 **프로토콜 차이 주석이 필수**입니다.

> [!WARNING]
> 🐛 **예산 충돌 버그 (v6 교훈)**  
> v6의 첫 벤치런은 `max_think`와 `max_new`가 같은 예산(256)을 공유해  
> thinking이 길어지면 answer 예산이 0으로 남는 버그가 있었습니다.  
> `generate()`에서 두 예산을 분리한 뒤 재측정한 결과가 아래 v6 수치입니다.

> [!NOTE]
> 🇰🇷 한국어 합 점수: 사고+일반 각 3문항 × 0–5 = **최대 30점**.

---

## 5. sft_base_v6 · 최신 결과

| | |
|:--|:--|
| 🏁 체크포인트 | `ckpt/sft_base_v6.pt` |
| 🧱 베이스 | `pretrain_base_v2` (v5와 동일) |
| 📦 SFT 데이터 | v4·v5와 동일 14파일 믹스 · `prep_sft` 개선 (thinking tail-preservation: trimmed 47,273 / demoted 17,540) · 9,500 step |
| 🔧 추론 수정 | `infer.py` EOS 가드 + thinking/answer **분리 예산** |
| 🎲 생성 | greedy · temp 0.0 |
| 📁 JSON | [`benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) |

> [!WARNING]
> 첫 v6 벤치런은 `max_think`와 `max_new`가 예산을 공유해 답이 비는 버그가 있었습니다.  
> `generate()`에서 예산을 분리한 뒤 재측정한 것이 아래 결과입니다 (§4 참고).

### 5.1 점수 카드

#### 💬 일반 채팅 (THINKING 끔) — 전체 평균 **3.93 / 5**

| 영역 | 점수 | 상태 |
|:-----|:----:|:----:|
| 🇰🇷 한국어 | **2.67 / 5** (5 / 0 / 3) | 🟢 fact 첫 만점 |
| 🇯🇵 일본어 | **3.00 / 5** (5 / 4 / 0) | 🟢 fact+math |
| 🇺🇸 영어 | **4.33 / 5** (5 / 5 / 3) | 🟢 |
| 💻 코딩 | **25/25** 테스트 · **5/5** 완전 통과 | 🟢 만점 유지 |

#### 🧠 THINKING 켬 — 평균 **2.93 / 5** (v5 대비 약 6배)

| 지표 | 값 | 상태 |
|:-----|:--:|:----:|
| 평균 점수 | **2.93 / 5** | 🟢 (v5 0.50) |
| 비어 있지 않은 답 | **14 / 14** | 🟢 전부 (v5 10/14) |
| 코딩 완전 통과 | **4 / 5** (+prime 부분 3/5) | 🟢 프로젝트 최초 (v5 0/5) |
| 한국어 합 (사고) | 2 / 15 | 🔴 여전히 약함 |

### 5.2 v5 → v6 하이라이트

| | 변화 |
|:--|:-----|
| ✅ | **THINKING 핸드오프 문제 해결** — 빈 답 4/14 → **0/14** (전부 답변 생성) |
| ✅ | THINKING 평균 **0.50 → 2.93** (약 6배) |
| ✅ | THINKING 코딩 **0/5 → 4/5** (+prime 3/5 부분 통과) — **6개 버전 만에 첫 코드 출력** |
| ✅ | 일반 채팅 평균도 동반 상승 **3.21 → 3.93** |
| ✅ | 한국어 fact **최초 만점**(5점): “대한민국의 수도는 서울특별시입니다.” |
| ✅ | 일본어 수학 **양쪽 모드 모두 첫 정답** (7 − 2 = 5) |
| ✅ | 한국어 합 **8/30 → 10/30** |
| 🔬 | 같은 베이스 · 같은 SFT 소스 믹스(14파일) · **prep_sft 전처리 + 추론 예산 분리만 변경** — 이 프로젝트에서 가장 깨끗한 단일 변수 비교 |
| ❌ | 한국어 산술 여전히 붕괴 (사고: 5−3=3, 일반: 3+3=9) |
| ❌ | 긴 thinking·답변 후반부 반복/퇴화 패턴 지속 (예: en_fact 사고에서 “Wait, the capital of France is Paris.” 반복) — RL/DPO 영역 과제로 이월 |
| ❌ | 일본어 번역은 양쪽 모드 모두 실패 |

---

### 5.3 질문과 답변 · 일반 채팅 (THINKING 끔)

#### 🇰🇷 한국어 — 평균 2.67

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **5** | 서울 | “대한민국의 수도는 서울특별시입니다.” | **한국어 사실 문항 첫 만점** 🟢 |
| 사과 5−3 | **0** | 2 | “3(사과) + 3(사과) = 9개” | 뺄셈이 덧셈으로, 산식 붕괴 |
| 요약 | **3** | 한 문장 | “오늘 날씨가 매우 좋아서 공원에 산책을 나갔다.” | v5와 동일 수준 유지 |

#### 🇯🇵 일본어 — 평균 3.00

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **5** | 東京 | 「日本の首都は東京です。」 | 완벽 |
| 사과 7−2 | **4** | 5 | 「7 - 2 = 5個です」 | **일본어 수학 첫 정답** · 동사 오류로 소폭 감점 |
| 번역 | **0** | 영문 | 원문을 그대로 반복 | 영어 없음 |

#### 🇺🇸 영어 — 평균 4.33

| 문항 | 점수 | 기대 | 모델이 한 일 | 코멘트 |
|:-----|:----:|:----:|:-------------|:-------|
| 수도 | **5** | Paris | “The capital of France is Paris.” | 완벽 |
| 기차 60×2 | **5** | 120 | 공식 대입 → “120 km” | 완벽 (v5는 부분점수 3) |
| 요약 | **3** | 한 문장 | “The weather was nice, and the children were having a great time.” | 산책/공원 정보는 탈락 |

---

### 5.4 질문과 답변 · THINKING 모드

v5까지 이 자리는 “핸드오프 실패” 사례 나열이었습니다. v6는 **답변 칸이 14/14 전부 채워졌습니다** — 다만 사고 과정의 정확도는 아직 답변보다 낮습니다.

| 문항 | 점수 | thinking 안 | answer 칸 | 코멘트 |
|:-----|:----:|:------------|:----------|:-------|
| ko_fact | **1** | “서울특별시입니다” 정답으로 시작하나 “거리 1km” 반복으로 탈선 | 거리 관련 잡문 (서울 엔티티만 존재) | 인출은 됐으나 탈선 |
| ko_math | **0** | “5개 − 3개 = 3개” (오답) | “사과를 3개” 무한 반복 | 산식·답 모두 붕괴 |
| ko_summary | **1** | 원문 재진술 후 상상 문구 반복 | “아이들이 뛰어놀고 있는 모습을 상상해 보세요.” | 요약 아닌 명령문 |
| ja_fact | **5** | 혼란스러운 추론 | 「答えは東京です。」 | **깔끔한 정답 — 핸드오프 성공 전형** |
| ja_math | **4** | 산식 붕괴 (1.7, 5.7 등) | 「答えは5です。」 | 사고는 틀렸지만 답은 정답 |
| ja_translate | **0** | 영어 문장 반복 (오역) | 따옴표만 나열 | 실패 |
| en_fact | **4** | “Wait, the capital of France is Paris.” 반복 루프 | “The capital of France is **Paris**.” + 반복 설명 | 정답 명시, 장황함으로 감점 |
| en_math | **0** | “60/2=30”, “30/2=15” | “The answer is 15.” | 산식 붕괴, 오답 |
| en_summary | **3** | 짧고 무난한 요약 | “The answer is that the park was nice and the children were playing.” | 요약 성립, 프레임 어색 |

> [!IMPORTANT]
> 🔑 **가장 중요한 변화**: v5는 ko_fact·ja_fact·en_summary에서 thinking 안에 정답이 있어도 답 칸이 비었습니다.  
> v6는 같은 유형 문항에서 **답 칸에 내용이 채워집니다** — 정확도는 아직 들쑥날쑥해도 “전달 안 됨” 자체는 해소됐습니다.

---

### 5.5 코딩 · 두 모드 비교

일반 채팅은 v5에 이어 **5/5 만점**을 유지합니다. THINKING 모드는 이번이 **프로젝트 최초로 실제 코드를 출력**한 버전입니다.

| 문제 | 일반 채팅 | THINKING | 코멘트 |
|:-----|:--------:|:--------:|:-------|
| `is_prime` | 5/5 | **3/5** | 사고 모드: 나머지 연산을 17까지 하드코딩 나열 — 일반화 실패로 부분 통과 |
| `reverse_string` | 5/5 | **5/5** | 사고 모드에서도 `s[::-1]` 그대로 통과 (thinking은 공란) |
| `factorial` | 5/5 | **5/5** | 재귀 base case, 양쪽 모드 동일 |
| `is_palindrome` | 5/5 | **5/5** | 사고 모드는 `s.strip()`, 일반은 `s.lower()` — 둘 다 통과 |
| `find_max` | 5/5 | **5/5** | 양쪽 모드 모두 내장 `max()` 활용 |

<details>
<summary><b>✅ THINKING 모드 통과 코드 펼치기</b></summary>

<br/>

```python
def is_prime(n):  # 3/5 부분 통과 — 17까지 나머지 연산 하드코딩
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    # ... n % 17 == 0 까지 반복 (일반화된 √n 분해 아님)

def reverse_string(s):
    return s[::-1]

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def is_palindrome(s):
    s = s.strip()
    return s == s[::-1]

def find_max(lst):
    max_num = max(lst)
    for num in lst:
        if num > max_num:
            max_num = num
    return max_num
```

</details>

---

### 5.6 남은 병목

thinking→answer 핸드오프가 풀리면서, 남은 문제는 **형식이 아니라 정확도**로 분명해졌습니다.

| 문제 | 예시 | 성격 |
|:-----|:-----|:-----|
| 🇰🇷 한국어 산술 | 5−3을 “3+3=9”(일반) / “5개−3개=3개”(사고)로 계산 | 지식·연산 오류 — 데이터 보강 필요 |
| 🔁 반복·퇴화 | en_fact 사고에서 같은 문장 반복, ko_fact 사고에서 “1km” 반복 | 디코딩 품질 — RL/DPO 영역 |
| 🇯🇵→🇬🇧 번역 | 원문 반복 또는 무관한 영어 문장 나열 | 번역 능력 자체 미형성 |
| 코딩 일반화 | 사고 모드 `is_prime`이 나머지 연산 17개 하드코딩 (분해 로직 아님) | 부분 이해 — 완전한 알고리즘 학습 아직 |

> [!NOTE]
> 📌 v1의 “닫기 실패” → v5의 “닫지만 전달 안 됨” → v6는 **“전달되지만 내용이 가끔 틀림”** 단계로 병목이 이동했습니다.  
> 형식 학습은 사실상 마무리 단계로 보입니다.

---

## 6. 이전 버전 아카이브

최신과 비교용입니다. 필요할 때만 펼치면 됩니다.

<details>
<summary><b>📔 sft_base_v5 · 2026-07-14 · pretrain_v2 · greedy</b></summary>

<br/>

📁 [`benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json)

첫 greedy(결정적) 벤치. 베이스를 `pretrain_base_v2`로 교체 (SFT 믹스는 v4와 동일).
일반 채팅 코딩 첫 만점, 한국어 회복 시작. THINKING 핸드오프(태그 뒤 답변 전달 실패)가 새 병목으로 드러남 (v6에서 해결).

#### 점수

| 모드 | 결과 |
|:-----|:-----|
| 일반 평균 | **3.21 / 5** |
| 🧠 THINKING 평균 | **0.50 / 5** · 비공란 답 **10/14** |
| 🇰🇷 일반 | **1.67/5** (2 / 0 / 3) |
| 🇯🇵 일반 | **1.33/5** (4 / 0 / 0) |
| 🇺🇸 일반 | **3.67/5** (5 / 3 / 3) |
| 💻 일반 | 완전 **5/5** · 테스트 25/25 (첫 만점) |
| 🇰🇷 합 (사고+일반) | **8/30** |

#### 코딩 (일반 · greedy)

| 문제 | 테스트 | 메모 |
|:-----|:------:|:-----|
| is_prime | 5/5 | √n 시도분해 |
| reverse_string | 5/5 | `s[::-1]` |
| factorial | 5/5 | base case 재귀 |
| is_palindrome | 5/5 | lower + 뒤집기 |
| find_max | 5/5 | 내장 `max()` |

#### 하이라이트

| | |
|:--|:--|
| ✅ | 코딩 완전 통과 3/5 → 5/5 (첫 만점 · 결정적 greedy) |
| ✅ | 한국어 합 1/30 → 8/30 (첫 요약 성공 · 사고 math에 정답 숫자) |
| ⚠️ | 같은 SFT 믹스, 다른 베이스 + greedy 전환 → “SFT만 더 잘됨”이 아님 |
| ❌ | THINKING 코딩 여전히 0/5 · thinking 안 정답이 답 칸으로 전달 안 됨 (핸드오프 병목 발견) |

</details>

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

> [!WARNING]
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

> [!NOTE]
> 📭 **pretrain_base_v2 단독 채팅 벤치는 아직 없음.**  
> v2의 효과는 현재 **sft_base_v5 · sft_base_v6** 다운스트림으로만 관측됩니다.

</details>

---

## 7. 느낀 점 · 다음에

### ✅ 버전에 걸쳐 잘 된 것

- Pretrain loss 11 → ~2, SFT 루프 정상 · 파이프라인 검증 OK
- v1: “형식 시도”가 벤치에 잡힘
- v2: THINKING **닫기** 대부분 복구 (0/14 → 13/14)
- v3: 일반 채팅 **코딩 4/5** — 데이터 보강 목표 달성
- v4: KO SFT 비중↑ · thinking 안 한국어 정답 단문 최초 출현 (전달은 실패)
- v5: 일반 코딩 5/5 · 평균 3.21 · KO 8/30 · greedy로 재현 가능
- **v6: THINKING 핸드오프 해결 · 사고 평균 0.50→2.93 · 사고 코딩 첫 통과(4/5) · 일반 평균도 3.21→3.93 동반 상승** — 현재 최고
- continued pretrain (v2) 이 **같은 SFT 믹스**에서도 다운스트림을 끌어올림
- prep_sft 전처리 + 추론 예산 분리만으로 핸드오프 문제를 해소 (베이스·SFT 소스는 그대로)

### 🚧 아직 막힌 것 (우선순위)

> ✅ v5까지 최우선이었던 “THINKING 핸드오프”와 “THINKING 코딩 포맷”은 **v6에서 해결**되었습니다 (prep_sft 전처리 + 추론 예산 분리). 아래는 그다음 과제입니다.

1. 🇰🇷 한국어 산술·사실 정확도 — 형식은 되나 계산이 자주 틀림 (사고 5−3=3, 일반 3+3=9)
2. 🔁 긴 생성에서 반복·퇴화 패턴 (thinking·답변 모두) — RL/DPO로 넘길 영역
3. 🇯🇵→🇬🇧 번역 능력 미형성 — 원문 반복이 전부
4. 코딩 일반화 — 사고 모드 `is_prime`이 나머지 연산 하드코딩 나열 (√n 분해 아님)
5. 실사용 품질까지는 아직 멀다

### 🧭 다음에

| 우선 | 내용 |
|:----:|:-----|
| 1 | 한국어 fact·산술 추가 보강 (pretrain + SFT 병행) — 유일하게 안 풀린 “정확도” 축 |
| 2 | 반복·퇴화 억제 — DPO / 교정 SFT / RLVR (형식은 되니 이제 품질을 다듬을 차례) |
| 3 | 일본어→영어 번역 데이터 보강 |
| 4 | 코딩 일반화 — THINKING 모드에서도 √n 분해 같은 일반 로직 학습 |
| 5 | pretrain_v2 단독 채팅 벤치 추가 (베이스 효과 분리, 여전히 미측정) |
| 6 | **같은 14문항**으로 버전 비교 유지 |

### 📐 해석 가이드 (다시 한 번)

```text
❌  “SFT를 더 돌릴수록 v6 > v5 > v4 > v3”
✅  “베이스 분기 + 프로토콜 변경 + 전처리 변경을 분리해서 읽기”
    · v3→v4: 같은 베이스 · 같은 temp0.7 · SFT 믹스 변화 (횡보 + 노이즈)
    · v4→v5: 같은 SFT 믹스 · 다른 베이스 · greedy 전환 (최고 점수)
    · v5→v6: 같은 베이스 · 같은 SFT 믹스 · prep_sft 전처리+추론 예산 분리 (핸드오프 해결)
```

---

### 📁 원본 파일

| 경로 | 내용 |
|:-----|:-----|
| [`ckpt/benchmark_sft_base_v6.json`](ckpt/benchmark_sft_base_v6.json) | ⭐ **최신** SFT v6 (핸드오프 해결 · greedy · pretrain_v2) |
| [`ckpt/benchmark_sft_base_v5.json`](ckpt/benchmark_sft_base_v5.json) | SFT v5 (greedy · pretrain_v2) |
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
