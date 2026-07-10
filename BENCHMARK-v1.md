<div align="center">

# 벤치마크 리포트 — Base V1

<img src="https://img.shields.io/badge/checkpoint-sft__base__v1-blue?style=flat" alt="sft_base_v1" />
<img src="https://img.shields.io/badge/params-~327M-informational?style=flat" alt="params" />
<img src="https://img.shields.io/badge/status-early--stage-orange?style=flat" alt="early-stage" />

**[한국어](BENCHMARK-v1.md)** &nbsp;·&nbsp; [English](BENCHMARK-v1.en.md) &nbsp;·&nbsp; [日本語](BENCHMARK-v1.ja.md)

[← README.md](README.md) &nbsp;|&nbsp; 원본 JSON: [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json) · [`benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json)

</div>

---

> **한 줄 요약**  
> `pretrain_base_v1`은 채팅 형식에 전혀 반응하지 않는다(0/28).  
> `sft_base_v1`(SFT 5,000 steps)은 **지시 따르기를 시도**하기 시작했지만, 정답률은 아직 낮다.  
> **THINKING 모드는 전 문항 답변 공란(0/14)** — 당분간 `--no-thinking` 사용을 권장한다.

## 목차

1. [측정 조건](#1-측정-조건)
2. [체크포인트](#2-체크포인트)
3. [핵심 결과 요약](#3-핵심-결과-요약)
4. [Pretrain vs SFT](#4-pretrain-vs-sft)
5. [카테고리별 상세](#5-카테고리별-상세)
6. [코딩 벤치마크](#6-코딩-벤치마크)
7. [해석과 다음 단계](#7-해석과-다음-단계)
8. [원본 데이터](#8-원본-데이터)

---

## 1. 측정 조건

| 항목 | 내용 |
|---|---|
| 프롬프트 수 | 14개 × 2 모드(THINKING on / off) = **28 생성** |
| 카테고리 | 한국어 QA·요약 · 일본어 QA·번역 · 영어 QA·요약 · Python 코딩 5문항 |
| QA / open 채점 | 0~5점 (트랜스크립트 기반 정성 채점) |
| 코딩 채점 | 유닛테스트 실행 (객관식 pass/fail) |
| 생성 일시 | 2026-07-09 |
| 채점 일시 | 2026-07-10 |

동일 문항·동일 채점 기준으로 pretrain 체크포인트와 SFT 체크포인트를 비교했다.

---

## 2. 체크포인트

| 이름 | 경로 | 학습 | 역할 |
|---|---|---|---|
| **pretrain_base_v1** | `ckpt/pretrain_base_v1.pt` | pretrain 18,000 steps, preset `base` | 순수 next-token 사전학습 |
| **sft_base_v1** | `ckpt/sft_base_v1.pt` | SFT 5,000 steps ← pretrain 위 | 대화·지시 미세조정 (본 리포트 주 대상) |

- 파라미터: 약 **326.7M** (`base` 프리셋)
- 토크나이저: BPE vocab 64k (실전 학습 버전)
- 메타: [`llm/ckpt/SFT_base_v1/sft_base_v1.json`](llm/ckpt/SFT_base_v1/sft_base_v1.json)

---

## 3. 핵심 결과 요약

### sft_base_v1 — THINKING 모드

| 지표 | 값 |
|---|---|
| 답변 생성 성공 | **0 / 14 (0%)** |
| 평균 점수 | **0.0** |

> **치명적 이슈**  
> 모든 THINKING 생성에서 모델이 `</THINKING>`을 닫기 전에 `<|eos|>`를 내보낸다.  
> `infer.py`는 이 경우 출력을 전부 thinking 버킷에 넣고 **answer를 빈 문자열로 반환**한다.  
> → 이 체크포인트로 채팅할 때는 **`--no-thinking`** 을 쓴다.

### sft_base_v1 — no-thinking 모드

| 영역 | 평균 / 통과율 |
|---|---|
| 한국어 (3문항) | **1.33 / 5** |
| 일본어 (3문항) | **0.33 / 5** |
| 영어 (3문항) | **2.67 / 5** |
| 코딩 테스트 | **4 / 17 (23.5%)** |
| 코딩 문제 완전 통과 | **1 / 5** (`is_palindrome`만) |

영어 사실 회상(Paris)이 가장 강하고, 한·일 사실/연산 QA와 지시 수행(번역·요약)은 약하다. 코딩은 대부분 실패, 회문 검사 1문항만 완전 정답.

---

## 4. Pretrain vs SFT

| 지표 | pretrain_base_v1 | sft_base_v1 |
|---|---|---|
| 전체 점수(28문항) | **0 / 28** | non-zero 다수 (no-think QA·open 6/9 등) |
| 코딩 테스트 (양쪽 모드) | **0 / 34** | no-think **4 / 17** |
| 지시 따르기 | 질문 에코 / 반복 루프 | 형식을 맞추려 시도 (내용은 자주 틀림) |
| 해석 | chat 토큰·THINKING 미학습 → 기대한 실패 | SFT로 “시도는 하는” 단계로 점프 |

SFT 5,000 step만으로도 *“전혀 지시 불가 → 지시 시도(내용 오류 다수)”* 로의 점프는 분명하다. 다만 실사용 품질과는 아직 거리가 있다.

---

## 5. 카테고리별 상세

점수는 **no-thinking** 기준 (THINKING은 전부 score 0 · 답 공란).

### 5.1 한국어

| ID | 유형 | 점수 | 키워드 | 요약 |
|---|---|---|---|---|
| `ko_fact_1` | QA | **2** | 서울 ✓ | “서울”은 나오나 자기모순·순환 논리 |
| `ko_math_1` | QA | **1** | 2 ✗ | 뺄셈 미수행, 오답 |
| `ko_summary_1` | open | **1** | — | 요약 대신 원문 그대로 반복 |

### 5.2 일본어

| ID | 유형 | 점수 | 키워드 | 요약 |
|---|---|---|---|---|
| `ja_fact_1` | QA | **0** | 東京 ✗ | 수도 대신 인구 얘기, 순환 논리 |
| `ja_math_1` | QA | **1** | 5 ✗ | 질문 재진술만, 연산 없음 |
| `ja_translate_1` | open | **0** | — | 영어 번역 지시 무시, 일본어만 출력 |

### 5.3 영어

| ID | 유형 | 점수 | 키워드 | 요약 |
|---|---|---|---|---|
| `en_fact_1` | QA | **4** | Paris ✓ | 핵심 정답, 이후 지리 환각·장황함 |
| `en_math_1` | QA | **1** | 120 ✗ | 잘못된 계수(×0.5), 120 미도출 |
| `en_summary_1` | open | **3** | — | 요지 파악, but 4문장 + 환각 디테일 |

---

## 6. 코딩 벤치마크

프롬프트는 함수 시그니처 수준(`is_prime(n)` 등). 추출 코드를 유닛테스트로 실행.

### no-thinking

| ID | 함수 | 테스트 | 점수 | 비고 |
|---|---|---|---|---|
| `code_prime` | `is_prime` | 0/5 | 0 | 사실상 `is_even` 로직 |
| `code_reverse` | `reverse_string` | 0/3 | 0 | `.lower()` 반환 + 마크다운 fence 누출 SyntaxError |
| `code_factorial` | `factorial` | 0/3 | 1 | 재귀 형태는 맞으나 `n==0` base 누락 |
| `code_palindrome` | `is_palindrome` | **3/3** | **5** | 전체 코딩 중 **유일한 완전 통과** |
| `code_maxlist` | `find_max` | 1/3 | 1 | 변수 섀도잉·비교 오류, 단일 원소만 우연 통과 |

### THINKING 모드 (코딩)

5문항 전부 **0/테스트 · score 0** — `</THINKING>` 미종료로 코드 추출 불가(또는 산문만 추출).

---

## 7. 해석과 다음 단계

### 잘된 점

- Pretrain → SFT 전환이 **벤치마크로 측정 가능한 능력 점프**를 만듦
- 영어 사실 QA·일부 지시 형식, 회문 코드 1건은 신호로 유효
- 실패 모드가 문서화된 로드맵(초기 검증 단계)과 일치 — 회귀가 아님

### 막힌 점

1. **THINKING 종료 버그** — 실사용 블로커 (v2에서 대부분 완화, 별도 벤치 예정)
2. **산술·다국어 사실** — 한·일 약함
3. **지시 엄수** — 요약·번역에서 에코·지시 무시
4. **코딩** — 1/5 완전 통과 수준

### 권장 후속 (프로젝트 로드맵과 정합)

- THINKING 종료 패턴을 강화한 SFT (이미 v2 방향)
- 코딩·산술 SFT / RLVR 비중 확대
- DPO·교정 SFT로 지시 따르기 정렬
- 동일 14문항 세트로 버전 간 비교 유지

---

## 8. 원본 데이터

| 파일 | 내용 |
|---|---|
| [`llm/ckpt/benchmark_sft_base_v1.json`](llm/ckpt/benchmark_sft_base_v1.json) | SFT v1 전체 문항·점수·노트 |
| [`llm/ckpt/benchmark_pretrain_base_v1.json`](llm/ckpt/benchmark_pretrain_base_v1.json) | Pretrain 동일 문항 비교 |
| [`llm/ckpt/SFT_base_v1/sft_base_v1.json`](llm/ckpt/SFT_base_v1/sft_base_v1.json) | 학습 메타(steps, lr, 데이터 해시) |

이 리포트는 위 JSON을 사람이 읽기 쉽게 정리한 것이다. 수치·인용은 원본 JSON을 따른다.

<div align="center">

---

[← README.md](README.md) &nbsp;|&nbsp; [ARCHITECTURE.md](ARCHITECTURE.md) &nbsp;|&nbsp; [POST-TRAINING.md](POST-TRAINING.md)

</div>
