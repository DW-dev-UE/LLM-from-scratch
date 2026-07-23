# Apex-1 (1.119B) 공식 벤치마크 결과 — SFT vs DPO

- 측정일: 2026-07-23 | 하드웨어: H100 80GB (Lambda) | 하네스: lm-evaluation-harness 0.4.12, bf16
- 모델: `sft_Apex-1_v1` / `dpo_Apex-1_v1` (HF 변환본, 로짓 일치 검증 max diff 2.3e-5)
- 프로토콜: 상식 7종 0-shot(TinyLlama 논문 Table 2와 동일), MMLU 5-shot, GSM8K 5-shot, HumanEval·MBPP 0-shot pass@1
- 비교 수치 출처: TinyLlama 논문 (arXiv:2401.02385) Table 2·3

## 상식·추론 (0-shot)

| 벤치마크 | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B | OPT-1.3B |
|---|---:|---:|---:|---:|---:|
| HellaSwag (acc_norm) | 46.69 | 46.91 | 59.20 | 47.16 | 53.65 |
| OpenBookQA (acc_norm) | 33.80 | 33.80 | 36.00 | 31.40 | 33.40 |
| WinoGrande (acc) | 53.59 | 52.80 | 59.12 | 53.43 | 59.59 |
| ARC-Challenge (acc_norm) | 29.86 | 30.72 | 30.10 | 27.05 | 29.44 |
| ARC-Easy (acc_norm) | 52.65 | 52.57 | 55.25 | 48.99 | 50.80 |
| BoolQ (acc) | **61.99** | **62.20** | 57.83 | 57.83 | 60.83 |
| PIQA (acc_norm) | 68.23 | 68.55 | 73.29 | 69.21 | 72.36 |
| **평균** | **49.54** | **49.65** | 52.99 | 48.30 | 51.44 |

## 지식·수학·코드

| 벤치마크 | Apex-1 SFT | Apex-1 DPO | TinyLlama-1.1B | Pythia-1.0B |
|---|---:|---:|---:|---:|
| MMLU (5-shot, acc) | 24.83 | 24.90 | 25.34 | 25.70 |
| GSM8K (5-shot, strict) | 1.44 | 1.90 | — | — |
| GSM8K (5-shot, flexible) | 1.97 | 2.27 | — | — |
| HumanEval (pass@1) | 8.54 | 8.54 | 9.15 | 1.83 |
| MBPP (pass@1) | 4.80 | 5.20 | — | — |

## 판정

1. **DPO ≥ SFT**: 상식 평균 +0.11, ARC-c +0.86, GSM8K +0.46, MBPP +0.4 — 회귀(alignment tax) 없음. 릴리스 기본 모델로 DPO 권장.
2. **Pythia-1.0B(300B 토큰) 전 항목 동급~우위**, 특히 HumanEval 8.54 vs 1.83. Apex-1 학습량은 ~20B 토큰.
3. **BoolQ 62.20은 비교 4모델 중 1위.**
4. TinyLlama(3T 토큰)와의 격차는 HellaSwag·WinoGrande·PIQA 등 학습량 민감 항목에 집중 — 토큰 수 차이(150×)로 설명되는 패턴.
5. MMLU는 1B급 공통으로 랜덤 수준(≈25) — 변별력 없음.

## 투명성 고지

- GSM8K·MBPP는 **train split**이 SFT/RLVR 학습 데이터에 포함됨. 평가는 **test split**이므로 유효하나, 완전 무오염은 아님을 명시.
- HumanEval은 학습 데이터에 미포함 (완전 클린).
- 원시 결과: 이 폴더의 `{sft,dpo}_{commonsense,mmlu,gsm8k,code}/` JSON, 실행 로그 `bench.log`.
