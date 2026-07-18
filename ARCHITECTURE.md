# 밑바닥부터 만드는 LLM

[![KO](https://img.shields.io/badge/KO-0969da)](ARCHITECTURE.md) [![EN](https://img.shields.io/badge/EN-lightgrey)](ARCHITECTURE.en.md) [![JA](https://img.shields.io/badge/JA-lightgrey)](ARCHITECTURE.ja.md)

[← README](README.md) · 용어가 막히면 [GLOSSARY](GLOSSARY.md)

---

이 문서는 **모델이 어떻게 생겼고, 어떻게 학습·추론하는지**를 적습니다.
§2에서 트랜스포머 순전파를 숫자 예제로 훑고, §3부터 이 저장소의 실제 설계(토크나이저 · 학습 · 추론 · 사고 모드)로 이어집니다.

---

## 1. 전체 로드맵

처음부터 완벽한 파이프라인을 짠 게 아닙니다. 대략 이 순서로 손이 갔습니다.

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

## 2. 트랜스포머 동작 원리 (교육용 워크스루)

교육용 숫자 예제입니다(GPT-2 스타일 GELU 등). 실제 본 저장소 설계는 다음 절(§3)의 LLaMA 스타일(RMSNorm / RoPE / SwiGLU / GQA)입니다.

### 트랜스포머 아키텍처

> LLM은 어떻게 학습되는지에 관한 서술입니다.

LLM 학습을 위해 데이터셋을 준비해야 합니다. 데이터셋은 '코퍼스'라고도 불리는데, 이번 예제의 코퍼스는 "나는 고양이를 좋아합니다"로 예를 듭니다.


#### 1) 토크나이저

코퍼스를 컴퓨터의 언어로 만드는 도구를 **토크나이저**라 부릅니다.

"나는 고양이를 좋아합니다" → <kbd>나는</kbd> <kbd>고양이</kbd> <kbd>를</kbd> <kbd>좋아</kbd> <kbd>합니다</kbd>

이렇게 나누면 **vocab**이라는 사전이 만들어집니다.

```text
vocab = {"나는": 1, "고양이": 2, "를": 3, "좋아": 4, "합니다": 5}
```

token ids: `[1, 2, 3, 4, 5]`


#### 2) 임베딩 테이블

각 토큰마다 랜덤한 실수로 칸을 채웁니다.

| # | token | d1 | d2 | d3 | d4 |
|--:|-------|---:|---:|---:|---:|
| 1 | 나는 | 0.12 | -0.53 | 0.33 | 0.90 |
| 2 | 고양이 | -0.51 | 0.30 | -2.10 | 0.87 |
| 3 | 를 | 0.05 | -0.44 | 1.32 | -0.06 |
| 4 | 좋아 | 0.71 | 0.18 | -0.29 | 0.55 |
| 5 | 합니다 | -0.33 | 0.92 | 0.14 | -0.78 |

칸의 크기를 **임베딩 차원**(d_model)이라 하며, 차원이 많을수록 토큰을 설명할 자리가 늘어 표현력이 올라갑니다.

실수 하나를 몇 바이트로 저장할지는 **정밀도**(FP32, FP16 등)의 문제입니다.


#### 3) 문제 만들기

토큰열을 한 칸씩 밀어서 입력/정답 쌍을 만듭니다.

입력: `[1, 2, 3, 4]` = "나는 고양이를 좋아"<br>
정답: `[2, 3, 4, 5]` = "고양이를 좋아 합니다"

<kbd>나는</kbd> → ? (정답: 고양이)<br>
<kbd>나는</kbd> <kbd>고양이</kbd> → ? (정답: 를)


#### 4) 정규화

x = 임베딩 값. 문제 [나는, 고양이, 를, 좋아]라고 가정하면 x는 "좋아"의 임베딩입니다.

x = <kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd>

정규화 후 N:

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>


#### 5) 어텐션

문맥을 섞는 단계 — 여기서부터가 트랜스포머 아키텍처의 핵심입니다.<br>
기존 bigram 아키텍처는 문맥을 못 봅니다. 같은 "좋아"라도 앞이 "고양이를"인지 다른 단어인지 구분하지 못합니다.

> [!TIP]
> 📌 **랜덤 행렬 3개가 N을 세 관점으로 변환**

문제 4 — 토큰 `[1, 2, 3, 4]` = "나는 고양이를 좋아", 정답 `[2, 3, 4, 5]` = "고양이를 좋아합니다".<br>
예측을 수행하는 위치는 「좋아」입니다.

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>

Query(질문) `Q = N · W_Q`. W_Q, W_K, W_V는 학습 시작 시 랜덤값이며 학습이 진행될수록 조절됩니다.

K(나는) · Q ÷ √4 = 0.1<br>
K(고양이) · Q ÷ √4 = 2.0<br>
K(를) · Q ÷ √4 = −0.6<br>
K(좋아) · Q ÷ √4 = 0.5

softmax(<kbd>0.1</kbd> <kbd>2.0</kbd> <kbd>-0.6</kbd> <kbd>0.5</kbd>) = <kbd>0.10</kbd> <kbd>0.69</kbd> <kbd>0.05</kbd> <kbd>0.15</kbd>

「좋아」가 「고양이」를 69% 참고했습니다 — 앞 문맥에 따라 같은 단어도 다른 벡터가 되는 이유입니다.

V 가중합: `0.10·V(나는) + 0.69·V(고양이) + 0.05·V(를) + 0.15·V(좋아)`

어텐션 출력 = <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd> — 문맥이 섞인 새 벡터입니다.


#### 6) 잔차 연결

x와 어텐션 출력을 더해 new v를 만듭니다. 원본 의미를 보존하면서 문맥 정보를 얹는 방식입니다.

<kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd> + <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd><br>
= <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd>

아직은 숫자에 불과하지만, 문맥을 파악하고 학습할 준비가 되었습니다.

---

### 블록 (Block)

여기서부터가 블록 아키텍처입니다.

#### 1) 정규화

new v를 정규화합니다.

N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd>


#### 2) FFN 1단계 — W₁ 확장

N₂를 특정 배율로 확장합니다. GPT-2는 4배율로 확장하므로 이번 예제도 동일하게 진행합니다.<br>
어텐션 출력은 V들의 가중합, 즉 섞어서 평균 내는 역할일 뿐이지만 펼쳐진 tensor는 "특정 패턴 감지기"처럼 작동합니다.

확장: `U = N·W₁ + b₁` (4×4 행렬 · 4×16 행렬 = 4×16, b₁은 편향 16칸, 초기값 0이라 생략)

> [!NOTE]
> **범례**: <span style="color:#1d9e75">■</span> 통과 예정(양수) · <span style="color:#d85a30">■</span> 차단 예정(음수) · <span style="color:#888780">■</span> 0에 가까움

**표 A — GELU 적용 전 (U)**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 나는 | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-1.53</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#1d9e75">1.96</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#d85a30">-2.58</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.29</span> | <span style="color:#d85a30">-1.37</span> | <span style="color:#d85a30">-0.28</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-0.26</span> | <span style="color:#d85a30">-0.65</span> | <span style="color:#1d9e75">2.15</span> | <span style="color:#1d9e75">1.48</span> |
| 고양이 | <span style="color:#d85a30">-1.96</span> | <span style="color:#d85a30">-0.90</span> | <span style="color:#1d9e75">1.93</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-1.33</span> | <span style="color:#1d9e75">0.89</span> | <span style="color:#d85a30">-2.79</span> | <span style="color:#1d9e75">0.39</span> | <span style="color:#1d9e75">2.57</span> | <span style="color:#d85a30">-1.80</span> | <span style="color:#d85a30">-0.58</span> | <span style="color:#1d9e75">0.77</span> | <span style="color:#d85a30">-1.60</span> | <span style="color:#1d9e75">1.57</span> | <span style="color:#d85a30">-1.06</span> | <span style="color:#d85a30">-2.67</span> |
| 를 | <span style="color:#1d9e75">2.35</span> | <span style="color:#1d9e75">0.64</span> | <span style="color:#d85a30">-2.43</span> | <span style="color:#d85a30">-2.82</span> | <span style="color:#1d9e75">2.28</span> | <span style="color:#d85a30">-1.91</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">0.50</span> | <span style="color:#d85a30">-2.27</span> | <span style="color:#1d9e75">0.94</span> | <span style="color:#1d9e75">0.31</span> | <span style="color:#d85a30">-0.76</span> | <span style="color:#1d9e75">0.90</span> | <span style="color:#d85a30">-1.31</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.10</span> |
| 좋아 | <span style="color:#d85a30">-0.88</span> | <span style="color:#d85a30">-2.45</span> | <span style="color:#1d9e75">1.63</span> | <span style="color:#1d9e75">3.40</span> | <span style="color:#d85a30">-1.20</span> | <span style="color:#1d9e75">2.67</span> | <span style="color:#d85a30">-3.24</span> | <span style="color:#1d9e75">2.07</span> | <span style="color:#1d9e75">0.79</span> | <span style="color:#d85a30">-0.62</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.21</span> | <span style="color:#1d9e75">0.82</span> | <span style="color:#d85a30">-0.63</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-1.77</span> |


#### 3) FFN 2단계 — GELU

GELU가 없으면 `W₂(W₁·N) = (W₂W₁)·N`, 즉 행렬 하나를 곱한 것과 수학적으로 동일해져서 16칸 확장이 헛수고가 됩니다.<br>
비선형이 끼어야 "이 특징은 켜고, 저 특징은 끈다"는 조건부 동작이 생깁니다.

`GELU(x) = x × Φ(x)` — 입력을 Φ(x)의 비율만큼만 통과시킵니다.<br>
Φ(x)는 표준정규분포에서 x 이하가 나올 확률(0~1)이라, x가 클수록 통과율이 1에 가까워지고 작을수록 0에 가까워집니다.

x가 -2.6이면 통과율 0.5% = <span style="color:#d85a30">-0.01</span> (사실상 차단)<br>
x가 3.10이면 통과율 99.9% = <span style="color:#1d9e75">3.10</span> (사실상 그대로)

단, Φ(x)의 확률은 「다음 토큰이 올 확률」과 무관합니다. 반응값 x가 클수록 많이 통과시키는 고정된 수학적 비율일 뿐이며, 다음 토큰 확률은 softmax에서 처음 나옵니다.

**표 B — GELU 적용 후**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 나는 | <span style="color:#1d9e75">1.37</span> | <span style="color:#d85a30">-0.10</span> | <span style="color:#d85a30">-0.13</span> | <span style="color:#1d9e75">1.91</span> | <span style="color:#1d9e75">2.24</span> | <span style="color:#d85a30">-0.13</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.11</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#d85a30">-0.11</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#d85a30">-0.10</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">2.12</span> | <span style="color:#1d9e75">1.37</span> |
| 고양이 | <span style="color:#d85a30">-0.05</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#1d9e75">0.72</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">0.25</span> | <span style="color:#1d9e75">2.55</span> | <span style="color:#d85a30">-0.06</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#1d9e75">0.60</span> | <span style="color:#d85a30">-0.09</span> | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-0.15</span> | <span style="color:#d85a30">-0.01</span> |
| 를 | <span style="color:#1d9e75">2.32</span> | <span style="color:#1d9e75">0.47</span> | <span style="color:#d85a30">-0.02</span> | <span style="color:#d85a30">-0.01</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-0.05</span> | <span style="color:#1d9e75">1.77</span> | <span style="color:#1d9e75">0.34</span> | <span style="color:#d85a30">-0.03</span> | <span style="color:#1d9e75">0.78</span> | <span style="color:#1d9e75">0.19</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#1d9e75">0.73</span> | <span style="color:#d85a30">-0.12</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">3.10</span> |
| 좋아 | <span style="color:#d85a30">-0.17</span> | <span style="color:#d85a30">-0.02</span> | <span style="color:#1d9e75">1.55</span> | <span style="color:#1d9e75">3.39</span> | <span style="color:#d85a30">-0.14</span> | <span style="color:#1d9e75">2.66</span> | <span style="color:#888780">0.00</span> | <span style="color:#1d9e75">2.03</span> | <span style="color:#1d9e75">0.62</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.09</span> | <span style="color:#1d9e75">0.65</span> | <span style="color:#d85a30">-0.17</span> | <span style="color:#d85a30">-0.16</span> | <span style="color:#d85a30">-0.07</span> |

표 A의 「좋아」 d7 = <span style="color:#d85a30">-3.24</span> → 표 B에서 <span style="color:#888780">0.00</span>으로 거의 소멸 — GELU가 강한 음수를 "차단"하는 걸 그대로 볼 수 있습니다.<br>
반대로 d4 = <span style="color:#1d9e75">3.40</span> → <span style="color:#1d9e75">3.39</span>로 거의 그대로 "통과"했습니다.

📌 예측 대상인 「좋아」 위치만 계속 추적하겠습니다.<br>
N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd> → GELU 통과 결과(표 B의 「좋아」 행):<br>
`-0.17 -0.02 1.55 3.39 -0.14 2.66 0.00 2.03 0.62 -0.17 0.00 -0.09 0.65 -0.17 -0.16 -0.07`


#### 4) 축소

W₂ 행렬곱 (16×4). "16개 신호를 가중합해서 다시 4칸으로 요약"하는 단계입니다.<br>
결과: <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd>


#### 5) FFN 출력

이 단계까지가 블록의 마지막입니다.<br>
new v <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd> + FFN출력 <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd><br>
= <kbd>7.86</kbd> <kbd>0.78</kbd> <kbd>-3.57</kbd> <kbd>6.56</kbd>

이런 블록을 여러 개 적용한 것이 다중 레이어 트랜스포머 아키텍처이며, GPT-2는 12층을 사용합니다.<br>
`x → 블록1 → h₁ → 블록2 → h₂ → … → 블록N → h_N → (출구 행렬곱 → logit → softmax → loss)` 순서로 작동합니다.

이것이 문맥을 파악할 수 있는 LLM을 만드는 원리이며, 이제 모델이 다음 단어를 찍고 그 값을 확률로 바꾸는 작업을 해야 합니다.

---

### 📌 logit: 출구 행렬곱

logit = 각 단어가 "다음 토큰"일 원점수(raw score). h와 각 단어 임베딩을 내적한 값입니다.

| token | logit | 비고 |
|---|---:|---|
| 좋아 | 10.36 | ← 제일 큼 (모델의 1등 추측) |
| 고양이 | 9.43 | |
| 나는 | 5.26 | |
| 를 | -5.06 | |
| 합니다 | -7.49 | ← 제일 작음 (근데 이게 정답) |

높을수록 모델이 그 단어일 거라고 믿지만, 핵심인 「합니다」가 정답이었습니다. (가중치가 랜덤이라 당연한 결과)<br>
이를 위해 softmax → 확률을 구해야 loss를 구할 수 있습니다.

softmax = logit(아무 범위) → 전부 양수 + 합=1.00인 확률로 변환

| token | logit | gap(=logit-10.36) | e^gap | 확률 |
|---|---:|---:|---:|---:|
| 좋아 | 10.36 | 0.00 | 1.0000 | 71.4% |
| 고양이 | 9.43 | -0.93 | 0.3946 | 28.2% |
| 나는 | 5.26 | -5.10 | 0.0061 | 0.4% |
| 를 | -5.06 | -15.42 | 0.0000002 | ≈0% |
| 합니다 | -7.49 | -17.85 | 0.0000000177 | ≈0% |

합 = 1.4007

여기까지가 순전파이며, loss는 `−ln(합니다 확률) = −ln(0.0000000126) ≈ 18.2`입니다. 상당히 높은 숫자인데, 이제 이를 역전파로 가중치를 개선해나가야 합니다.

실제 구현 스택은 아래 [§3 모델 아키텍처 설계](#3-모델-아키텍처-설계)를 참고하세요.

---

## 3. 모델 아키텍처 설계

### 3.1 기본 구조 — Decoder-only Transformer

처음엔 GPT-2 그대로 가도 되나 싶었습니다.
그런데 요즘 레시피(LLaMA 쪽)를 보면 바꿀 게 꽤 있습니다.
용어가 낯설면 [GLOSSARY](GLOSSARY.md#모델-구조-용어)를 보면 됩니다.

```text
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

| 구성요소 | GPT-2 (2019) | 본 설계 | 이유 |
|---|---|---|---|
| 정규화 | LayerNorm | **RMSNorm** | 더 단순, 학습 안정 |
| 위치 정보 | 학습형 절대 위치 | **RoPE** | 길이 확장이 비교적 용이 |
| 활성함수 | GELU | **SwiGLU** | 성능 향상 |
| 어텐션 | MHA | **GQA** (Grouped-Query) | KV 캐시 메모리 절감 |
| 바이어스 | 있음 | **제거** | 파라미터 절약, 안정 |

한 줄로 말하면: **학습이 덜 깨지고, 추론 메모리를 아끼고, 길이 확장이 그나마 수월해서** 이렇게 골랐습니다.

### 3.2 모델 크기 (GPU 예산별)

| 프리셋 | 파라미터 | layers | d_model | heads (Q/KV) | ctx | 필요 VRAM(학습) |
|---|---|---|---|---|---|---|
| **nano** (튜토리얼) | ~30M | 6 | 384 | 6 / 2 | 1024 | ~4GB |
| **small** (GPT-2급) | ~124M | 12 | 768 | 12 / 4 | 2048 | ~12GB |
| **base** (GPT-2 이상) | ~350M | 24 | 1024 | 16 / 4 | 4096 | ~24GB (또는 A100 클라우드) |

- vocab: **32,000** (BPE) — 한국어 포함 시 48k~64k 권장
- FFN hidden: `d_model × 8/3` (SwiGLU 기준, 예: 768 → 2048)

실제 구현([llm/](llm/))은 **nano** 로 학습·추론·피드백 루프까지 돌려 봤습니다.
small·base 로 키우는 이야기는 [README §2](README.md#2-파라미터는-천천히-키운다) 를 그대로 따릅니다.

### 3.3 핵심 코드 스켈레톤 (PyTorch)

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

## 4. 토크나이저

- **BPE (Byte-level)** — HuggingFace `tokenizers`
- 특수 토큰 (사고 모드 때문에 중요):

```text
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

## 5. 데이터셋 전략

아래 전략에 맞춰 한국어·영어·일본어 자연어 + 11개 언어 코드 데이터를
인증 없는 공개 소스에서 받아 [llm/datasets/](llm/datasets/) 에 정리해 두었습니다.
약 11GB, 사전학습 텍스트 + SFT/THINKING 32만여 행.
목록·출처는 [llm/datasets/README.md](llm/datasets/README.md).

### 5.1 사전학습 (Pretraining)

| 프리셋 | 데이터셋 | 토큰 수 | 비고 |
|---|---|---|---|
| nano | **TinyStories** | ~500M | 하루 안에 완주, 문법적 생성 확인용 |
| small | **FineWeb-Edu (10B 샘플)** | 5~10B | GPT-2 재현급 |
| base | FineWeb-Edu + **The Stack (smol)** | 10~30B | **코딩**은 코드 비율 15~25%가 핵심 |
| 한국어 | + AI Hub / 모두의말뭉치 / ko-wikipedia | +α | 한국어가 필요할 때 |

Chinchilla 요지([README §2](README.md#2-파라미터는-천천히-키운다)): 최적 토큰 수 ≈ 파라미터 × 20  
(124M 모델이면 최소 2.5B 토큰).

### 5.2 SFT (대화 형식)

- **OpenHermes-2.5**, **UltraChat**, **KoAlpaca** 같은 공개 instruction 데이터
- 형식:

```text
<|system|>You are a helpful assistant.<|eos|>
<|user|>파이썬으로 퀵소트 구현해줘<|eos|>
<|assistant|>...코드...<|eos|>
```

### 5.3 사고 모드 데이터

- **OpenThoughts / OpenR1-Math / Raiden-DeepSeek-R1** 같은 reasoning 트레이스를 `<THINKING>` 형식으로 바꿈
- 형식:

```text
<|user|>3자리 소수 중 가장 큰 것은?<|eos|>
<|assistant|><THINKING>
3자리 최대 수는 999. 999=3×333 합성수. 998 짝수. 997을 확인:
√997≈31.6, 2~31 소수로 나눠보면... 나누어떨어지지 않음 → 소수.
</THINKING>
가장 큰 3자리 소수는 **997**입니다.<|eos|>
```

- **손실 마스킹**: user 토큰은 `-100` (손실 제외).  
  `<THINKING>` 포함 assistant 전체에 손실을 줘서, 사고 과정도 스스로 쓰게 합니다.

---

## 6. 학습 파이프라인

### 6.1 하이퍼파라미터 (small 기준)

| 항목 | 값 |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| LR 스케줄 | Warmup 2k step → Cosine decay |
| Peak LR | 6e-4 (pretrain) / 2e-5 (SFT) |
| Batch | 유효 배치 ~0.5M 토큰 (grad accumulation) |
| 정밀도 | bfloat16 (mixed precision) |
| Grad clip | 1.0 |
| 기타 | `torch.compile`, Flash Attention (SDPA) |

### 6.2 학습 루프 스켈레톤

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

### 6.3 단계별 순서

```text
[1] Pretraining  : 원시 텍스트+코드, next-token prediction     (수일~수주)
[2] SFT          : 대화 형식, user 마스킹                        (수시간)
[3] Thinking SFT : <THINKING> 데이터 혼합 (일반:사고 = 3:7)      (수시간)
    ※ 2·3을 합쳐 한 번에 해도 됨. 사고 모드 on/off는
      system 프롬프트("think step by step in <THINKING>")로 제어 가능
[4] (선택) DPO/GRPO : 정답 검증 가능한 수학·코딩 문제로 강화학습 → 사고 품질 향상 (POST-TRAINING.md)
```

---

## 7. 추론 엔진

### 7.1 생성 루프 (KV 캐시 + 사고 모드)

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
    # 사고 구간 분리 → UI에서 접기/숨기기
    thought, _, answer = text.partition("</THINKING>")
    return {"thinking": thought.strip(), "answer": answer.strip()}
```

### 7.2 사고 모드가 돌아가는 방식

1. **학습**: `<THINKING>...</THINKING>` 이 있는 assistant 응답으로 SFT → “먼저 사고, 그다음 답” 패턴을 익힘  
2. **추론**: assistant 직후 `<THINKING>` 을 프리필해서 사고를 강제 (off 면 프리필 생략 + system 변경)  
3. **후처리**: `</THINKING>` 기준으로 사고/답 분리. 멀티턴 히스토리에는 **답만** 저장 (컨텍스트 절약)  
4. **안전장치**: 사고 토큰 상한(예: 1024) 넘으면 `</THINKING>` 강제 삽입  

---

## 8. 평가

| 능력 | 벤치마크 | 도구 |
|---|---|---|
| 언어 이해 | HellaSwag, ARC, MMLU(일부) | `lm-evaluation-harness` |
| 코딩 | **HumanEval, MBPP** | pass@1 채점 스크립트 |
| 수학·사고 | GSM8K | 사고 모드 on/off 비교 |
| 기본 | validation loss / perplexity | 자체 |

---

## 9. 현실적 기대치와 확장

- **nano/small 자체 사전학습**: GPT-2 원본 수준 유창함 + 간단한 지시까지는 됩니다.  
  본격 코딩 능력은 기대하지 않는 게 맞습니다.
- **코딩을 실제로 올리려면**: ① 코드 데이터 비율을 늘리고(The Stack), ② 350M~1B+ 로 키우고,  
  ③ 사고 모드 + 코딩 SFT(예: Magicoder-OSS-Instruct)를 더합니다.
- **지름길**: 공개 소형 베이스(Qwen2.5-0.5B/1.5B 등)를 받아  
  [POST-TRAINING](POST-TRAINING.md) 의 SFT→Thinking→추론만 직접 해도,  
  같은 지식을 배우면서 GPT-2보다 훨씬 쓸 만한 품질을 얻을 수 있습니다.
- 참고: `karpathy/nanoGPT`, `karpathy/build-nanogpt`, HuggingFace `smol-course`.

---

[GLOSSARY](GLOSSARY.md) · [README](README.md) · [POST-TRAINING](POST-TRAINING.md) · [BENCHMARK](BENCHMARK-v1.md)
