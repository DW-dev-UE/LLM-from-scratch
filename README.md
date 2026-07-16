> **이 저장소는 계속 다듬는 중입니다.**  
> 학습 결과, 벤치마크, 문서가 버전마다 바뀔 수 있습니다. 최신 커밋을 기준으로 봐 주세요. Issue / PR 환영합니다.

> 시작하기에 앞서, 이 글은 가독성이 좋지 않을 수 있습니다. 
> AI를 사용하여 "가독성이 좋게 다듬어줘" 라는 말을 할 수 있지만, 나중에 제가 어떤 생각을 했고 어떤 과정을 거쳤는지 없어질 수 있기때문입니다.
> 물론 AI를 통해 배우는것은 좋지만 문장을 몇 번씩 다듬으면서 나의 지식을 정리하는게 더 중요합니다.

---

좋아요, 그럼 시작하겠습니다.

## 이 프로젝트가 하는 일

NVIDIA CUDA 위에서 돌아가는 **범용 LLM**과, 코딩·문서 작업처럼 실무에 쓸 수 있는 AI를 **밑바닥부터** 만드는 실험입니다.

- 토크나이저 학습
- 모델 아키텍처 (Decoder-only Transformer)
- 사전학습 · SFT · DPO · GRPO
- KV 캐시 추론 · 사고 모드 (THINKING)
- 사람 피드백 수집 UI · 승격 게이트

오픈소스 LLM을 받아 파인튜닝하는 편이 훨씬 빠르지만, 그럼에도 처음부터 다시 짠 이유는 하나뿐입니다.

LLM 내부를 손으로 짜 보지 않으면, 결국 남의 코드를 빌려 쓰는 쪽에 머문다.

> LLM 내부를 손으로 짜 보지 않으면, 결국 남의 코드를 빌려 쓰는 쪽에 머문다.

그래서 이 저장소는 **외부 상용/오픈 LLM 가중치에 기대지 않습니다.**

---

## 문서 안내

| 문서 | 무엇을 보나요 |
|:-----|:--------------|
| **README** (지금 문서) | 설계 철학, 파라미터 전략, 구조, 실행 |
| [GLOSSARY](GLOSSARY.md) | Transformer, RoPE, DPO 같은 용어 |
| [ARCHITECTURE](ARCHITECTURE.md) | 모델 · 토크나이저 · 학습 · 추론 |
| [POST-TRAINING](POST-TRAINING.md) | 배포 후 인간 피드백 루프 |
| [BENCHMARK v1](BENCHMARK-v1.md) | Base 모델 벤치마크 (학습 과정 · Q&A 포함) |

[한국어](README.md) · [English](README.en.md) · [日本語](README.ja.md)

---

## 1. 아키텍처

### 트랜스포머 아키텍처

> LLM은 어떻게 학습되는지에 관한 서술입니다.

LLM 학습을 위해 데이터셋을 준비해야 합니다. 데이터셋은 '코퍼스'라고도 불리는데, 이번 예제의 코퍼스는 "나는 고양이를 좋아합니다"로 예를 듭니다.

---

#### 1) 토크나이저

코퍼스를 컴퓨터의 언어로 만드는 도구를 **토크나이저**라 부릅니다.

"나는 고양이를 좋아합니다" → <kbd>나는</kbd> <kbd>고양이</kbd> <kbd>를</kbd> <kbd>좋아</kbd> <kbd>합니다</kbd>

이렇게 나누면 **vocab**이라는 사전이 만들어집니다.

```text
vocab = {"나는": 1, "고양이": 2, "를": 3, "좋아": 4, "합니다": 5}
```

token ids: `[1, 2, 3, 4, 5]`

---

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

---

#### 3) 문제 만들기

토큰열을 한 칸씩 밀어서 입력/정답 쌍을 만듭니다.

입력: `[1, 2, 3, 4]` = "나는 고양이를 좋아"<br>
정답: `[2, 3, 4, 5]` = "고양이를 좋아 합니다"

<kbd>나는</kbd> → ? (정답: 고양이)<br>
<kbd>나는</kbd> <kbd>고양이</kbd> → ? (정답: 를)

---

#### 4) 정규화

x = 임베딩 값. 문제 [나는, 고양이, 를, 좋아]라고 가정하면 x는 "좋아"의 임베딩입니다.

x = <kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd>

정규화 후 N:

N = <kbd>1.10</kbd> <kbd>-0.28</kbd> <kbd>-1.50</kbd> <kbd>0.68</kbd>

---

#### 5) 어텐션

문맥을 섞는 단계 — 여기서부터가 트랜스포머 아키텍처의 핵심입니다.<br>
기존 bigram 아키텍처는 문맥을 못 봅니다. 같은 "좋아"라도 앞이 "고양이를"인지 다른 단어인지 구분하지 못합니다.

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

---

#### 6) 잔차 연결

x와 어텐션 출력을 더해 new v를 만듭니다. 원본 의미를 보존하면서 문맥 정보를 얹는 방식입니다.

<kbd>0.71</kbd> <kbd>0.18</kbd> <kbd>-0.29</kbd> <kbd>0.55</kbd> + <kbd>-0.36</kbd> <kbd>0.18</kbd> <kbd>0.75</kbd> <kbd>0.09</kbd><br>
= <kbd>0.35</kbd> <kbd>0.36</kbd> <kbd>0.46</kbd> <kbd>0.64</kbd>

아직은 숫자에 불과하지만, 문맥을 파악하고 학습할 준비가 되었습니다.

---

### 블록 (Block)

여기서부터가 블록 아키텍처입니다.

#### ① 정규화

new v를 정규화합니다.

N₂ = <kbd>-0.88</kbd> <kbd>-0.79</kbd> <kbd>0.06</kbd> <kbd>1.61</kbd>

---

#### ② FFN 1단계 — W₁ 확장

N₂를 특정 배율로 확장합니다. GPT-2는 4배율로 확장하므로 이번 예제도 동일하게 진행합니다.<br>
어텐션 출력은 V들의 가중합, 즉 섞어서 평균 내는 역할일 뿐이지만 펼쳐진 tensor는 "특정 패턴 감지기"처럼 작동합니다.

확장: `U = N·W₁ + b₁` (4×4 행렬 · 4×16 행렬 = 4×16, b₁은 편향 16칸, 초기값 0이라 생략)

> **범례**: <span style="color:#1d9e75">■</span> 통과 예정(양수) · <span style="color:#d85a30">■</span> 차단 예정(음수) · <span style="color:#888780">■</span> 0에 가까움

**표 A — GELU 적용 전 (U)**

| token | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 | d16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 나는 | <span style="color:#1d9e75">1.48</span> | <span style="color:#d85a30">-1.53</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#1d9e75">1.96</span> | <span style="color:#1d9e75">2.26</span> | <span style="color:#d85a30">-1.30</span> | <span style="color:#d85a30">-2.58</span> | <span style="color:#1d9e75">3.07</span> | <span style="color:#d85a30">-0.29</span> | <span style="color:#d85a30">-1.37</span> | <span style="color:#d85a30">-0.28</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-0.26</span> | <span style="color:#d85a30">-0.65</span> | <span style="color:#1d9e75">2.15</span> | <span style="color:#1d9e75">1.48</span> |
| 고양이 | <span style="color:#d85a30">-1.96</span> | <span style="color:#d85a30">-0.90</span> | <span style="color:#1d9e75">1.93</span> | <span style="color:#1d9e75">3.69</span> | <span style="color:#d85a30">-1.33</span> | <span style="color:#1d9e75">0.89</span> | <span style="color:#d85a30">-2.79</span> | <span style="color:#1d9e75">0.39</span> | <span style="color:#1d9e75">2.57</span> | <span style="color:#d85a30">-1.80</span> | <span style="color:#d85a30">-0.58</span> | <span style="color:#1d9e75">0.77</span> | <span style="color:#d85a30">-1.60</span> | <span style="color:#1d9e75">1.57</span> | <span style="color:#d85a30">-1.06</span> | <span style="color:#d85a30">-2.67</span> |
| 를 | <span style="color:#1d9e75">2.35</span> | <span style="color:#1d9e75">0.64</span> | <span style="color:#d85a30">-2.43</span> | <span style="color:#d85a30">-2.82</span> | <span style="color:#1d9e75">2.28</span> | <span style="color:#d85a30">-1.91</span> | <span style="color:#1d9e75">1.83</span> | <span style="color:#1d9e75">0.50</span> | <span style="color:#d85a30">-2.27</span> | <span style="color:#1d9e75">0.94</span> | <span style="color:#1d9e75">0.31</span> | <span style="color:#d85a30">-0.76</span> | <span style="color:#1d9e75">0.90</span> | <span style="color:#d85a30">-1.31</span> | <span style="color:#1d9e75">1.88</span> | <span style="color:#1d9e75">3.10</span> |
| 좋아 | <span style="color:#d85a30">-0.88</span> | <span style="color:#d85a30">-2.45</span> | <span style="color:#1d9e75">1.63</span> | <span style="color:#1d9e75">3.40</span> | <span style="color:#d85a30">-1.20</span> | <span style="color:#1d9e75">2.67</span> | <span style="color:#d85a30">-3.24</span> | <span style="color:#1d9e75">2.07</span> | <span style="color:#1d9e75">0.79</span> | <span style="color:#d85a30">-0.62</span> | <span style="color:#888780">0.00</span> | <span style="color:#d85a30">-0.21</span> | <span style="color:#1d9e75">0.82</span> | <span style="color:#d85a30">-0.63</span> | <span style="color:#d85a30">-0.57</span> | <span style="color:#d85a30">-1.77</span> |

---

#### ③ FFN 2단계 — GELU

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

---

#### ④ 축소

W₂ 행렬곱 (16×4). "16개 신호를 가중합해서 다시 4칸으로 요약"하는 단계입니다.<br>
결과: <kbd>7.51</kbd> <kbd>0.42</kbd> <kbd>-4.03</kbd> <kbd>5.92</kbd>

---

#### ⑤ FFN 출력

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

---

여기까지가 순전파이며, loss는 `−ln(합니다 확률) = −ln(0.0000000126) ≈ 18.2`입니다. 상당히 높은 숫자인데, 이제 이를 역전파로 가중치를 개선해나가야 합니다.

### 처음에 생각했던 길

처음 계획은 단순했습니다.

| 단계 | 내용 |
|:----:|:-----|
| **1** | 자연어 데이터만으로 LLM을 먼저 만든다 |
| **2** | 그 위에 코딩 특화 데이터를 얹는다 |
| **3** | 사람이 강화학습으로 다듬는다 |

코딩 AI라도 질문은 자연어로 오니까,  
**“말을 알아듣는 능력”을 먼저 완성해야 한다**고 봤기 때문입니다.

### 논문이 알려 준 것

[Llama 3](https://ar5iv.labs.arxiv.org/html/2407.21783) 를 보면 그 순서가 없습니다.  
사전학습 단계부터 이미 한 믹스에 섞습니다.

| 비중 (대략) | 데이터 |
|:-----------:|:-------|
| 50% | 자연어 |
| 25% | 수학 · 추론 |
| 17% | 코드 |
| 8% | 다국어 |

후속학습에서 instruction, preference, coding, tool-use 를 강화할 뿐입니다.  
**“자연어를 먼저 끝낸 뒤 코드를 얹는다” 단계는 없습니다.**

[DeepSeek-Coder](https://arxiv.org/html/2401.14196v1) 도 같은 방향입니다.  
코드 87% + 코드 관련 영어 10% + 기타 3% 로 처음부터 학습했고,  
자연어가 13%뿐인데도 당시 오픈 코드 LLM 중 상위권이었습니다.

### 그래서 바꾼 파이프라인

| | 순서 |
|:--|:-----|
| 예전에 생각했던 것 | 자연어만 → 코드만 → instruction |
| **지금 쓰는 것** | **혼합 pretrain** → 코드 비중 강화 → instruction → preference |

```text
  자연어 + 코드 + 수학 + 문서
            │
            ▼
   base pretraining
            │
            ▼
  code-heavy continued pretrain
            │
            ▼
    instruction tuning (SFT)
            │
            ▼
   preference (DPO / GRPO …)
```

### 데이터 믹스 가이드 (300M ~ 1B)

| 종류 | 비율 | 예시 |
|:-----|:----:|:-----|
| Raw source code | 35–45% | 소스 파일 |
| Code-adjacent | 15–20% | README, issue, PR, commit |
| 기술 자연어 | 20–30% | 영어 · 한국어 기술 문서 |
| Math / reasoning | 5–10% | 수학 · 논리 |
| Logs / tests / diffs | 5–10% | 로그, 테스트, diff |
| 한국어 · bilingual | 5–10% | 지시 · 이중 언어 |

## 2. 파라미터는 천천히 키운다

파라미터는 모델 크기입니다. (용어는 [GLOSSARY](GLOSSARY.md))  
클수록 여지는 커지지만, 이 프로젝트는 **300M → 1B → 3B** 처럼 단계를 나눕니다.

[Chinchilla](https://arxiv.org/abs/2203.15556) 요지:

> 같은 compute 예산이면 **모델 크기와 토큰 수를 같이** 키워야 한다.

당시 대형 모델은 크기에 비해 데이터가 부족한 경우가 많았고,  
더 작은 Chinchilla(70B, 1.4T 토큰)가 더 큰 Gopher보다 나은 점수를 내기도 했습니다.

그래서 원칙은 이렇습니다.

1. 작은 모델에서 **tokenizer · dataloader · loss · eval · checkpoint** 가 도는지 확인  
2. 그다음 규모를 올린다  

[StarCoder2](https://arxiv.org/abs/2402.19173) 도 3B / 7B / 15B 를 각각 다른 토큰 예산으로 나눠 학습한 사례입니다.

### 이 프로젝트가 밟는 단계

| 규모 | 목적 |
|:-----|:-----|
| 10M ~ 50M | 학습 루프, tokenizer, loss 감소 확인 |
| 100M ~ 300M | completion, FIM, 기본 instruction |
| 1B | 작은 coding assistant 실험 |
| 3B | 사내 도구 연동 후보 |
| 7B+ | 외부 노출을 고민할 최소 규모 |

현재 벤치에 올린 최신 체크포인트는 **`sft_base_v6`** (base 약 327M) 입니다. → [§5 벤치마크 한눈에](#5-벤치마크-한눈에) · 버전별 상세 기록 [BENCHMARK v1](BENCHMARK-v1.md)

---

## 3. 모델 vs RAG · Tool

회사 지식을 전부 가중치에 넣으려 하면 감당이 안 됩니다.  
코드, 문서, 빌드 로그는 계속 바뀌기 때문입니다.

| 담당 | 맡는 것 |
|:-----|:--------|
| **모델 (가중치)** | 사고 방식, 코딩 패턴, 문맥 이해, 답변 톤, 도구 사용 **방법** |
| **RAG / Tools** | 최신 사실, 사내 문서, repo, 테스트 실행, 빌드 결과 |

> 모델 = **어떻게 생각하고 도구를 쓸지**  
> RAG / Tool = **지금 무엇이 사실인지**

(RAG · Tool 런타임 연동은 로드맵에 있고, 현재 공개 범위의 핵심은 학습 · 추론 · 정렬 파이프라인입니다.)

---

## 4. 프로젝트 구조

```text
AI/
├── README.md                 설계 철학 · 실행 안내
├── GLOSSARY.md               용어
├── ARCHITECTURE.md           모델 · 학습 · 추론
├── POST-TRAINING.md          피드백 후속학습
├── BENCHMARK-v1.md           벤치 리포트
└── llm/                      구현
    ├── model.py              RMSNorm + RoPE + SwiGLU + GQA
    ├── tokenizer.py
    ├── data.py
    ├── train.py              pretrain / sft / dpo
    ├── infer.py              채팅 · 로그
    ├── feedback.py           피드백 웹 UI
    ├── reward.py · rlhf.py   RM + GRPO / RLVR
    └── eval_gate.py          승격 게이트
```

> 대용량 가중치(`.pt`), 원본 코퍼스, 일부 소스는 저장소에 올리지 않습니다.  
> 문서로 설계와 벤치를 먼저 공개하는 형태입니다.

---

## 5. 벤치마크 한눈에

**학습 GPU**: H100, A100

`base` (~327M) 기준, 동일 14문항 × THINKING on/off.  
최신 스냅샷: **`sft_base_v6`** (`ckpt/benchmark_sft_base_v6.json`, 2026-07-15).

| 체크포인트 | 요약 |
|:-----------|:-----|
| pretrain v1 | 채팅 형식에서 사실상 0점 (지시 미학습) |
| sft v1 | 지시는 시도함. 정답률은 낮음. THINKING 답 0/14 (태그 미종료) |
| sft v2 | THINKING 종료 대부분 복구 (비어 있지 않은 답 13/14). 코딩은 여전히 약함 |
| sft v3 | 코딩(일반 채팅) 4/5 완전 통과. 영어 강세, 한국어 전 문항 0점. THINKING 코딩은 여전히 0/5 |
| sft v4 | 한국어 SFT 비중↑ (10.3%→22.5%). 벤치는 거의 횡보, 한국어 회복은 미미 |
| sft v5 | 베이스를 pretrain_v2로 교체 + greedy 디코딩. 코딩 첫 만점(5/5), 한국어 회복 시작(8/30) |
| **sft v6** | **THINKING 핸드오프 해결.** prep_sft 전처리 + 추론 예산 분리만으로 사고 평균 0.50→2.93, 사고 코딩 첫 통과(4/5) |

**sft_base_v6 · 일반 채팅 모드 (THINKING 끔)** · 평균 점수 3.93 / 5

| 영역 | 결과 |
|:-----|:-----|
| 한국어 | 평균 **2.67 / 5** (fact 문항 첫 만점) |
| 일본어 | 평균 3.00 / 5 |
| 영어 | 평균 **4.33 / 5** |
| 코딩 | 테스트 **25 / 25** (완전 통과 **5 / 5**) |
| THINKING 켬 | 평균 **2.93 / 5** · 비어 있지 않은 답 **14/14** · 코딩 완전 통과 **4/5** (+prime 부분 3/5) |

v6 하이라이트 (v5 대비):

- **THINKING 핸드오프 문제 해결**: 빈 답 4/14 → 0/14 (전부 답변 생성), THINKING 평균 0.50 → 2.93 (약 6배)
- THINKING 코딩 **0/5 → 4/5** — 프로젝트 6개 버전 만에 첫 실제 코드 출력 (`+prime` 부분 통과 3/5)
- 일반 채팅 평균도 동반 상승: **3.21 → 3.93**, 코딩 5/5 만점 유지
- 한국어 fact 최초 만점 · 일본어 수학 양쪽 모드 첫 정답 (한국어 합 8/30 → 10/30)
- 베이스·SFT 소스 믹스는 v5와 완전히 동일 — 변화는 오직 데이터 전처리(`prep_sft`)와 추론 예산 분리뿐
- 남은 문제: 한국어 산술은 여전히 붕괴, 긴 생성의 반복/퇴화, 일→영 번역 미형성

학습 과정·데이터셋·문항별 Q&A (버전별 상세 기록):

- [BENCHMARK-v1.md](BENCHMARK-v1.md) (한국어)  
- [BENCHMARK-v1.en.md](BENCHMARK-v1.en.md)  
- [BENCHMARK-v1.ja.md](BENCHMARK-v1.ja.md)  

아직 early-stage 입니다. 버전을 올릴 때마다 같은 문항으로 비교할 예정입니다.

---

## 6. 참고 문헌

| 논문 | 링크 |
|:-----|:-----|
| Llama 3 Herd of Models | https://ar5iv.labs.arxiv.org/html/2407.21783 |
| DeepSeek-Coder | https://arxiv.org/html/2401.14196v1 |
| Chinchilla (compute-optimal) | https://arxiv.org/abs/2203.15556 |
| StarCoder 2 | https://arxiv.org/abs/2402.19173 |

---

<div align="center">

**이어서 읽기**

[용어](GLOSSARY.md) · [아키텍처](ARCHITECTURE.md) · [후속학습](POST-TRAINING.md) · [벤치마크](BENCHMARK-v1.md)

</div>
