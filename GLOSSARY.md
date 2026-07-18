# GLOSSARY

[![KO](https://img.shields.io/badge/KO-0969da)](GLOSSARY.md) [![EN](https://img.shields.io/badge/EN-lightgrey)](GLOSSARY.en.md) [![JA](https://img.shields.io/badge/JA-lightgrey)](GLOSSARY.ja.md)

[← README](README.md)

---

> [!TIP]
> 이 문서는 사전입니다.  
> [ARCHITECTURE](ARCHITECTURE.md)나 [POST-TRAINING](POST-TRAINING.md)을 읽다가 막히면 여기로 오면 됩니다.  
> 처음부터 다 외울 필요는 없습니다. 모르는 단어만 찾아보면 됩니다.

---

## 모델 구조

| 용어 | 설명 |
|---|---|
| **Transformer (트랜스포머)** | Self-Attention을 핵심 연산으로 쓰는 신경망 구조. 2017년 이후 나온 거의 모든 LLM(GPT, Llama, DeepSeek 등)이 이 위에서 돌아간다. |
| **Decoder-only** | "지금까지 나온 토큰을 보고 다음 토큰을 예측"하는 디코더 블록만 쌓은 구조. GPT 계열. 이 프로젝트도 이 방식. |
| **Parameter (파라미터)** | 학습으로 값이 바뀌는 숫자 개수. "300M 모델"이면 파라미터가 3억 개. 흔히 모델 "크기"로 부른다. |
| **Weight (가중치)** | 파라미터가 실제로 갖고 있는 값. 예: `nn.Linear(768, 768)` 은 768×768개 가중치. 학습은 이 숫자들을 조금씩 조정하는 과정. |
| **Weight tying** | 입력 임베딩과 출력층(LM Head)이 같은 가중치 행렬을 공유. 파라미터를 줄이고 학습을 안정시킨다. |
| **토크나이저 (Tokenizer)** | 텍스트를 정수 토큰 ID로 바꾸는 도구. "파이썬은" → 4123 같은 식. 결과는 벡터가 아니라 정수고, 벡터로 바꾸는 건 다음 Embedding이 한다. |
| **Embedding (임베딩)** | 토큰 하나를 고정 길이 숫자 벡터로 바꾸는 층. 문자 그대로가 아니라 벡터 연산으로 언어를 다루게 한다. |
| **Vision Encoder (비전 인코더)** | 이미지를 모델이 이해할 벡터로 바꾸는 별도 구조. "이미지도 픽셀=숫자"라고 해서 텍스트 LLM이 바로 이미지를 이해하는 건 아니다. 이 프로젝트는 텍스트·코드 전용이라 없음. |
| **RMSNorm** | 층 정규화 기법 중 하나. LayerNorm보다 계산이 단순하고 성능은 비슷하다. |
| **Pre-Norm** | 정규화를 attention·FFN **전**에 적용. 층이 깊어도 학습이 덜 무너진다. |
| **RoPE (Rotary Position Embedding)** | 토큰 위치를 회전 변환으로 attention에 넣는 방식. 수식으로 계산해서 길이 확장(NTK-aware, YaRN 등)이 상대적으로 쉽다. 그래도 학습 길이보다 훨씬 길게 쓰면 성능은 떨어진다. |
| **Self-Attention** | 문장 안 토큰들이 서로를 보고 관련 정도를 계산하는 연산. Transformer의 핵심. |
| **Causal masking** | 미래 토큰을 미리 못 보게 가림. next-token prediction에 필수. |
| **MHA (Multi-Head Attention)** | Self-Attention을 여러 head로 나눠 병렬 계산. head마다 다른 관계(문법, 의미 등)를 학습할 수 있다. |
| **GQA (Grouped-Query Attention)** | Query head보다 KV head를 적게 둬서 추론 시 KV 캐시를 줄임. MHA보다 메모리 절약. |
| **KV 캐시 (Key-Value Cache)** | 이미 계산한 Key/Value를 저장해 다음 토큰 때 재사용. 매 토큰마다 전체를 다시 계산하지 않는다. |
| **Flash Attention / SDPA** | Self-Attention을 GPU 메모리 적게, 빠르게 돌리는 커널. PyTorch `F.scaled_dot_product_attention`. |
| **FFN (Feed-Forward Network)** | Attention 다음에 오는, 토큰마다 독립 적용되는 완전연결 망. |
| **SwiGLU** | FFN 활성 함수 조합. GELU보다 좋다고 해서 최근 LLM이 많이 씀. |
| **Residual connection (잔차 연결)** | 층 입력을 출력에 한 번 더 더함. 깊게 쌓아도 정보가 덜 사라지고 학습이 안정. |
| **LM Head** | 마지막 hidden state를 vocab 크기 logits로 바꾸는 최종 선형층. |
| **Logits** | 소프트맥스 전, 다음 토큰 후보마다 "얼마나 그럴듯한지" 원시 점수. |

---

## 학습 (Training)

| 용어 | 설명 |
|---|---|
| **Pretraining (사전학습)** | 대량 원시 텍스트·코드로 next-token prediction. 언어 구조·지식·코딩 패턴을 몸에 익히는 첫 단계. |
| **Next-token prediction** | "지금까지 토큰을 보고 바로 다음을 맞춰라". 예: "파이썬은" → "프로그래밍" → … 반복하면 문장이 완성된다. |
| **Fine-tuning (미세조정)** | 사전학습 모델을 더 작고 목적이 뚜렷한 데이터로 한 번 더 학습. |
| **SFT (Supervised Fine-Tuning)** | (질문, 모범 답) 쌍으로 지도 학습하는 미세조정. |
| **Instruction tuning** | 지시를 잘 따르도록 대화 형식으로 미세조정. SFT의 한 형태. |
| **Loss (손실)** | 예측이 정답과 얼마나 다른지. 학습은 이 값을 낮추는 방향으로 파라미터를 움직임. |
| **Cross-entropy** | 언어모델에서 가장 흔한 손실. 정답 토큰에 낮은 확률을 줄수록 커짐. |
| **Loss masking (손실 마스킹)** | 특정 토큰(예: 사용자 질문)을 손실에서 제외(`-100`). 모델이 그 부분을 "생성"하도록 배우지 않고 답 쪽만 배우게 함. |
| **Perplexity** | validation loss를 지수로 변환. 낮을수록 다음 토큰 예측이 잘 됨. |
| **Epoch / Step / Batch** | 전체 한 바퀴(epoch), 파라미터 한 번 업데이트(step), 한 번에 묶는 데이터(batch). |
| **Gradient accumulation** | 큰 배치를 못 올릴 때, 여러 스텝 gradient를 쌓았다가 한 번에 업데이트. |
| **Optimizer / AdamW** | 손실을 줄이는 방향으로 파라미터를 실제로 갱신. AdamW가 사실상 표준. |
| **Learning rate / Warmup / Cosine decay** | 보폭(lr)을 처음엔 서서히 올리고(warmup), 이후 코사인으로 낮춤(decay). |
| **Gradient clipping** | gradient가 너무 커지지 않게 잘라 발산을 막음. |
| **Mixed precision / bfloat16** | 무거운 연산은 16비트(bfloat16), 중요한 가중치·옵티마이저 상태는 32비트. 속도와 메모리 절약. |
| **Checkpoint (체크포인트)** | 가중치 스냅샷 파일. `ckpt/sft_nano.pt` 같은 것. |
| **Chinchilla scaling law** | 정해진 compute 안에서 모델 크기와 토큰 수를 어떤 비율로 키울지. [README §2](README.md#2-파라미터는-천천히-키운다). |
| **FIM (Fill-In-the-Middle)** | 앞뒤는 주고 중간을 채우게 학습. 코드 자동완성에 중요. |

---

## 추론 · 생성

| 용어 | 설명 |
|---|---|
| **Context window (컨텍스트 윈도우)** | 한 번에 참고할 수 있는 최대 토큰 길이. |
| **Sampling (샘플링)** | logits → 확률분포 → 다음 토큰을 무작위로 뽑음. 매번 같은 답이 안 나오게 함. |
| **Temperature** | 분포를 평평하게(다양) / 뾰족하게(확신) 조절. |
| **Top-p (nucleus sampling)** | 누적 확률 p를 넘는 상위 후보만 남기고 샘플링. 이상한 희귀 토큰을 줄임. |
| **THINKING 모드 (Chain-of-Thought)** | 답 전에 `<THINKING>...</THINKING>` 에 사고 과정을 먼저 쓰게 함. [ARCHITECTURE](ARCHITECTURE.md). |
| **Prefill** | 프롬프트 전체를 한 번에 통과시켜 KV 캐시를 미리 채우는 단계. |

---

## RAG / Tool

| 용어 | 설명 |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | 관련 문서를 외부에서 검색해 프롬프트에 넣고 생성. 재학습 없이 최신 지식을 넣을 수 있음. |
| **Tool use / Function calling** | 모델이 "지금은 계산기 / 코드 실행" 같은 외부 도구를 호출하는 능력. |

---

## 후속학습 · 정렬

| 용어 | 설명 |
|---|---|
| **RLHF** | 사람 선호 피드백을 보상으로 써서 강화학습으로 모델을 맞추는 방법 전체. |
| **Preference pair / Chosen / Rejected** | 같은 질문에 대한 두 답 중 나은 쪽(chosen)과 아닌 쪽(rejected). |
| **DPO (Direct Preference Optimization)** | 보상모델 없이 선호쌍만으로 정렬. RLHF보다 구현이 단순하고 안정적. |
| **Reward Model, RM** | 답에 점수를 매기도록 학습한 모델. 마지막 층을 스칼라 출력으로 바꿈. |
| **GRPO** | 같은 질문에 여러 답을 샘플링하고, 그룹 안 상대 우열(advantage)로 업데이트. PPO와 달리 value 모델이 필요 없음. |
| **RLVR** | 수학·코딩처럼 자동 채점 가능한 문제에서, RM 대신 "정답 일치" 같은 규칙 보상. |
| **KL divergence / KL penalty** | 새 모델이 참조 모델에서 너무 멀어지지 않게 막는 항. |
| **Rejection sampling** | 여러 후보 중 기준을 통과한 것만 골라 학습 데이터로 씀. |
| **Reward hacking (보상 해킹)** | 실제 품질은 안 올리고 보상 점수만 올리는 꼼수를 배워 버리는 현상. |

---

## 평가

| 용어 | 설명 |
|---|---|
| **Benchmark (벤치마크)** | 표준 문제셋으로 능력을 측정. 예: HellaSwag·MMLU(언어), HumanEval·MBPP(코딩), GSM8K(수학). |
| **pass@1** | 코드를 한 번 생성했을 때 테스트 통과 비율. |

---

[README](README.md) · [ARCHITECTURE](ARCHITECTURE.md) · [POST-TRAINING](POST-TRAINING.md)
