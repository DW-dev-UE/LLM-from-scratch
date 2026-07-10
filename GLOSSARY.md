<div align="center">

# 용어 정리

**[한국어](GLOSSARY.md)** &nbsp;·&nbsp; [English](GLOSSARY.en.md) &nbsp;·&nbsp; [日本語](GLOSSARY.ja.md)

[← README.md로 돌아가기](README.md)

</div>

---

LLM을 처음 접하는 사람도 [ARCHITECTURE.md](ARCHITECTURE.md)와 [POST-TRAINING.md](POST-TRAINING.md)를
읽을 수 있도록, 본문에 나오는 용어를 먼저 여기 정리해 둔다. 아는 용어는 건너뛰고, 읽다가 막히는 단어가
나오면 다시 돌아와 찾아보면 된다.

## 모델 구조 용어

| 용어 | 설명 |
|---|---|
| **Transformer (트랜스포머)** | Self-Attention을 핵심 연산으로 쓰는 신경망 구조. 2017년 이후 나온 거의 모든 LLM(GPT, Llama, DeepSeek 등)이 이 위에서 돌아간다. |
| **Decoder-only** | Transformer 중에서도 "지금까지 나온 토큰을 보고 다음 토큰을 예측"하는 디코더 블록만 쌓은 구조. GPT 계열이 여기 속하고, 이 프로젝트도 이 방식을 쓴다. |
| **Parameter (파라미터)** | 학습을 통해 값이 바뀌는 숫자들의 총 개수. "300M 모델"이라고 하면 파라미터가 3억 개라는 뜻이고, 흔히 모델의 "크기"를 나타내는 지표로 쓰인다. |
| **Weight (가중치)** | 파라미터가 실제로 갖고 있는 값. 예를 들어 `nn.Linear(768, 768)` 층 하나는 768×768개의 가중치를 갖고, 학습은 이 숫자들을 조금씩 조정해 나가는 과정이다. |
| **Weight tying** | 입력 임베딩 층과 출력층(LM Head)이 같은 가중치 행렬을 공유하게 만드는 기법. 파라미터 수를 줄이고 학습을 안정시킨다. |
| **토크나이저 (Tokenizer)** | 텍스트를 모델이 처리할 수 있는 정수 토큰 ID로 바꿔주는 도구. 예를 들어 "파이썬은"이 사전(vocab)에 있다면 이를 4123처럼 정수 하나로 바꾼다. 이 결과물은 벡터가 아니라 정수이며, 정수를 실제 숫자 벡터로 바꾸는 건 바로 다음 Embedding이 하는 일이다. |
| **Embedding (임베딩)** | 토큰 하나하나를 고정된 길이의 숫자 벡터로 바꿔주는 층. 문자 그대로가 아니라 벡터 연산으로 언어를 다룰 수 있게 해준다. |
| **Vision Encoder (비전 인코더)** | 이미지를 모델이 이해할 수 있는 벡터로 바꿔주는 별도의 구조. "이미지도 결국 픽셀, 즉 숫자"라는 이유만으로 텍스트 LLM이 이미지를 이해하게 되는 것은 아니며, 이 구조가 따로 있어야 한다. 이 프로젝트는 현재 텍스트·코드 전용이라 Vision Encoder가 없다. |
| **RMSNorm** | 층 정규화(값의 크기를 일정 범위로 맞춰 학습을 안정시키는 것) 기법 중 하나. 기존 LayerNorm보다 계산이 단순하면서 성능은 비슷하다. |
| **Pre-Norm** | 정규화를 attention·FFN 연산 "전"에 적용하는 배치 방식. 층이 깊어져도 학습이 잘 무너지지 않는다. |
| **RoPE (Rotary Position Embedding)** | 각 토큰이 "몇 번째 위치인지"를 회전 변환으로 attention 계산에 녹여 넣는 위치 인코딩 방식. 위치를 고정된 학습 파라미터가 아니라 수식으로 계산하기 때문에, 학습형 절대 위치 임베딩보다 컨텍스트 길이를 늘리는 확장(NTK-aware scaling, YaRN 등)이 상대적으로 쉽다. 다만 보정 없이 학습 길이보다 훨씬 긴 문장에 그대로 쓰면 성능이 떨어지는 건 RoPE도 마찬가지다. |
| **Self-Attention** | 문장 안의 모든 토큰이 서로를 보고 "이 토큰과 저 토큰이 얼마나 관련 있는지"를 계산하는 연산. Transformer의 핵심이다. |
| **Causal masking** | 미래 토큰을 미리 보지 못하게 가리는 장치. "다음 토큰 예측"이라는 목표를 지키려면 반드시 필요하다. |
| **MHA (Multi-Head Attention)** | Self-Attention을 여러 개의 독립적인 "head"로 나눠 병렬로 계산하는 방식. 각 head가 서로 다른 종류의 관계(문법, 의미 등)를 학습할 수 있다. |
| **GQA (Grouped-Query Attention)** | Query head 수보다 Key/Value head 수를 적게 둬서, 추론할 때 저장해야 하는 KV 캐시 크기를 줄이는 attention 변형. MHA보다 메모리를 아낀다. |
| **KV 캐시 (Key-Value Cache)** | 추론 중 이미 계산한 Key/Value 값을 저장해두고 다음 토큰 생성 때 재사용하는 기법. 매 토큰마다 문장 전체를 처음부터 다시 계산하지 않아도 된다. |
| **Flash Attention / SDPA** | Self-Attention 연산을 GPU 메모리를 적게 쓰면서 빠르게 계산하도록 구현한 커널. PyTorch의 `F.scaled_dot_product_attention`이 이를 제공한다. |
| **FFN (Feed-Forward Network)** | Attention 다음에 오는, 토큰 하나하나에 독립적으로 적용되는 완전연결 신경망. |
| **SwiGLU** | FFN에서 쓰는 활성화 함수 조합. 기존 GELU보다 성능이 좋다고 알려져 최근 LLM들이 널리 채택한다. |
| **Residual connection (잔차 연결)** | 층에 들어온 입력을 출력에 그대로 한 번 더 더해주는 연결. 층을 깊게 쌓아도 정보가 사라지지 않고 학습이 안정된다. |
| **LM Head** | 모델의 마지막 hidden state를 vocab 크기의 점수(logits)로 바꿔주는 최종 선형층. |
| **Logits** | 소프트맥스를 적용하기 전, "다음 토큰 후보마다 얼마나 그럴듯한지"를 나타내는 원시 점수 벡터. |

## 학습(Training) 관련 용어

| 용어 | 설명 |
|---|---|
| **Pretraining (사전학습)** | 대량의 원시 텍스트·코드로 next-token prediction을 학습시켜, 언어 구조·지식·코딩 패턴을 몸에 익히는 첫 번째 학습 단계. |
| **Next-token prediction** | "지금까지의 토큰들을 보고 바로 다음 토큰을 맞혀라"는, LLM 학습의 가장 기본이 되는 목표. 예: "파이썬은" → "프로그래밍" 예측 → "파이썬은 프로그래밍"을 다시 입력 → "언어입니다." 예측 → 반복하면 "파이썬은 프로그래밍 언어입니다."가 완성된다. |
| **Fine-tuning (미세조정)** | 사전학습이 끝난 모델을 더 작고 목적이 뚜렷한 데이터로 한 번 더 학습시키는 것. |
| **SFT (Supervised Fine-Tuning, 지도 미세조정)** | 사람이 작성하거나 검수한 (질문, 모범 답안) 쌍으로 지도학습 방식으로 미세조정하는 것. |
| **Instruction tuning** | 사용자의 지시를 잘 따르도록 대화 형식 데이터로 미세조정하는 것. SFT의 한 형태다. |
| **Loss (손실)** | 모델의 예측이 정답과 얼마나 다른지를 수치로 나타낸 것. 학습은 이 값을 낮추는 방향으로 파라미터를 조금씩 움직이는 과정이다. |
| **Cross-entropy** | 언어모델 학습에서 가장 흔히 쓰이는 손실 함수. 정답 토큰에 낮은 확률을 준 만큼 손실이 커진다. |
| **Loss masking (손실 마스킹)** | 특정 토큰(예: 사용자 질문 부분)을 손실 계산에서 제외(`-100`으로 표시)해, 모델이 그 부분은 생성하도록 학습하지 않고 정답 부분만 학습하게 만드는 기법. |
| **Perplexity** | validation loss를 지수 변환한 지표. 모델이 다음 토큰을 얼마나 잘 예측하는지 나타내며, 낮을수록 좋다. |
| **Epoch / Step / Batch** | 전체 데이터를 한 바퀴 도는 단위(epoch), 파라미터를 한 번 업데이트하는 단위(step), 한 번에 묶어서 처리하는 데이터 묶음(batch). |
| **Gradient accumulation** | GPU 메모리 제약으로 큰 배치를 한 번에 올릴 수 없을 때, 여러 스텝에 걸쳐 gradient를 누적한 뒤 한 번에 업데이트하는 기법. |
| **Optimizer / AdamW** | 손실을 줄이는 방향으로 파라미터를 실제로 업데이트하는 알고리즘. AdamW는 지금 LLM 학습의 사실상 표준이다. |
| **Learning rate / Warmup / Cosine decay** | 파라미터를 한 번에 얼마나 움직일지 정하는 보폭(learning rate)을, 처음엔 서서히 올렸다가(warmup) 이후 코사인 곡선을 그리며 서서히 낮추는(decay) 학습률 스케줄. |
| **Gradient clipping** | gradient의 크기가 너무 커지는 것을 막아 학습이 발산하지 않게 하는 기법. |
| **Mixed precision / bfloat16** | 계산량이 큰 부분(행렬 곱, activation 등)은 16비트(bfloat16)로 계산해 속도와 메모리를 아끼고, 정밀도가 중요한 마스터 가중치·옵티마이저 상태는 32비트로 유지하는 학습 기법. 두 정밀도를 섞어 쓴다고 해서 mixed precision이라 부른다. |
| **Checkpoint (체크포인트)** | 학습 도중이나 이후 모델 가중치 상태를 저장한 파일. `ckpt/sft_nano.pt` 같은 파일이 여기 해당한다. |
| **Chinchilla scaling law / Compute-optimal** | 정해진 연산량(compute) 안에서 모델 크기와 학습 토큰 수를 어떤 비율로 늘려야 가장 효율적인지 보여주는 경험 법칙. [README.md의 §2](README.md#2-파라미터의-개수)에서 자세히 다룬다. |
| **FIM (Fill-In-the-Middle)** | 문장(코드)의 앞뒤는 주어지고 중간 부분을 채워 넣도록 학습시키는 과제. 코드 자동완성에 필수적인 능력이다. |

## 추론(Inference)·생성 관련 용어

| 용어 | 설명 |
|---|---|
| **Context window (컨텍스트 윈도우)** | 모델이 한 번에 참고할 수 있는 최대 토큰 길이. |
| **Sampling (샘플링)** | logits로 확률분포를 만든 뒤, 그 분포에서 다음 토큰을 무작위로 뽑는 과정. 매번 같은 답이 나오지 않고 다양한 답이 나오게 해준다. |
| **Temperature** | 샘플링 시 확률분포를 얼마나 평평하게(다양하게) 또는 뾰족하게(확신 있게) 만들지 조절하는 값. |
| **Top-p (nucleus sampling)** | 누적 확률이 p를 넘는 상위 후보 토큰들 중에서만 샘플링하는 기법. 말이 안 되는 희귀 토큰이 뽑히는 걸 막아준다. |
| **THINKING 모드 (Chain-of-Thought)** | 최종 답변 전에 `<THINKING>...</THINKING>` 구간에 사고 과정을 먼저 생성하게 해, 복잡한 문제의 정답률을 높이는 기법. [ARCHITECTURE.md](ARCHITECTURE.md)에서 자세히 다룬다. |
| **Prefill** | 프롬프트 전체를 한 번에 통과시켜 KV 캐시를 미리 채워 넣는 추론 단계. |

## RAG / Tool 관련 용어

| 용어 | 설명 |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | 질문과 관련된 문서를 외부 저장소에서 검색(retrieval)해 프롬프트에 함께 넣어준 뒤 답변을 생성하게 하는 기법. 모델을 재학습하지 않고도 최신 지식을 반영할 수 있다. |
| **Tool use / Function calling** | 모델이 스스로 "지금은 계산기를, 지금은 코드 실행기를 써야겠다"고 판단해 외부 도구를 호출하는 능력. |

## 후속학습(Post-Training)·정렬(Alignment) 관련 용어

| 용어 | 설명 |
|---|---|
| **RLHF (Reinforcement Learning from Human Feedback)** | 사람의 선호 피드백을 보상 신호로 삼아 강화학습으로 모델을 정렬(사람이 원하는 방향으로 맞춤)하는 방법론 전체를 가리키는 말. |
| **Preference pair / Chosen / Rejected (선호쌍)** | 같은 질문에 대한 두 답변 중 더 나은 쪽(chosen)과 그렇지 않은 쪽(rejected)으로 사람이 라벨링한 데이터 쌍. |
| **DPO (Direct Preference Optimization)** | 별도의 보상모델 없이, 선호쌍 데이터만으로 직접 모델을 정렬하는 기법. RLHF보다 구현이 단순하고 안정적이다. |
| **Reward Model, RM (보상모델)** | 답변에 점수를 매기도록 학습된 모델. 기존 LLM의 마지막 층을 스칼라(숫자 하나) 출력으로 바꿔 학습시킨다. |
| **GRPO (Group Relative Policy Optimization)** | 같은 질문에 대해 여러 답변을 한 번에 샘플링하고, 그 그룹 안에서의 상대적 우열(advantage)로 모델을 업데이트하는 강화학습 기법. PPO와 달리 별도의 value 모델이 필요 없어 더 단순하다. |
| **RLVR (Reinforcement Learning from Verifiable Rewards)** | 수학·코딩처럼 정답을 자동으로 채점할 수 있는 문제에서, 보상모델 대신 "정답 일치 여부" 같은 규칙 기반 보상을 쓰는 강화학습. |
| **KL divergence / KL penalty** | 새로 학습되는 모델이 원래(참조) 모델에서 너무 멀리 벗어나지 않도록 억제하는 페널티 항. 정렬 과정에서 모델이 이상하게 망가지는 걸 막아준다. |
| **Rejection sampling** | 여러 답변 후보 중 일정 기준(점수 등)을 통과한 것만 골라 학습 데이터로 재사용하는 기법. |
| **Reward hacking (보상 해킹)** | 모델이 실제 답변 품질을 높이지 않고 보상 점수만 높이는 꼼수를 학습해버리는 현상. |

## 평가(Evaluation) 관련 용어

| 용어 | 설명 |
|---|---|
| **Benchmark (벤치마크)** | 모델의 특정 능력을 표준화된 문제셋으로 측정하는 평가 방법. 예: HellaSwag·ARC·MMLU(언어 이해), HumanEval·MBPP(코딩), GSM8K(수학). |
| **pass@1** | 코드 생성 평가에서, 모델이 한 번 생성한 코드가 테스트를 통과하는 비율. |

<div align="center">

---

[← README.md](README.md) &nbsp;|&nbsp; [ARCHITECTURE.md →](ARCHITECTURE.md)

</div>
