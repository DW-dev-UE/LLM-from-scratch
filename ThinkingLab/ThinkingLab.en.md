[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-8B949E?style=flat-square)](ThinkingLab.md) [![English](https://img.shields.io/badge/English-0969DA?style=flat-square)](ThinkingLab.en.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-8B949E?style=flat-square)](ThinkingLab.ja.md)

# ThinkingLab

[← README](../README.en.md)

---

Welcome to the ThinkingLab.

This is where I build my own theories and check whether they actually hold up.
The project is to develop an LLM around the level of GPT-5.6 / Claude Fable 5.

Compared to OpenAI and Anthropic, I'm a tiny ant passing by — but I do have one advantage.

I'm allowed to waste time on dumb questions.
No normal company would tolerate that, but I genuinely enjoy rebuilding my theories through exactly this kind of dumb question.

Okay, let me give an example.

## Does more readable coding AI just mean less code?

I've been coding since elementary school, and I've spent a lot of that time researching how to shorten code.
If code is too verbose, it becomes unpleasant to even look at.

A classic idea: remove the braces.

**BEFORE**
```c
if (A >= B)
{
    printf("A >= B");
}
```

**AFTER**
```c
if (A >= B)
    printf("A >= B");
```

Simpler. Personally, I find it easier to read too.
Can we shrink it further? Let's collapse it to one line.

**BEFORE**
```c
if (A >= B)
    printf("A >= B");
```

**AFTER**
```c
if (A >= B) printf("A >= B");
```

It looks shorter, but for code review, BEFORE actually looks better.
Some people probably find the brace-included version easier to read, too.

Shortening code for readability was a good idea — but shortening code indiscriminately turns out to be the wrong idea.

Maybe everyone's own complicated personal rules belong in a usage guide, while day-to-day work should follow whatever "official style guide" the group has agreed on.
That's probably exactly why GPT, Claude, and other LLMs still emit braces.

This kind of brainstorming is, at the very least, useful to me personally.
So now let me think through how to build a smarter AI. This part gets philosophical.

> [!NOTE]
> The conclusions in this folder are brainstorming snapshots from a specific point in time. For "settled facts" — what was actually adopted, and at what scale — always defer to the [README](../README.en.md).

---

## 1. Why do this project at all — AGI

Humans still don't understand LLMs. We know how to build them, but not why intelligence emerges from them.

- Why does that computation become intelligence? We can't interpret what internal algorithm the weights actually learned.
- Why does training work this well? The theory is incomplete on why gradient descent finds a solution among countless local optima that happens to generalize, and why a model with more parameters than data points generalizes instead of just memorizing.

Scaling up can even produce capabilities nobody predicted.

> [!TIP]
> According to Wei et al., *Emergent Abilities of Large Language Models* (2022), models like GPT-3, Gopher, and Chinchilla show a pattern where performance on certain tasks sits at random-guessing level below a certain scale, then suddenly jumps above random once that scale is crossed.
> On MMLU, a broad multi-domain knowledge benchmark, performance is reported to break out of random-guessing territory somewhere around **70B parameters** (based on Chinchilla).
> ( Whether this is genuine emergent capability or an illusion caused by the choice of metric is still debated. )

Worth noting: the "it can perform a new task from just a few examples, with no training" few-shot capability isn't actually this paper's claim — that's from Brown et al., *Language Models are Few-Shot Learners* (2020), the paper that introduced GPT-3. The two papers make different claims, so I want to keep them separate in my own head.

We also don't know how to induce a "mood" in an LLM.

```text
Person: "How's the weather today?"
LLM: "It's sunny today. A great day for a 'walk in the park'."
```

The point isn't storing "the user likes walks and likes sunny weather" in user memory — it's that the model connects "walk in the park" purely from the fact that the weather is nice, and we can't fully explain how that association gets made.

**Q. "At what age did you first become conscious?"**

Most people would be caught off guard by that question. But if you think about it, the moment real consciousness appears is later than you'd expect — before that, I think we behave more like a machine.

This lines up with developmental psychology too: mirror self-recognition emerges around ~18 months, autobiographical memory around ~age 3. Consciousness doesn't look like a switch — it looks like something that turns on gradually.

What even is consciousness, to begin with?
Countless tiny "balls" bouncing around inside the brain, colliding with each other, producing something we call "thought" — and that is "consciousness," that is "me"?

Listen to how a young child talks, and you'll notice something interesting.

```text
Q. "What ice cream flavor does Alex like?"
Alex: "Alex likes strawberry ice cream."
```

Young Alex hasn't yet grasped the concept of "I." Alex is just saying it that way because everyone else calls them "Alex."

Personally, going into this project, I'm going to start from one hypothesis:

> "Past a certain threshold, consciousness emerges. But we don't yet know exactly where that threshold is, or how to observe it."

That said, it's only a hypothesis — I don't actually believe it. Let me also note the counterarguments.

- **Whether emergence itself is real**: there's a real counterargument about whether this is genuine emergence, or an illusion produced by the metric being used.
- **Doubt that the threshold is scale alone**: rather than simply scaling up parameter count or FLOPs, an architectural change is plausibly a more important trigger.

I don't yet have anything concrete on how to actually test each threshold, so I've left that out of this log. Once I have an actual testable idea, I'll come back and fill this section in.

---

## 2. Under limited resources, you shouldn't go multilingual

Training a 7B-class model from scratch on a single H100 works out to roughly **6.6 years** (this varies a lot depending on assumed token count and GPU utilization). Current LLMs run at 1T-parameter scale or beyond, so limited resources have to be spent efficiently.

That means the model pretty much has to lean on an English-heavy corpus.

This isn't actually a new conclusion — the [README](../README.en.md)'s "Lessons from building the 326.7M-scale model" section already states that "supporting multilingual data under a small-model budget is fatal up to roughly the 7B scale." What's here is just re-confirming why that conclusion is obvious, from a resource-budget angle.

## 3. Let's just build separate LLMs that are good at coding, math, etc.

I wondered whether, instead of one do-everything LLM, it'd be better to build LLMs that are each good at one domain and then merge them.
The model-management pipeline would need careful design, but it seemed worth exploring.

Okay, let's look into it.

Turns out this architecture already exists. I had imagined building fully separate complete models and picking/combining them at inference time via a router or orchestration layer — but what actually exists is **MoE (Mixture of Experts)**: multiple sub-networks living inside one large model.

It's a way of bundling FFNs that each respond to a different domain across multiple layers.

```text
MoE Layer ← Router operates here
   ├── Router (Gating Network)
   ├── Expert 1 (FFN)
   ├── Expert 2 (FFN)
   ├── Expert 3 (FFN)
   └── ...
   ↓ (sum of Top-K expert outputs)
[Output]
```

MoE has a huge total parameter count but low inference cost, and each expert naturally becomes stronger at its own thing.
The Mixtral paper also observed some degree of expert specialization emerging.

That said, if the router doesn't train properly, you get **Expert Collapse** — worth being careful about.

> [!IMPORTANT]
> MoE is not applied to Apex-1's initial training.
> The moment we adopt the first 1B scale, a new tokenizer, a new corpus, and QK-norm, that's already 4 new variables at once. If loss spikes, there'd be no way to narrow down the cause, and measuring how much MoE actually helps requires a dense baseline trained on the same data first.

## 4. Pushing compute efficiency to the limit for a higher-tier model

Compute currently runs in BF16.
But at that rate, even a 1B model takes way too long to train.

I need to find ways to optimize compute without losing model quality.

Looking into it, the weights themselves absolutely can't be touched. Gradients get very small late in training, and computing in FP8 risks rounding those small updates down to zero.
That's a serious problem — you'd speed things up only to lose the very progress you were trying to make.

But the forward/backward computation that happens every single step seems like a reasonable place to apply FP8.
Those are transient values, and FP8 training has reportedly reached near-lossless quality. INT8, on the other hand, still causes far more harm than good, so I'm ruling it out.

Worth trying. From here on, training runs in FP8. Though there are several different architectures for this too, so I need to check carefully before committing.

**Master weights = BF16 + compute = FP8 (mixed precision).**

> [!TIP]
> The DeepSeek-V3 technical report (2024) points the same direction. Most matrix multiplications run in FP8, while embedding, the output head, MoE gating, normalization, attention, and the master weights/optimizer states stay in higher precision (BF16/FP32). Reassuring, since it's the same conclusion as my own "keep the weights untouched, only compute in FP8" principle.

---

## 5. Problems and solutions during Apex-2 development

### Bottleneck 1. Compute

It goes without saying, but the more compute an LLM uses, the better the model performs.
But an individual can't afford that. Never mind DDP — an H100, or a B200 if you can spare it, is probably the limit of what can be experimented with.
In the end, there's no choice but to optimize to the extreme within limited compute.

In my case, the GH200 was cheaper than the H100 at **$2.29** per hour, so I trained up to the 4B-class parameters with it.
Up to 1B-class I could fit a batch on a single GPU, but from the larger models up, VRAM ran short and OOM occurred.

I had to agonize over many things, like whether to process compute in FP8 or reduce the precision of D_MODEL.

> [!NOTE]
> The Apex-2 model was designed as a 3.87B-class model, 16 Experts, 1.45B active parameters, ~1T tokens, and the optimization path is as follows.

### Bottleneck 2. VRAM

As parameters grow, the amount of data that must always stay on the GPU — the optimizer, FP32 parameters, FP32 grads, and so on — becomes extremely large.
The original plan, fp32 parameters + autocast + fp32 AdamW, took **69.7GB** for state memory alone. Adding activations brought it to **91GB**, so 2 micro-batches barely fit into 80GB. In the end, I used 8-bit AdamW to avoid OOM.

Adopting 8-bit AdamW for the optimizer memory cut **31GB** down to **9.6GB**, and even that was placed on Grace as Paged, thanks to the Grace architecture.
As a result, I managed to raise mb to 4.

For compute precision, I adopted GEMM=BF16, with master and grad accumulation in FP32.

> [!CAUTION]
> => Pure BF16 was rejected because of numerical collapse.

### Bottleneck 3. MFU

No matter how good the tensor cores are, if VRAM is referenced heavily, memory bandwidth leaves the tensor cores IDLE.
This is called a memory bottleneck.

The first way I reduced the memory bottleneck was cutting parameter casting + grad accumulation.
Parameters are stored in fp32, but GEMM runs in bf16. Autocast converts each weight to bf16 every time forward meets a linear, and in backward the GEMM emits a bf16 dW.

The bytes I actually measured are as follows.

- forward cast fp32 -> bf16: read 15.5GB, write 7.7GB
- backward cast bf16 dW -> fp32: read 7.7GB, write 15.5GB
- AccumulateGrad p.grad += dW: read 31.0GB, write 15.5GB

In total it used **93GB**; converted at HBM3 bandwidth of about 4TB/s that's 23ms, and the measured value was 28.4ms.
In other words, this 93GB costs the same whether it processes 8,192 tokens or 16,384 tokens. So, to cut the pointless work, I stored the attention and experts weights directly in BF16.

> [!TIP]
> The result was 46GB, nearly halved.

Beyond that, I reduced the casting bottleneck by changing FP32 to BF16 and the like in CE head dW accumulation, MoE dispatch/combine, and so on.

As a result, I managed to raise MFU by more than **9%p**.
In this state, with FP32 and loss matching to 4 digits, I got most of what could be gained.

> [!WARNING]
> Of course, Hopper has FP8 Tensor cores, but optimizing beyond this is an unsafe choice. Replacing the optimizer is the same.

### Final architecture: DiLoCo

B200 GPUs couldn't be rented, and the remaining alternatives were GH200 (96GB) and H100 (80GB).

In the end I chose the GH200: with 96GB of VRAM I could use MicroBatch to raise the batch-size up to 4, and while the H100 SXM5 was $4 per hour, the GH200 was $2.29 per hour despite being the same H100.
Either way the GH200 was advantageous, so I chose it. However, since it was the Grace architecture, appropriate code changes were necessary.

But even raising MFU to 40%+, the time to train 80B~1T tokens was hopelessly short. So I adopted the DiLoCo architecture.
The principle is that each instance keeps a copy of the model, trains H steps independently on its own data, and merges afterward. It had drawbacks — if either one stops, the other instance's steps for that round become unusable, and the corpus data has to be managed as two sets — but
since the data centers were physically very close to each other, merging usually completed within a minute. So I made two instances and could finish pretraining about **1.9x** faster.

---

## 6. Problems and solutions during Apex-3 development, and notable points

> [!NOTE]
> Apex-3 is 7.3B parameters with 3.36B active MoE, 30 layers, and a target token count of ~2T.

### Bottleneck 1. Storage

As tokens grew to the Trillion scale, even the compressed .bin corpus alone — not the raw data — came to **5TB**, which couldn't even be stored on a typical home SSD.

I rented a virtual machine on Azure to do the downloading and tokenizing, and solved it by using Google Drive's storage plan.

> [!WARNING]
> I felt that if the parameters go beyond 21B, 70B, or more, the bottleneck would inevitably become impossible for an individual to handle.

### Notable Point 1. Emergent Abilities

As described on the page above, humans still don't understand the principles of LLMs. We don't know "why" these outputs come out. The goal was to observe the related phenomena in order to study this more deeply.
For this model, the goal is to observe the following 3 phenomena.

#### Phenomenon 1. Induction head phase transition

> [!NOTE]
> Early in training, the ability to copy patterns within context suddenly appears over a short interval, and a small bend shows up in the loss curve

Since it appears within the first few billion tokens, it should be observable.

#### Phenomenon 2. Emergent abilities

> [!NOTE]
> Benchmark scores such as multi-step arithmetic shoot up at some point

> [!TIP]
> With discontinuous metrics like accuracy it looks like a sudden jump, but with continuous metrics most of it rises gradually — this counterargument is strong. Logging both metrics together lets you observe both

#### Phenomenon 3. Grokking

> [!NOTE]
> Generalization suddenly happens long after the memorization stage. Originally discovered on small algorithmic datasets

> [!TIP]
> In 2025, there was research that found domain-specific grokking in 7B-class MoE pretraining through routing patterns, and since Apex-3 is also a 7B MoE, it was worth trying.

---

## References

| Source | Link |
|:-------|:-----|
| Wei et al., Emergent Abilities of Large Language Models (2022) | https://arxiv.org/abs/2206.07682 |
| Brown et al., Language Models are Few-Shot Learners (2020, GPT-3) | https://arxiv.org/abs/2005.14165 |
| Jiang et al., Mixtral of Experts (2024) | https://arxiv.org/abs/2401.04088 |
| DeepSeek-AI, DeepSeek-V3 Technical Report (2024) | https://arxiv.org/abs/2412.19437 |

---

[← README](../README.en.md)
