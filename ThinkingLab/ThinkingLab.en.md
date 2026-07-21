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

## References

| Source | Link |
|:-------|:-----|
| Wei et al., Emergent Abilities of Large Language Models (2022) | https://arxiv.org/abs/2206.07682 |
| Brown et al., Language Models are Few-Shot Learners (2020, GPT-3) | https://arxiv.org/abs/2005.14165 |
| Jiang et al., Mixtral of Experts (2024) | https://arxiv.org/abs/2401.04088 |
| DeepSeek-AI, DeepSeek-V3 Technical Report (2024) | https://arxiv.org/abs/2412.19437 |

---

[← README](../README.en.md)
