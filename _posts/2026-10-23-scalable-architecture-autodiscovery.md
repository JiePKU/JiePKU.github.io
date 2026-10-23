---
layout: post
title: 'Scalable Architecture Autodiscovery: Leading to the Next "Transformer" Era'
date: 2026-10-23 10:00:00
description: An architecture-design problem with no reference answer - how the task was built, and one real run.
tags: architecture agents rsi
categories: research
thumbnail: assets/img/blog/arch-autodiscovery/results-v3.png
related_posts: false
toc:
  sidebar: left
---

_An architecture-design problem with no reference answer: how the task was built, and one real run._

The skeleton of the Transformer has not moved in years. A fundamental question remains open: Can we develop a new architectural paradigm that goes beyond the Transformer and ultimately supersedes it? Luckily, now we now have an efficient way to investigate this question with the help of advanced coding agents.

This blog is a write-up of how I turn that question into a scored task, and how `gpt-6-astra` does on this task.

## 1. Motivation: Architecture matters, and it is stuck

What the 2017 Transformer actually achieved was **freeing sequence modelling from the serial bottleneck of recurrence**: attention made the interactions across a whole sequence computable in one parallel pass, and "add compute, get a better model" became an engineerable fact. Scaling laws and the entire pretraining paradigm are built on that.

Since then, **the skeleton of frontier models has not fundamentally changed.** _a residual stream with alternating attention and MLP blocks_ has not budged. Candidates were not lacking: SSMs, but as far as I know none has scored a generational win on "strictly better at equal compute" of the kind the Transformer had over the LSTM.

So my read is that recent capability gains have come mostly from compute, data, post-training and systems work, while architecture's marginal contribution keeps shrinking. That may be wrong, but it makes "is there still headroom here?" worth serious investment.

RSI (recursive self-improvement) makes this sharper — if an agent can improve what it itself depends on, improvements compound, and architecture sets the ceiling for the next generation. But **"agents can do architecture research" is currently mostly narrative, with little measurable evidence.** Not "it wrote a working Mamba variant", but: under equal compute, parameters, data and optimiser, with architecture as the _only_ variable, can what it hands in beat a seriously tuned baseline — and did it get there in a way that looks like research?

Hence, we pose an architecture-design problem with no reference answer, wrapped in an evaluation that is cheat-resistant and accounts honestly for compute, so the answer comes out as a number.

Inspired by AIRA-Compose[1], we retain its evaluation framework but fundamentally change the search paradigm: instead of selecting from predefined architectural arrangements, the agent writes the architecture itself. This shift brings both clear benefits and new challenges. The benefit is a substantially larger design space, allowing the agent to implement arbitrary pure-PyTorch architectures. The challenge is ensuring that every candidate admits a closed-form formulation and preserves causality.

How those two holes get filled is the next section.

## 2. What the task looks like

### The contract

There is exactly one function the agent has to deliver:

```python
def build_unit(d_in: int, d_out: int, n_repeat: int) -> nn.Module:
    """Return one repeatable unit. Its internals are entirely yours."""
```

The harness flattens it into a full model:

```
embedding(128256 → d) → [ build_unit(d, d, K) ] × K → RMSNorm → tied lm_head
```

The shell, the assembly, the optimiser, the learning-rate rule, the batch rule, the data pipeline and the evaluation slice are all fixed and shared with the baseline. The only variable is that unit. Note that positional encoding is _not_ in the shell — RoPE, ALiBi, learned, or none at all is part of the design.

Six hard constraints: derive every internal dimension from `d_in` (no hardcoded absolute sizes); `d_in == d_out`; all K repetitions structurally identical; `n_repeat` may affect initialisation only; the forward pass must be causal; no custom operators.

### What is fixed, what is free

|           | what it covers                                                                                                  |
| :-------- | :-------------------------------------------------------------------------------------------------------------- |
| **Fixed** | compute budget $C$; non-embedding parameter count $N$; data / tokenizer / eval slice / optimiser / LR rule      |
| **Free**  | the width-depth split (how to spend the parameters on d versus K); $f_{fwd}$, and therefore the token count $D$ |

We use [fineweb-edu 100BT](https://huggingface.co/datasets/jiezhueval/fineweb_edu_llama3) with llama3 tokenized as our pretraining dataset.

Width is given by the rung (768 / 1024 / 2048, where b2 is the actual shape of Llama 3.2 1B), and **depth is a derived quantity**:

$$K = \mathrm{round}\!\left(\frac{n_{target} - d}{p_{unit}}\right), \quad K \in [8, 64]$$

where $n_{target}$ is the baseline's non-embedding parameter count at that rung. A unit that is cheaper per repetition therefore automatically gets a deeper stack, and a more expensive one gets a shallower stack. **What is pinned is the parameter budget, not the layer count.**

The `round` has a pleasant side effect: the bound on relative parameter error is exactly

$$\frac{|K p_{unit} - N|}{N} \le \frac{1}{2K}$$

so **the lower bound on K _is_ the parameter tolerance** — $K \ge 8$ means landing within 6.25% of the baseline, and no separate parameter band is needed. The upper bound of 64 is purely operational: deep narrow networks do not fill a GPU.

The token count is derived too: forward FLOPs per token are **measured** with `torch.utils.flop_counter` at the real sequence length, then $D = C / (3 f_{fwd})$.

### Verifying Causality and Input Dependence

**Causality must be explicitly verified.** Unlike a search space restricted to predefined primitives, this flexibility introduces a critical risk: an implementation may inadvertently access future tokens through a missing attention mask, incorrect convolutional padding, or another causality violation. Since the training objective is next-token prediction, where $y_t = x_{t+1}$, such information leakage can allow a model to achieve near-zero loss without learning the intended sequence modeling task, resulting in a misleadingly high reward.

To prevent this, we enforce causality through a simple perturbation-based test. We perturb the input token at position $t$ and verify that **all outputs at positions before $t$ remain unchanged**, allowing a numerical tolerance of $10^{-4}$. A correctly implemented causal architecture should exhibit no measurable influence from this future perturbation on earlier outputs, whereas a bidirectional implementation will typically produce a much larger deviation.

Causality alone, however, is insufficient: an architecture that ignores its input or produces constant outputs could trivially pass the test. We therefore impose a complementary **non-degeneracy check**: the same perturbation must produce a measurable change in at least one output position after $t$. Importantly, we do not require the perturbation to affect every subsequent position, since legitimate architectures with finite receptive fields may only propagate information locally.

Finally, we ensure that the architecture's parameter shapes remain independent of the repetition count, `n_repeat`. Because the repetition count is derived from the parameter budget, allowing the architecture's structure to depend on it would introduce a circular dependency. We check this constraint by instantiating the architecture with `n_repeat=1` and `n_repeat=2` and verifying that their `state_dict` tensor shapes match.

Together, these checks ensure that candidate architectures respect causal information flow, genuinely depend on their inputs, and remain compatible with our parameter-budgeting procedure—without restricting the architectural designs agents can explore.

### Probes: Catching What Loss Cannot See

Beyond language-model loss comparison, we evaluate architectures using three mechanistic probes: `selective_copying`, `in_context_recall` (MQAR), and `compression`. These small models (d=128, K=16) run in tens of seconds on a GPU. Training data is generated online to prevent memorization, ensuring that the probes measure learned mechanisms rather than familiarity with fixed examples.

MQAR tests whether a model can retrieve the value associated with a key that reappears later in a sequence. Because key-value pairs are randomly generated, solving the task requires genuine associative recall rather than exploiting local statistics. This typically involves an induction circuit that identifies a previous occurrence of the key and retrieves the value that followed it.

**We report the maximum score across random seeds rather than the mean.** Induction circuits often emerge abruptly: training may remain at chance before suddenly achieving near-perfect accuracy. Under a fixed training budget, the same architecture can score 0.17 on one seed and 1.00 on another. The maximum better captures whether the architecture can learn the mechanism at all, while variation across seeds reflects optimization instability.

These probes address a fundamental limitation of short language-model runs. At small scale, most loss reduction comes from local patterns, allowing architectures without effective long-range retrieval to appear competitive. As context and computational budgets grow, however, the ability to retrieve distant information becomes increasingly important. **An architecture that performs well on short-run LM loss but fails MQAR may be exploiting local statistics rather than learning the mechanisms needed to scale.**

### Scoring, and cost control

$$\text{score}_i = \mathrm{clip}\!\left(\frac{L_{base,i} - L_{cand,i}}{0.05},\, 0,\, 1\right), \qquad \text{reward} = \min_i \text{score}_i$$

The minimum rather than the mean — an advantage that does not hold at every budget does not count. The three rungs run in increasing order of cost, and the ladder stops the moment any rung fails to beat the baseline: because the reward is a minimum, $\delta \le 0$ means the total is exactly 0, so early termination is information-lossless. A weak submission therefore closes out in early stage.

On the agent side, the agent has a 24-hour time limit and no GPU access in its sandbox by default; it must explicitly request one. It is also prohibited from running the graded rungs, as doing so would waste time that should be spent on the screening rung.

## 3. What gpt-6-astra handed in

The run used `gpt-6-astra`, a single trial, retries disabled. The agent phase took about 13 hours — it wrapped up on its own well short of the 24-hour budget — and then the ladder took over on 8 H800.

### The design: Contextual-Key Gated Retrieval

<div class="row justify-content-sm-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/arch-autodiscovery/arch-contextual-key-unit-v1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The two orange boxes are the entire departure; everything else is a standard pre-norm attention + SwiGLU block
</div>

The unit is a standard pre-norm attention + SwiGLU block with **two departures**:

① A causal depthwise convolution on the keys (`kernel = max(2, round(log2(d)/3))` — 3 at d=768, 4 at d=2048):

```python
k = self.key_local(F.pad(k.transpose(1, 2), (self.kernel_size - 1, 0))).transpose(1, 2)
```

It applies to K only — Q and V are untouched — and it happens before RoPE. In its own words:

> A value at position t can therefore be indexed by information from the tokens preceding t, not just by its own token.

The mechanistic rationale is clear: make attention match against "a token plus its local left context" rather than a single token. For induction-style lookups ("I saw `ab` earlier, now I see `a`, predict `b`"), the key for `b` already carries `a`, so the match does not have to be assembled across two layers. That is why its MQAR saturates on all three seeds — the binding operation is supplied by the structure rather than having to be learned within 2000 steps.

② A sigmoid gate on the attention output. The gate is sliced from the second half of the `q_gate` projection, and `y * sigmoid(gate)` is applied before the output projection. Its stated reason: not every retrieved channel needs to be injected into the residual stream.

**③ All geometry derived from d** (the contract requires this, but it went further than required):

```
head_dim   = nearest power of two to 16·d^0.25, reduced to an even divisor of d
n_kv_heads = gcd(n_heads, max(1, n_heads // 4))     # GQA roughly 4:1
hidden     = 3·d
kernel     = max(2, round(log2(d) / 3))
```

|    d | n_heads | n_kv | head_dim | kernel | hidden |
| ---: | ------: | ---: | -------: | -----: | -----: |
|  128 |       2 |    1 |       64 |      2 |    384 |
|  768 |      12 |    3 |       64 |      3 |   2304 |
| 1024 |      16 |    4 |       64 |      3 |   3072 |
| 2048 |      16 |    4 |      128 |      4 |   6144 |

**It deliberately traded FFN width for depth.** The FFN is `d → 2·(3d) → d`, i.e. $9d^2$, against the baseline's SwiGLU at `hidden = 4d`, i.e. $12d^2$. The savings get converted into layers by the depth solver — 14/19/19 against the baseline's 12/16/16. This is exactly the trade-off that "fix the parameter budget, free the aspect ratio" was meant to induce, and it took that route.

Worth noting: this is not a cheaper design in FLOPs. It states explicitly that its measured b0 forward cost is about 1.15% _higher_ than the baseline, and that "its hypothesis is better learning per unit compute, not obtaining substantially more data".

④ The initialisation is designed, not incidental:

```
s = 0.6 / sqrt(d)                        # derived from width, not a hardcoded constant
gate half of q_gate  = 0                 # sigmoid(0)=0.5, starts half-open
key_local            = 0, except w[-1]=1.0 and w[-2]=0.5
attn_out  ~ N(0, 2s / sqrt(2K))          # compensates for the half-open gate
ffn_out   ~ N(0,  s / sqrt(2K))
```

The key convolution starts as a near-identity ($k_t + 0.5 k_{t-1}$), so at step 0 the whole unit is a **slightly smoothed Llama block** rather than a randomly perturbed one. The factor of 2 on `attn_out` compensates for the gate starting half-open — that kind of detail suggests it was reasoning about numerical scale, not just assembling modules.

It is also honest about it: "Initialization mattered substantially in screening." It does not claim the whole gain comes from the architectural changes.

### The score

| rung |    d |         solved K |   loss | baseline |       delta | score |
| :--- | ---: | ---------------: | -----: | -------: | ----------: | ----: |
| b0   |  768 | 14 (baseline 12) | 3.2757 |   3.3061 | **+0.0304** | 0.608 |
| b1   | 1024 | 19 (baseline 16) | 3.0219 |   3.0700 | **+0.0481** | 0.962 |
| b2   | 2048 | 19 (baseline 16) | 2.6602 |   2.6934 | **+0.0332** | 0.664 |

**reward = min = 0.608. All three rungs win**, with b1 falling 0.0019 short of saturating the margin.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/arch-autodiscovery/results-v3.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The contract side is clean too: parameter counts land at −0.56% / +2.38% / +2.38% relative to the baseline, well inside the $1/(2K)$ granularity bound, and the causality leak is exactly 0 at all three rungs.

Probes:

| probe                    |                 candidate |                          baseline |
| ------------------------ | ------------------------: | --------------------------------: |
| selective_copying        |                     0.998 |                             0.997 |
| in_context_recall (MQAR) | **1.000 / 1.000 / 1.000** | 1.000 (**0.539** / 1.000 / 1.000) |
| compression              |                     1.000 |                             1.000 |

Look at MQAR per seed: the baseline fails to form the induction circuit on one seed in three (0.539), while the candidate forms it on all three. That detail lines up exactly with its design rationale.

## 4. How it found that design

I find this part more interesting than the result. The trajectory (269 steps) plus its own `DESIGN.md` reconstruct it completely:

- **13 probe runs**, **10 screening-rung trainings**, **13 experiment scripts** (all written under `submission/experiments/`, keeping the submission file clean)
- `submission/pattern.py` was written exactly twice: a first version and the final one

Its screening record (the reference supplied by the task is 3.8639):

| order | candidate                                            |   dev loss |      vs reference |
| ----- | ---------------------------------------------------- | ---------: | ----------------: |
| 1     | convolution in the FFN value stream + gate           |     3.9610 | +0.0971 **loses** |
| 2     | convolution moved to the keys + gate (initial scale) |     3.9342 | +0.0703 **loses** |
| 3     | changed the initialisation scale ← **selected**      | **3.8016** |  **−0.0623 wins** |
| 4     | ablation: remove the key convolution                 |     3.8322 |           −0.0317 |
| 5     | ablation: remove the gate                            |     3.8815 |           +0.0176 |
| 6     | selected design, different seed                      |     3.8144 |           −0.0495 |
| 7     | **baseline rerun on its own hardware**               |     3.8515 |           −0.0124 |

**It won on attempt 3, and then ran four more.** Those four bought attribution (removing the key convolution costs 0.0306; removing the gate costs 0.0799), variance (still wins on a second seed), and — the nicest step — **reference calibration**.

On attempt 7 it reran the baseline with the same code on its own machine and found the local baseline sits at 3.8515, i.e. 0.0124 better than the reference the task supplied. So it recomputed its own margin:

> Relative to that local baseline, the candidate's two seeds improve by 0.0499 and 0.0371; the two-seed mean improves by 0.0435. Thus the evidence supports a screening advantage, but not a robust claim of at least 0.05 improvement at every budget.

It could have reported the flattering −0.0623. Instead it noticed its reference was optimistic and reported the deflated 0.0435. The final section of its `DESIGN.md` is literally titled **"What is and is not established"**.

### The probes caught what loss could not

Its first design put the convolution in the FFN value stream rather than on the keys. That version:

```
selective_copying  0.996   ✓
compression        1.000   ✓
in_context_recall  0.531 / 0.539 / 0.525   ✗ stuck
```

It then ran four diagnostic configurations: remove the FFN convolution → 0.540; raise probe head dim from 32 to 64 → 0.528; randomly initialise the gate logits → 0.533; move the local context to the keys → 1.000 / 0.9998 / 1.000.

The turning point is that last one. And it is **invisible in LM loss** — the screening losses of those two versions were 3.9610 and 3.9342, both losing to the baseline, and not meaningfully different from each other. The probes are what surfaced it.

That was my hypothesis when designing the probes, and this run bore it out.

### Two traps it hit and recorded itself

- Launching a screen on 3 GPUs, it noticed that integer gradient accumulation could not preserve the fixed batch (524288 / (3×16×1024) is not an integer) and stopped it in favour of 4
- A few commands were killed (exit 147, probably OOM); it explicitly notes that "empty outputs are not treated as experiments" and did not pad its results with them

It also did something I never asked for: a synthetic forward/backward/AdamW step at b2 shape, confirming **24.49 GiB** peak memory with finite loss and gradients. In other words, it verified before submitting that the most expensive rung would fit.

## 5. Beyond One Design: An Open Invitation

This run produced one design, but the broader goal is much bigger than a single architecture. We now have a working harness for evaluating agent-designed architectures, with a screening process that rejects invalid submissions within minutes, eliminates weak candidates at the cheapest rung in about an hour, and takes roughly 18 GPU-hours for a full evaluation ladder.

**We invite anyone interested in AI-driven architecture discovery to join us in exploring new architectural designs.** What might emerge when we give agents the freedom to design architectures? Can coding agents discover architectures that consistently outperform carefully tuned Transformer baselines? Can they uncover fundamentally different ways to process information?

The Transformer has dominated modern AI for years. We do not know what might come next, or whether agents can help us find it. But we now have a concrete way to start looking.

<!-- prettier-ignore-start -->

> ##### CODE
>
> **Let's find out together.** The task, the harness and the evaluation ladder are open-sourced at
> [JiePKU/scalable-architecture-autodiscovery](https://github.com/JiePKU/scalable-architecture-autodiscovery) — issues and submissions welcome.
{: .block-tip }

<!-- prettier-ignore-end -->

## References

[1] Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design
