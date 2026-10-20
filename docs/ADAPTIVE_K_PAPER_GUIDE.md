# Additional Papers for Adaptive-K in AutoDSPy

## Purpose

Our project uses the AutoDSPy policy to generate several candidate pipelines for one GSM8K question, score them, and update the policy. In the static-K baseline, every training episode uses the same number of candidate pipelines. The Adaptive-K goal is to choose how many candidates to generate for each question, within a defined compute budget, while preserving useful learning and measuring accuracy and cost.

The four papers below are useful as design references. They do not provide a drop-in implementation for AutoDSPy: their models, reward assumptions, and training loops differ from our GPT-2 pipeline policy and inference-model setup.

## 1. Reinforce-Ada

**Paper:** [Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives](https://arxiv.org/abs/2510.04996)  
**Download:** [PDF](https://arxiv.org/pdf/2510.04996)

**Simple idea:** Allocate more sampling effort to prompts that are difficult to learn from, rather than using an identical sampling budget everywhere. The paper proposes both an estimation-based method and a sequential sampling method.

**Why it may help AdaGRPO:** It is a close reference for making the rollout budget depend on the learning signal observed for a question. It can guide the design of a bounded rule that decides whether to collect additional pipelines before updating.

**Important difference:** It does not mean “stop as soon as one promising answer appears.” Its allocation is based on the method's estimate of prompt difficulty and learning value. Its objective and assumptions differ from AutoDSPy's original summed group-centered objective, so copying its stopping rule could change the estimator or update.

**Use for our project:** Study the sequential sampling procedure and its assumptions. Treat it as inspiration for a candidate Adaptive-K rule, then explicitly analyze whether early stopping changes AutoDSPy's group rewards, advantages, and update.

## 2. GVM

**Paper:** [Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL](https://arxiv.org/abs/2505.02391)  
**Download:** [PDF](https://arxiv.org/pdf/2505.02391)

**Simple idea:** Give different prompts different numbers of samples. GVM uses signals such as acceptance rates and gradient norms to estimate where extra samples can reduce gradient-estimation variance under a compute budget.

**Why it may help AdaGRPO:** It helps us ask what the allocation rule should measure. Reward spread alone may not tell us whether another pipeline will improve the policy update; gradient-related signals may be more directly relevant.

**Important difference:** The paper's chain-of-thought RL setup is not AutoDSPy's discrete pipeline-generation policy. Some of its signals may be expensive or unavailable in our implementation.

**Use for our project:** Compare a simple reward-based allocation signal with a gradient-informed signal, if gradient information can be measured reliably. Keep this as a later design comparison if it makes the first prototype too complex.

## 3. VIGOR

**Paper:** [Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning](https://arxiv.org/abs/2607.22002)  
**Download:** [PDF](https://arxiv.org/pdf/2607.22002)

**Simple idea:** Start with a small number of rollouts for each example in a batch. Then spend the remaining fixed budget on examples with the highest observed group-reward variance.

**Why it may help AdaGRPO:** It gives a concrete progressive-allocation pattern: probe first, inspect the results, then allocate extra attempts where the outcomes vary. That is useful to consider when designing a maximum budget and deciding when a question gets another pipeline.

**Important difference:** VIGOR allocates across multiple examples in a batch. Our current training loop samples one GSM8K question per episode. Adapting the idea may require batching questions or maintaining an allocation across episodes. The paper's analysis also relies on verifiable reward assumptions.

**Use for our project:** Treat it as an allocation baseline or a later batch-level extension, not as a direct copy of the current per-question training loop.

## 4. DPPO

**Paper:** [Unbiased Dynamic Pruning for Efficient Group-Based Policy Optimization](https://arxiv.org/abs/2603.04135)  
**Download:** [PDF](https://arxiv.org/pdf/2603.04135)

**Simple idea:** Dynamically skip some work to save compute, and use importance-sampling corrections to account for the changed sampling distribution so the gradient estimate remains unbiased under the paper's setup.

**Why it may help AdaGRPO:** It highlights a mathematical issue we must consider: if we decide which pipelines to sample or keep based on their observed outcomes, we can change the distribution of the group used for the update. The resulting update may no longer match the fixed-K baseline.

**Important difference:** DPPO's pruning and correction method is designed for its own group-based policy-optimization setup. Its correction cannot be assumed to preserve the AutoDSPy objective without a derivation.

**Use for our project:** Use it as a warning and a guide for analysis. State the Adaptive-K sampling rule precisely, then determine whether it needs a correction or whether the changed estimator is an intentional part of the new method.

## How the four papers fit together

| Paper | Main question it helps answer | Possible role in AdaGRPO |
|---|---|---|
| Reinforce-Ada | When should a prompt receive more sampling? | Sequential allocation and stopping-rule inspiration |
| GVM | What signal should guide allocation? | Compare reward-based and gradient-informed signals |
| VIGOR | How can a fixed budget be allocated progressively? | Budget-allocation baseline or batch-level extension |
| DPPO | How does selective sampling affect the update? | Bias and estimator analysis |

## Practical reading order

1. **Reinforce-Ada** for the adaptive-sampling concept closest to our goal.
2. **VIGOR** for a concrete progressive allocation procedure.
3. **GVM** for possible allocation signals beyond reward variation.
4. **DPPO** for the effect of sampling decisions on the policy-gradient estimate.

## Project-specific caution

These papers motivate Adaptive-K; they do not establish that it will improve AutoDSPy. Our current branch follows AutoDSPy's summed group-centered objective. The static-K results can motivate a hypothesis, but they do not prove that early stopping will preserve accuracy, reduce wall-clock time, or keep the same update. Those claims need a defined algorithm and a fair evaluation against static-K baselines with matched compute budgets.

## Primary sources

- Reinforce-Ada: [arXiv abstract and paper page](https://arxiv.org/abs/2510.04996) · [PDF download](https://arxiv.org/pdf/2510.04996)
- GVM: [arXiv abstract and paper page](https://arxiv.org/abs/2505.02391) · [PDF download](https://arxiv.org/pdf/2505.02391)
- VIGOR: [arXiv abstract and paper page](https://arxiv.org/abs/2607.22002) · [PDF download](https://arxiv.org/pdf/2607.22002)
- DPPO: [arXiv abstract and paper page](https://arxiv.org/abs/2603.04135) · [PDF download](https://arxiv.org/pdf/2603.04135)

## Best two inspirations among the nine local papers

For our current one-question-per-episode loop with sequential pipeline execution, the strongest methodological pair is **AERO and Reinforce-Ada**. This is a judgment about fit to our implementation goal, not a ranking of overall paper quality. AutoDSPy remains the foundational paper defining the system and baseline objective.

### 1. AERO: probe first, then give bounded extra attempts

AERO first samples a small probe group, then assigns additional small rollout batches to low-success prompts. Its rescue phase stops on the first successful extra batch or budget exhaustion. This provides a concrete structure for generating fewer pipelines when an initial group suffices and reserving bounded extra work for unsuccessful groups. It also directly studies the zero-advantage issue motivating our project. Read Sections 4.1–4.2 and Algorithm 1. [AERO method](https://arxiv.org/html/2602.14338#S4)

Borrow the allocation structure selectively: the full method replaces empirical means with Bayesian baselines for some groups and downsamples failures. Those changes do not preserve our required empirical group-centered advantages. Its shared batch budget also needs adaptation to our per-episode budget.

### 2. Reinforce-Ada: make the stopping condition explicit

Reinforce-Ada-Seq repeatedly samples until a stated condition or a maximum budget is reached. It offers positive-count and balanced positive/negative-count exit conditions. The balanced variant is particularly relevant to obtaining contrasting rewards for a policy update. It also distinguishes the number of samples generated from the number used for the backward pass. Read Section 3.2 and Algorithm 2. [Reinforce-Ada method](https://arxiv.org/html/2510.04996#S3.SS2)

Borrow bounded sequential collection as an inspiration. The complete method intentionally targets a nonlinear pass-rate objective and includes balanced downsampling and explicit inverse-pass-rate weighting. Copying that procedure would change our update. These are rules about verified rewards, not subjective confidence that one pipeline looks promising.

### Why the other seven are secondary for this specific decision

- **AutoDSPy:** essential foundation; it is the system we are extending.
- **GRPO-LEAD:** supporting reference for difficulty-aware advantage reweighting and length-sensitive rewards, rather than a direct adaptive-K method. These changes would alter our baseline objective. [Official paper](https://aclanthology.org/2025.emnlp-main.287/)
- **VIGOR:** a strong later baseline for allocating across a batch; ranking prompts by reward variance requires multiple prompts, unlike our current episode. [Section 4](https://arxiv.org/html/2607.22002#S4)
- **GVM:** useful for gradient-informed allocation, but its estimator requires probe samples and gradient measurements, adding complexity to the first prototype. [Section 3.1, Algorithm 2](https://arxiv.org/html/2505.02391#S3.SS1)
- **DAPO:** filters entire uniform-reward groups and samples more prompts to fill a batch; it does not directly adapt the group size for one question. [Section 3.2](https://arxiv.org/html/2503.14476#S3.SS2)
- **SAGC:** controls group size using parallel-generation straggler risk; that bottleneck differs from sequential pipeline execution. [Section 4](https://arxiv.org/html/2606.02218#S4)
- **DPPO:** valuable for bias analysis, but its completion pruning follows generation, so that step cannot recover the execution cost already spent. Its prompt pruning also changes prompt selection. [Section 4.2](https://arxiv.org/html/2603.04135#S4.SS2)

**Implementation boundary:** preserve the Figure 2 summed loss and empirical group-centered advantages unless a methodology change is explicitly agreed and documented. Outcome-dependent stopping can still change the expected update even if the loss expression is unchanged. Test compute savings and held-out accuracy rather than assuming either. With binary correctness rewards, a mixed group produces nonzero centered advantages; continuous semantic rewards need a separately defined criterion.


## Added reference: GRPO-LEAD

**Paper:** [GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models](https://aclanthology.org/2025.emnlp-main.287/)  
**Download:** [PDF](https://aclanthology.org/2025.emnlp-main.287.pdf)  
**Authors:** Jixiao Zhang and Chunsheng Zuo. **Venue:** EMNLP 2025 main proceedings.  
**Local PDF:** `../Papers/GRPO-LEAD_EMNLP2025_2025.emnlp-main.287.pdf`

GRPO-LEAD studies length-regularized rewards, penalties for incorrect answers, and difficulty-aware advantage reweighting. It is useful background for studying training stability and difficulty, but it does not directly supply our adaptive pipeline-count rule. Importing those reward or advantage changes would require documenting a separate methodology change.

The local Papers collection now contains nine PDFs: AutoDSPy, AERO, DAPO, SAGC, Reinforce-Ada, GVM, VIGOR, DPPO, and GRPO-LEAD. AERO and Reinforce-Ada remain the recommended pair for initial adaptive-K sampling inspiration.
