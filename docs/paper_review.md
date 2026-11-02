# Paper review: adapting ten research papers to Adaptive-K AutoDSPy

Date: 2026-11-01 (Asia/Dhaka). Status: research review and implementation proposal, not an implemented method or experimental result.

## 1. Purpose and recommendation

This review studies all ten PDFs in the workspace's `Papers/` directory to identify ideas for selecting GRPO candidate-group size dynamically in AutoDSPy. The immediate question is:

> Can we allocate expensive pipeline executions more effectively across training questions, while retaining held-out performance under a comparable resource budget?

The strongest sources for allocation design are **VIGOR, Reinforce-Ada, AERO, and GVM-RAFT**. They support different allocation strategies; they do not establish one universally correct rule. **SAGC** supplies a useful controller pattern and fairness discussion, but its synchronization bottleneck differs from this project's sequential execution. **DPPO** is particularly useful for thinking about selection bias. **AutoDSPy, DeepSeekMath, DAPO, and GRPO-LEAD** establish the task, objective distinctions, and relevant diagnostics.

Recommended sequence:

1. Use an allocation-only method over the existing sizes `{5, 8, 10, 15}`, retaining the project's Figure 2 loss and reward protocol.
2. Begin with a controller that selects K **before the current group is sampled**, using past observations and a bounded exploration mechanism. This provides a simpler first experiment than reward-dependent stopping within the current group.
3. Compare that controller against a bounded, within-question rescue rule inspired by AERO/Reinforce-Ada. This explicitly tests whether investing in homogeneous prefixes is more effective than concentrating on already informative groups.
4. Treat VIGOR-style cross-question ranking, posterior-based advantages, sample retention, reward shaping, and loss normalization as separate extensions with their own controls.

These are proposed adaptations. None of the papers directly validates this combination of GPT-2 pipeline selection, a frozen Ollama executor, GSM8K, provided-context HotpotQA, and the project's unnormalized summed loss.

### Current project context

The user reports that 200-episode GSM8K and HotpotQA experiments across seeds are running and are monitored by a collaborator. Their completion or results are not assumed here. Experiment management, checkpoint selection, resource logging, and saved-run analysis already exist in `Code/AUTODSPy`; building those again is not the proposed next task.

Direct notebook inspection confirmed `generate_pipeline`, `group_advantages`, `figure2_loss`, `update_policy`, `train_grpo`, checkpoint functions, and `analyze_saved_static_k` in both independent notebooks. The current update and training functions enforce a fixed `cfg['k']`. Adaptive-K therefore requires deliberate changes to group-size validation, configuration identity, controller state, and budget accounting. This review does not modify those notebooks.

## 2. Sources, reading scope, and notation

All paper claims below refer to the **local PDF versions**, not an assertion about their latest online revisions or independent reproduction. Page references are one-based PDF page numbers, including front matter. Full PDF text was extracted for all ten documents (208 pages total); the review focuses on their methods, experimental comparisons, limitations, and relevant theoretical/implementation appendices. Selected key equation and algorithm pages were also rendered for visual checking. This is a critical methodological reading, not a line-by-line verification of every proof or cited reference.

Source inventory:

- **[P1] [AutoDSPy](../../../Papers/AutoDSPy.pdf)**, 16 pages. Main mechanism: pp. 4-6; K sensitivity and adaptive-K motivation: pp. 11-12; implementation/protocol details: pp. 14-16.
- **[P2] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](../../../Papers/DEEPSEEKMATH.pdf)**, 30 pages. GRPO: Section 4, pp. 12-15, especially Equations 3-4 and outcome supervision on p. 14.
- **[P3] [DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization](../../../Papers/DAPO.pdf)**, 16 pages. Sections 3.1-3.4 and Algorithm 1: pp. 4-7; training/evaluation: pp. 8-9.
- **[P4] [Train Less, Learn More: Adaptive Efficient Rollout Optimization for Group-Based Reinforcement Learning (AERO)](../../../Papers/Aero.pdf)**, 20 pages. Sections 4-5: pp. 3-8; theoretical details: pp. 11-13; implementation: pp. 16-17.
- **[P5] [SAGC: Straggler-Aware Group Control](../../../Papers/SAGCS.pdf)**, 13 pages. Controller: pp. 4-6; results: pp. 7-8; limitations: p. 9; budget normalization and implementation: pp. 12-13.
- **[P6] [Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives](../../../Papers/REINFORCE-ADA.pdf)**, 27 pages. Framework: pp. 2-6; Algorithms 1-2 and sequential sampling: pp. 7-9; evaluation: pp. 10-12; variance analysis: pp. 22-23.
- **[P7] [Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL (GVM-RAFT)](../../../Papers/GVM-RAFT.pdf)**, 32 pages. EM framework: pp. 4-6; allocation derivation: pp. 7-10; GRPO extension and results: pp. 11-14; practical settings: p. 19.
- **[P8] [GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models](../../../Papers/GRPO-LEAD.pdf)**, 13 pages. Reward/advantage changes: pp. 3-4; setup and ablations: pp. 5-8; limitations: p. 9.
- **[P9] [Unbiased Dynamic Pruning for Efficient Group-Based Policy Optimization (DPPO)](../../../Papers/DPPO.pdf)**, 20 pages. Hierarchical correction and pruning: pp. 3-5; packing/results: pp. 6-8; Algorithm 1: p. 12; variance analysis: pp. 15-17. DPPO here means **Dynamic Pruning Policy Optimization**.
- **[P10] [Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning (VIGOR)](../../../Papers/Vigor.pdf)**, 21 pages. Allocation: pp. 5-7; matched-budget experiments: pp. 8-10; proof assumptions: pp. 16-18. VIGOR means **VarIance Guided Online Rollout allocation**.

Notation used in this review:

- `q`: training question; `P_i`: sampled candidate pipeline; `r_i`: its recorded reward.
- `K_t`: candidates generated for episode t; `C_t`: cumulative attempted candidate evaluations.
- `p(q)`: probability of a binary successful outcome under the current policy, executor, and reward protocol.
- `s_i = grad_theta log pi_theta(P_i | q)`: pipeline policy score gradient, including its sampled actions.
- A generated candidate, an executed pipeline, an executor request, a judge request, and a retained training sample are different quantities. Pipeline length, retries, and judge caching prevent treating them as interchangeable.

## 3. Compatibility with the actual AutoDSPy objective

### 3.1 What is trained here

AutoDSPy trains a policy to select pipeline components; the execution LLM is frozen. Most of the other papers train the language model that generates reasoning-token trajectories. Their response-length regularizers and GPU batching optimizations therefore do not automatically apply to the GPT-2 pipeline policy. [P1, pp. 4-5; P2, pp. 13-15]

The original `DSPy_GRPO.ipynb` and both current Static-K notebooks implement the conceptual objective:

```text
r_bar = sum(r_i) / K
A_i = r_i - r_bar
L = -sum_i A_i * sum_j log pi_theta(action_ij | state_ij)
```

`group_advantages` and `figure2_loss` preserve this centered, summed objective. There is no group-standard-deviation division, PPO clipping, KL penalty, or averaging by K in this loss. DeepSeekMath's GRPO uses standardized rewards and a clipped policy-ratio objective with KL regularization; the name GRPO does not imply identical mathematics. [P2, pp. 13-14]

### 3.2 Variable K changes update scale as well as information

**Review derivation, not a claim copied from a paper:** for a fixed question, fixed K, iid pipeline/reward samples, a fixed policy throughout collection, and a valid score-function identity `E[s_i] = 0`,

```text
E[sum_i (r_i - r_bar) * s_i] = (K - 1) * E[r * s].
```

The own-sample contribution to the empirical baseline removes one unit of expected score contribution. Consequently, choosing larger K under the existing summed objective increases expected raw gradient scale, conditional on those assumptions. A selector that systematically gives certain questions larger K also changes their effective weighting. Adam, finite-sample variability, and policy-score approximations complicate translating this identity into parameter movement.

Retaining the Figure 2 loss is the clean compatibility choice, but it does **not** isolate variance reduction from update scaling. Report the K distribution and raw gradient norms. An eventual `1/(K-1)` scaling comparison would be a separately named objective-control experiment requiring a documented methodology change; it is not part of the initial proposal or current authorized implementation.

### 3.3 Homogeneous rewards and optimizer movement

If all recorded rewards are identical, centered advantages and the current reward-driven gradient are zero. This includes homogeneous fractional groups, not only all-zero/all-one binary groups. Nonzero reward variance is necessary for a nonzero centered reward signal but does not guarantee a nonzero total gradient: score directions can cancel.

The current notebooks still call `optimizer.step()` for such groups. Existing project auditing established that Adam momentum can move parameters after a zero current gradient. Therefore distinguish zero advantage, zero measured gradient, and zero parameter movement. Introducing a skip-update policy changes optimization behavior and should be its own ablation. See [Phase 1 audit](PHASE_1_AUDIT.md) and `update_policy` in both notebooks.

## 4. Individual paper reviews

### 4.1 P1: AutoDSPy

**Research problem and mechanism.** A learned policy constructs DSPy pipelines from modules and signatures and receives downstream task rewards. The executor and the policy have separate roles. The K ablation reports different trends for GSM8K and HotpotQA and explicitly proposes adaptive K as future work. [P1, pp. 4-6, 11-12]

**Evidence.** The paper reports GSM8K accuracy declining from 82.4% at K=5 to about 60% at K=15, while HotpotQA increases from 73.9% to 77.1%. These are paper-reported results under its setup, not expected values for this repository. Its causal explanations involving noise and task complexity are interpretations of those observations, not a general law about K. [P1, pp. 11-12]

**Adaptation.** Keep pipeline-level decisions, dataset-specific reporting, and a bounded K action space. Measure whether K depends on the current policy and question rather than hard-coding "math gets small K, QA gets large K." The current Static-K runs are directly relevant to identifying that relationship.

**Limits.** Paper wording about binary rewards, semantic fallback, and partial correctness is not a precise substitute for the implemented reward protocol. The project's existing audit also records episode/sampling and HotpotQA formulation ambiguities. The first-token signature scoring approximation constrains action diversity independently of K. Adaptive allocation cannot create distinctions that the policy representation cannot express. See [decisions](DECISIONS.md), [reward modes](REWARD_MODES.md), and [saved GSM8K analysis](STATIC_K_RESULTS_ANALYSIS.md).

**Priority: essential task foundation.** Adapt the research question and pipeline-level accounting; do not assume the published K trend will reproduce.

### 4.2 P2: DeepSeekMath

**Research problem and mechanism.** GRPO removes the learned value model and estimates relative advantages from several responses to the same question. Outcome-supervised rewards are centered and divided by the group standard deviation. The objective includes clipping and a reference-policy KL term. [P2, pp. 13-14]

**Evidence.** The reported RL recipe uses DeepSeekMath-Instruct 7B, 64 outputs per question, a maximum length of 1024, and roughly 144K RL questions. The paper reports 88.2% GSM8K and 51.7% MATH accuracy for the resulting RL model. These results combine model preparation, data, and RL; they are not evidence that changing K alone yields those scores. [P2, p. 15]

**Adaptation.** Preserve same-question grouping, freeze the behavior policy while collecting a group, and keep advantages linked to the final group membership. Use its formulation to explain precisely how this repository differs from standardized GRPO.

**Limits.** It does not supply an adaptive-K selection algorithm. Its memory advantages concern eliminating a large critic when training a large reasoning model; the project's small GPT-2 and frozen executor have different costs.

**Priority: mathematical reference.** Useful for verification and terminology, not the primary allocation design.

### 4.3 P3: DAPO

**Research problem and mechanism.** DAPO addresses entropy collapse, loss weighting, truncated responses, and diminishing informative groups. Dynamic sampling generates fixed-size groups, filters all-correct/all-incorrect groups, and accumulates informative groups to fill a training batch. Other components include asymmetric clipping, token-level loss reduction, and overlong reward shaping. [P3, pp. 4-7]

**Evidence.** The Qwen-32B experiment reports 50% AIME 2024 average accuracy with fewer training steps than its reported comparison. This is a combined training recipe; individual components and step counts cannot be translated directly into AutoDSPy executor savings. Training uses batches of 512 prompts and 16 responses per prompt. [P3, pp. 8-9]

**Adaptation.** Log the fraction of groups with identical rewards, policy entropy, and truncation failures. Distinguish learning-signal loss from executor/parsing failure. A filtering comparator can reveal whether an allocation method is doing more than selecting medium-difficulty groups, if such a comparator is later authorized.

**Limits.** DAPO dynamic sampling is primarily prompt-group filtering, not selecting variable K within a single question. It pays for candidates before filtering. Dropping a scheduled question and replacing it would change the repository's prompt schedule. Clip-higher, token-level loss, and reward shaping also conflict with preserving the initial Figure 2 protocol.

**Priority: diagnostics and comparator inspiration.** Do not copy the complete DAPO optimizer into an allocation-only pilot.

### 4.4 P4: AERO

**Research problem and mechanism.** AERO separates exploration, rescue, and training-sample curation. A small initial group estimates success; low-success questions can receive extra samples in increments until a success or budget exhaustion. Other paths retain or downsample candidates. It also replaces empirical success estimates by a Beta-Binomial posterior mean in homogeneous groups. [P4, pp. 4-5]

For c successes among n binary trials, its posterior mean is:

```text
p_tilde = (c + alpha0) / (n + alpha0 + beta0),
alpha0 = beta0 = 1 in the reported configuration.
```

**Evidence.** The main configuration uses a total generation budget of 16 per query on average and an exploration budget of 8, with about 4.6 retained training rollouts per query. It reports approximately 1.8-1.9x per-step speedups on its math settings. Crucially, its cost discussion says rollout FLOPs are identical across the main methods and attributes much of the gain to reduced model-update work. Generated and retained counts must therefore remain distinct. [P4, pp. 6-7]

**Adaptation.** Borrow bounded rescue, small increments, explicit budget exhaustion, and a separate state for groups that remain homogeneous. Use posterior uncertainty for allocation decisions if appropriate. Keep every generated candidate in the initial AutoDSPy update, avoiding retention as an extra variable.

**Limits and critical reading.** Replacing the empirical baseline by the posterior mean changes advantages; it is not merely choosing K. Nonzero individual advantages do not guarantee a nonzero aggregate gradient. A balanced positive/negative composition maximizes a scalar term in the paper's bound under its assumptions, not necessarily actual gradients or held-out accuracy. The main text and appendix use different budget details (including a per-question cap of 32 in the appendix); do not treat "16" as an unambiguous universal hard cap. [P4, pp. 5, 13, 17]

**Priority: high for rescue; separate ablations for posterior advantages and retention.** Its full reported speedup should not be used as an AutoDSPy forecast.

### 4.5 P5: SAGC

**Research problem and mechanism.** In synchronous batched generation, the longest response can stall the group. SAGC defines the straggler event from `max(length)/median(length) > tau`. A CPU controller selects group size using a utility-minus-risk score, discounted Beta estimates, Thompson sampling, and a dual penalty:

```text
score(G) = log2(G) - lambda * sampled_straggler_probability(G)
lambda_next = max(0, lambda + eta * (observed_rate - target_rate))
```

Neighbor-only moves limit abrupt changes. [P5, pp. 4-6]

**Evidence.** Experiments use Qwen2.5-3B and Llama-3.2-3B under GRPO/DAPO, with static sizes `{4,8,16}` and a single H200. The method is not faster than every static choice: for Qwen GRPO, Table 1 reports 7.93 hours for SAGC versus 7.77 for static G=8 and 4.05 for G=4, with different quality outcomes. [P5, pp. 6-8]

**Adaptation.** Borrow bounded actions, gradual adaptation, saved controller state, explicit exploration, and a cost constraint. For AutoDSPy, replace synchronization risk with measured sequential execution cost or an execution-budget constraint; call that a SAGC-inspired adaptation, not a reproduction.

**Limits.** Sequential pipelines do not experience the paper's intra-batch wait-for-the-slowest GPU barrier. Longer candidates still cost time, but the physical mechanism is different. Its fixed effective batch `B*G=64` does not fix total generations: Appendix B explicitly gives total generations `P*G` and varying optimizer steps. A fixed per-step budget is not a common total-compute budget. [P5, pp. 12-13]

**Priority: medium for controller structure; low for direct straggler claims.**

### 4.6 P6: Reinforce-Ada

**Research problem and mechanism.** Finite groups can hide rare successes or failures. Reinforce-Ada frames adaptive allocation through a nonlinear objective `E_q[f(p(q))]`, whose gradient weights questions by `f'(p)`. Its estimation variant uses historical/model-based success estimates; the practical hybrid allocates roughly proportional to `1/sqrt(p_hat + epsilon)` and adds an explicit residual advantage weight. Its sequential variants sample until enough positive, or both positive and negative, outcomes are found, subject to a cap. [P6, pp. 3-9]

The history estimator discounts old counts and combines them with a Beta prior. The sequential pipeline collects a variable pool, computes statistics over that pool, then forms a fixed-size update group through downsampling. [P6, pp. 7-9]

**Evidence.** Experiments cover Qwen2.5-Math 1.5B/7B, Qwen3-4B, and Llama-3.2-3B on mathematical reasoning. The reported sequential implementation can generate up to 32 responses but use only four in the update. The paper acknowledges only coarse comparisons for the stochastic sequential budgets. [P6, pp. 10-12]

**Adaptation.** This is a strong source for distinguishing "no observed signal" from "impossible question," capped sequential rescue, discounted question history, and separating generation from update counts. A smoothed historical estimate avoids interpreting zero observed successes as an exactly zero success probability.

**Limits and critical reading.** Its log objective, explicit weights, balancing, and downsampling are substantive changes to this project's objective. Algorithm 2 lists centered advantages, whereas the surrounding discussion additionally introduces `1/p_hat` correction; a faithful reproduction needs those details reconciled. Do not copy only the sampling loop and claim all theoretical properties follow. The familiar expected waiting time `1/p` is for an uncapped iid stop-at-first-success process, not for capped, batched sampling with a changing policy. [P6, pp. 7-9]

Appendix B derives different variance-minimizing allocations for different estimators: approximately `1/sqrt(p)` for the RFT case and `sqrt((1-p)/p)` for baselined REINFORCE under its assumptions. This reinforces that an allocation rule depends on the objective and estimator. [P6, pp. 22-23]

**Priority: very high for rescue, uncertainty, and allocation-bias reasoning.**

### 4.7 P7: GVM-RAFT

**Research problem and mechanism.** GVM-RAFT treats chain-of-thought as a latent variable and allocates inference budgets to reduce the variance of an EM/rejection-sampling gradient estimator. Its allocation combines acceptance rate p_i and a gradient second-moment quantity G_i. The regularized rule is proportional to:

```text
n_i proportional to G_i / sqrt(p_i + alpha / p_i**(beta - 1)).
```

The regularizer suppresses excessive allocation as success probability approaches zero. The practical algorithm first estimates acceptance rates and gradient quantities using probe samples. [P7, pp. 7-10]

**Evidence.** The paper tests Qwen2.5-Math 1.5B/7B and extends allocation to GRPO. Its 2-4x convergence gains concern the reported convergence curves, not a universal wall-clock speedup. It also explains that some plotted "steps" are normalized sample-budget units rather than literal optimizer steps. [P7, pp. 12-14]

**Adaptation.** Compare reward-only allocation with a low-cost proxy for pipeline-policy score-gradient information if reward variance proves insufficient. Include the probe and gradient-estimation cost in any efficiency claim. Retain regularization/caps so extremely low success estimates cannot consume the entire budget.

**Limits and critical reading.** The derivation is for a different EM estimator; the GRPO extension is experimentally demonstrated rather than a proof of this project's objective. Computing per-candidate gradients solely to select K may erase savings. There is a notation/estimator detail to resolve before implementation: the derivation defines a squared-norm second moment, while the practical sections present norm and squared-norm expressions inconsistently. Use the derivation to specify the intended statistic and verify it on a finite example rather than copying a symbol. [P7, pp. 8-11, 19]

**Priority: medium initially; high if simple reward-based controllers fail.**

### 4.8 P8: GRPO-LEAD

**Research problem and mechanism.** LEAD changes reward and advantage construction to encourage concise, correct reasoning. Correct answers receive `exp(-alpha*z)` using standardized lengths among correct responses; incorrect answers receive an explicit penalty. A logistic difficulty weight uses empirical group correctness and treats positive and negative advantages differently. [P8, pp. 3-4]

**Evidence.** The main models are distilled 7B/14B reasoning models, trained with group size eight. The 7B ablations show accuracy and length trade-offs, with the incorrect-answer penalty increasing length relative to some preceding variants. The 14B recipe also includes staged data/SFT/curriculum choices. Its performance cannot be attributed to a K selector. [P8, pp. 5-8]

**Adaptation.** Use the lesson that success rate and response cost contain different information. Record executor output length and latency beside accuracy. Study difficulty-conditioned allocation as a hypothesis, not an assumption that the hardest questions always deserve the largest K.

**Limits.** LEAD is not an adaptive-group-size method. A frozen executor cannot be trained directly to shorten its reasoning through a GPT-2 token-level penalty; any effect occurs indirectly through changed pipeline choices. Length rewards, explicit penalties, standardized advantages, and difficulty weights would confound the first allocation-only comparison.

**Review derivation:** changing binary rewards from `{0,1}` to `{-1,1}` alone gives `A'_i = 2*A_i` under the project's centered objective. It does not create contrast in a homogeneous group. A constant additive shift cancels entirely. The paper's complete reward scheme includes more than that simple affine transformation.

**Priority: supporting diagnostics and later cost-aware objectives.**

### 4.9 P9: DPPO

**Research problem and mechanism.** DPPO prunes low-absolute-advantage completions probabilistically after generation and uses historical prompt scores for prompt pruning before generation. Retention probabilities remain positive, and retained samples receive importance-based rescaling. Dense prompt packing mitigates sparse training batches. [P9, pp. 3-6]

**Evidence.** Main experiments use Qwen3-4B/8B with eight H100 GPUs and a verl/vLLM stack. The paper reports up to 2.65x speedup in its GSM8K table, with changes in model-update workload, prompt generation, and packing. These are not isolated gains from adaptive K. [P9, pp. 6-8]

**Adaptation.** The strongest lesson is to make selection probabilities explicit, retain exploration support, and test whether a claimed estimator still targets the intended quantity. Completion pruning and prompt pruning save different costs; only pre-generation decisions avoid already incurred executor calls.

**Limits and critical reading.** A density-ratio identity with a known normalizer does not by itself prove that every finite implementation is unbiased. Section 4.2 replaces normalizers with realized retention fractions. Correctness depends on the loss denominator, selection mechanism, and treatment of empty retained sets. For a finite fixed pool with known inclusion probabilities, inverse-inclusion weighting can preserve a fixed sum; recalculating group advantages after pruning changes that sum's terms. Reward-dependent sequential stopping is a different selection problem and is not automatically corrected by DPPO's formula. [P9, pp. 4-5, 12]

**Priority: high as a correctness lens; low as the first speed optimization.**

### 4.10 P10: VIGOR

**Research problem and mechanism.** VIGOR begins with a small rollout group for every question in a batch, ranks questions by observed reward variance, retains a top fraction for additional generation, and grows their incremental allocation. It accumulates all generated groups before the policy update. Its illustrated configuration uses four rounds, initial allocation two, expansion ratio two, and selection ratio one-half. [P10, pp. 5-8]

**Evidence.** The paper reports up to 2.3x fewer rollouts to reach a target math accuracy and 1.49x rollout efficiency in a coding comparison. Coding full-pass performance and average individual-test pass rate are distinct metrics. These are model/dataset-specific reported findings, not forecast speedups for AutoDSPy. [P10, pp. 8-10]

**Adaptation.** This is the strongest direct source for progressive, fixed-budget allocation using current reward variation. It motivates a comparator that invests extra candidates where informative contrasts are already visible, rather than rescuing homogeneous groups. Keep all collected candidates and recompute the empirical group mean once using the final group.

**Limits and critical reading.** VIGOR ranks multiple questions under one unchanged policy. With this project's one-question episode, a top-fraction ranking is degenerate; implementing it faithfully needs a multi-question collection window and delayed updates. Ranking historical questions from different policy states is an adaptation, not the same algorithm.

Its gradient theorem gives an **upper bound**, not a guarantee that high variance produces a large actual gradient or better generalization. Its binary-reward assumptions and standardized GRPO differ from the project's fractional-capable summed objective. Its Pareto model is an idealized tail model: bounded binary reward variance cannot literally have an unbounded Pareto support. Its convergence-timescale calculation should not be treated as a measured wall-clock law. [P10, pp. 6-7, 16-18]

A tiny homogeneous probe can rank a genuinely informative question at zero variance, starving it of future allocation. This is precisely the failure mode the rescue papers try to address.

**Priority: very high for variance-guided allocation; requires an explicit decision about batching.**

## 5. What to combine, and what to keep separate

### 5.1 The main scientific tension

Two plausible strategies emerge:

- **Exploit observed information:** VIGOR assigns more generation to questions with high observed reward variance. Under binary rewards this often favors success rates near the middle.
- **Recover missing information:** AERO and Reinforce-Ada spend more on questions whose small groups hide rare success/failure outcomes. GVM also allocates according to estimator uncertainty and acceptance rates, with regularization at extreme difficulty.

Neither strategy dominates by definition. An all-zero prefix could mean a difficult but recoverable question, little pipeline diversity, a bad label/extractor, repeated execution failure, or a nearly deterministic unsuccessful policy. These require different responses.

For an ideal iid binary group with success probability p:

```text
Pr(mixed outcomes at K) = 1 - p**K - (1-p)**K.
```

For example, at p=0.1, a fixed K=5 group has a mixed-outcome probability of approximately 0.4095, versus 0.7941 at K=15. These are illustrative mathematical probabilities, not project observations. They show why a homogeneous prefix does not prove further sampling is useless.

### 5.2 Reward assumptions must match the running baseline

The notebooks expose `original_semantic` and binary/exact modes. Original semantic rewards can be fractional. A success-count Beta-Binomial model is appropriate for a declared binary event, not automatically for fractional rewards. Turning partial credit into a success/failure threshold changes what the controller observes and needs a versioned definition; it must not silently change training rewards.

If the running baseline uses fractional rewards, prioritize empirical variance and continuous reward diagnostics, or define a separate binary controller signal and document it. If it uses binary rewards, the success-rate models are more natural. Use the actual run manifests when the collaborator's jobs finish; do not infer reward identity from notebook defaults. [REWARD_MODES.md](REWARD_MODES.md)

Judge/cache failures recorded as reward zero remain failures in the controller ledger. They must not silently become evidence that a question is semantically hard. Retaining their existing training treatment while excluding them from a difficulty estimate is a controller design choice that must be stated. Repeated identical pipelines also provide less evidence about policy exploration than genuinely distinct pipelines; iid confidence calculations can be overconfident when observations are dependent.

## 6. Proposed Adaptive-K implementation specification

Everything in this section is a **new project proposal**, not an existing implementation or a claim that the papers establish it as optimal.

### 6.1 First method: select K before collecting the current group

Use the current one-question episode structure and admissible sizes `{5,8,10,15}`. Select K using only prior training observations, then sample exactly that many candidates under one unchanged policy and perform the original centered, summed update.

Inputs can include question-level history when available, its age in policy updates, prior reward dispersion, failure rate, and observed execution cost. Avoid a new value network initially: question histories and lightweight scalar statistics fit the hardware and are easier to audit.

For a binary protocol, a candidate allocation utility is the expected probability of discovering a mixed group:

```text
v_q(k) = 1 - E[p_q**k + (1-p_q)**k | historical observations]
```

With `p_q ~ Beta(a,b)`, the moments can be computed from Beta-function ratios. At fixed posterior, `v_q(k)` is nondecreasing in k, so maximizing it alone always favors the largest K. Add an explicit cost term or constraint:

```text
choose k from {5,8,10,15} to maximize
    v_q(k) - lambda * estimated_cost_q(k)
```

Costs may initially be candidate counts; a later version can use measured execution/judge time. Lambda and controller rules must be selected on development evidence, then frozen. Include a prespecified randomized exploration probability over all four K values; persist its dedicated RNG. Use identical uninformative-history behavior for all new questions rather than inventing difficulty labels.

This proposal borrows posterior smoothing from AERO/Reinforce-Ada and constrained control from SAGC. It is not a full reproduction of those methods. Maximizing mixed-group probability is a proxy objective, not a guarantee of useful gradients or accuracy. Under fractional rewards, this particular binary utility is not applicable without a declared binary event; use a separately specified continuous-reward controller or begin with the bounded rescue comparator below.

Historical uncertainty is a real limitation: 200 episodes over a 200-question pool sampled with replacement leave many questions with few visits. Priors, cold starts, and policy drift can dominate. Discount history by elapsed optimizer updates and record observation age; a merely visit-based decay would ignore long gaps. If history coverage is inadequate, report that outcome instead of adding an opaque predictor to rescue the hypothesis.

### 6.2 Comparator: bounded sequential rescue

An interpretable within-question alternative:

```text
levels = [5, 8, 10, 15]
collect 5 candidates under the current policy
for next_level in [8, 10, 15]:
    if declared reward-contrast condition is satisfied:
        stop
    if failure guard or remaining global budget prevents extension:
        stop
    collect only next_level - current_count additional candidates
compute centered advantages using ALL collected candidates
apply the original summed Figure 2 loss once
```

For binary rewards, contrast means at least one zero and one one. For fractional rewards, define a development-fixed minimum spread/tolerance in reward units and report it; do not label it a binary-success model. Always record whether the stop was due to contrast, cap, failure guard, or budget exhaustion.

This is a simplified AERO/Reinforce-Ada-inspired comparator. It does not include posterior advantages, balanced retention, a log objective, or the papers' complete correction schemes. It may spend the most on hopeless groups and stop early on already informative ones. Compare both failures and successes rather than presenting it as inherently efficient.

**Selection effect:** final K depends on observed rewards. Fixed-K iid gradient identities do not apply unchanged. Reusing the collected empirical group mean does not prove unbiasedness. Analyze it as a distinct training procedure with held-out evaluation; test its estimator behavior in a small enumerable setting before making theoretical claims.

### 6.3 Second-stage comparator: VIGOR-style allocation window

A closer VIGOR adaptation needs several training questions collected under the same policy before updating. Choose the window size, total candidate budget, per-question minimum/maximum, ranking/tie-break rules, and treatment of inactive groups in advance. Allocate new candidates to groups with high reward variance while reserving some exploration budget for homogeneous groups.

This changes update timing and potentially prompt weighting. Its proper control is a static allocation method with the same window, loss aggregation, initialization, and total budget. The existing one-question Static-K experiments remain useful context but alone do not isolate this batching change.

### 6.4 Notebook integration points

Keep dataset workflows independent and follow the existing self-contained notebook architecture. Suggested insertion points, verified by function names in both notebooks:

1. **Configuration:** declare `allocation_mode`, controller version, allowed K values, prior/decay/exploration parameters, decision timing, and hard candidate cap. Keep baseline identity and reward identity separate.
2. **`initialize_experiment`:** initialize controller state, dedicated RNG, and budget ledger with a distinct Adaptive-K run namespace.
3. **`train_grpo`:** request the pre-group K, or collect increments for rescue. Record actual size explicitly. Freeze policy parameters throughout all collection rounds.
4. **`update_policy`:** validate matching nonempty trajectory/reward counts against the realized allowed K, rather than fixed `cfg['k']`. Preserve `group_advantages`, `figure2_loss`, replay checks, and the single optimizer step.
5. **Checkpoint payload/load:** persist controller state, RNG, statistics, question ages, allocated/consumed budgets, decision history, and protocol identity. Reject incompatible resumes.
6. **Journaling/recovery:** make decisions and incurred external work durable. If an incomplete episode is rolled back, controller learning state must roll back consistently, while the external cost ledger still records already incurred calls. Explicitly decide whether partial groups can be recovered or replayed.
7. **Summary/comparison:** support variable realized K without labeling adaptive histories as fixed-K runs. Extend existing analysis rather than overwriting previous exports or weakening their compatibility checks.

Existing `update_policy` replay validation is valuable: a decision controller must not change policy mode/parameters between sampling and replay. Preserve canonical pipeline identities and independently record exact duplicate pipelines, outputs, and judge-cache hits.

## 7. Pilot protocol and ablations

### 7.1 Use the running Static-K work appropriately

Keep the collaborator's jobs as the baseline effort already in progress. Once available, check their actual protocol identities and compare compatible runs. Select a Static-K comparator with validation/development results, not final held-out test outcomes. If final tests have already guided method design, treat that evidence as exploratory and reserve fresh confirmation data for the final adaptive claim.

Before starting a live adaptive pilot, freeze the selector, budgets, seed pairing, checkpoint-selection rule, evaluation budget, and primary comparison. The review does not mandate a new training horizon while the current multi-seed work is running.

### 7.2 Minimal informative comparison

Compare static K selected on development data, the pre-group selector, and bounded rescue. A randomized K schedule with a similar mean budget is an important allocation control: it helps test whether learned selection does more than occasionally choose small groups. Exact matching of the final realized K distribution after seeing outcomes is retrospective, so predeclare the randomized schedule or label that comparison exploratory.

Report two distinct resource views:

- **Equal episodes:** hold question exposure and update opportunities fixed; report the resulting candidate/time differences.
- **Equal candidate cap:** stop each arm before exceeding a common cumulative cap; update counts may differ. Use the same saved schedule prefix for shared episodes and a predefined extension if one arm can afford more episodes. At a remainder smaller than K_min, end training and report unused budget instead of silently creating an out-of-range group.

Equal candidates are not necessarily equal executor calls, tokens, or wall time. Report those measured quantities and account for checkpoint/validation overhead separately. Validation opportunities and tuning effort also consume resources and should be comparable across arms.

### 7.3 Ablations that answer specific questions

- **Adaptive versus random K:** does the selection signal outperform a simple mixture of sizes?
- **History versus current-prefix rescue:** does adaptation require current observations, or can past data guide a less selective sampler?
- **Rescue versus variance allocation:** is compute more valuable on homogeneous prefixes or already mixed groups? Use the matched-window control for a VIGOR-style arm.
- **Controller exploration on/off:** does the allocation become stuck because it stops observing certain K values or questions?
- **Reward variance versus gradient proxy:** only after profiling, does richer information justify its measurement cost? This follows GVM's motivation.
- **Optional objective controls:** K-dependent scaling, posterior advantages, or selective retention each require separately named experiments. Do not add several at once and attribute their joint effect to K selection.

Run dataset-specific analyses for GSM8K and HotpotQA. Their reward/answer extractors and task costs differ; do not pool their raw reward statistics as interchangeable observations.

### 7.4 Primary and diagnostic outcomes

The main outcomes are held-out task performance and measured resource cost across paired seeds. Retain paired question predictions and report each seed plus uncertainty in the performance difference. HotpotQA EM and answer F1 remain distinct. Training reward, Pass@k, majority voting, and single-pipeline inference accuracy answer different questions.

Diagnostics should include realized K counts, cumulative candidates, executor/judge requests, cached judgments, returned usage, execution/optimization/checkpoint/validation time, raw gradient norms, homogeneous-group classes, nonzero-gradient rate, and failure rates. Add rescue success per extra candidate, cap-hit rate, history age/coverage, selection probabilities, duplicates, controller overhead, and memory peaks where measurable.

"More informative groups" is a mechanism diagnostic, not proof of better accuracy. A small nonsignificant accuracy difference is not evidence of equivalence; define an acceptable loss and use adequate evaluation precision before claiming preserved accuracy.

## 8. Lightweight verification before live training

These checks are proposed implementation acceptance criteria; none was executed as a new Adaptive-K implementation in this review.

1. Fixed-selector mode reproduces existing fixed-K collection and updates on deterministic fixtures.
2. Every completed group has an allowed size; incremental collection never regenerates earlier indices or exceeds the global cap.
3. Final advantages use the complete final group. Gradient accumulation matches the summed Figure 2 reference for mixed and homogeneous binary/fractional fixtures.
4. Exact enumeration or repeated tiny-policy simulation compares fixed-K, pre-group selection, and reward-dependent stopping expectations. Claims of unbiasedness require the appropriate target and assumptions.
5. Constant rewards, canceled score directions, and Adam momentum are distinguished. Nonfinite rewards cannot enter updates.
6. Transport, parser, execution, and judge failures remain visible; they do not trigger unlimited rescue or masquerade as clean success-rate evidence.
7. A meaningful checkpoint/resume fixture restores the controller, its RNG, chosen K sequence, budgets, optimizer, and next scheduled question. Interruptions between allocation, collection, update, and checkpoint are covered.
8. Reward/cache/model identity drift is rejected using the existing protocol controls. Dedicated controller randomness does not accidentally change the saved question schedule or policy-initialization seed.
9. Variable-K run analysis rejects incompatible methods and labels actual budgets correctly. No final test data reaches selector tuning.

## 9. Defensible research contribution and open decisions

Adaptive rollouts already exist in these papers. "We dynamically change K" alone is not an established novelty claim. A plausible contribution to investigate is:

> Budget-constrained candidate allocation for a compact policy that selects modular pipelines executed by a separate frozen LLM, comparing uncertainty-driven signal recovery and variance-guided allocation across numerical and contextual QA tasks.

The interesting project-specific issues are executor-dominated cost, limited/aliased pipeline actions, duplicate pipelines, reward-judge noise, fractional rewards, sparse revisits, and the summed-loss weighting induced by variable K. Whether this is novel enough for publication requires a broader literature check beyond these ten local papers and empirical evidence. No priority or publication-readiness claim is made here.

Before implementation, resolve these choices explicitly:

- Which reward protocol do the current baselines actually use, and does the first selector model a binary or continuous signal?
- Is the first target fewer pipeline executions at similar accuracy, or better accuracy at a fixed execution budget?
- Will the primary selector choose K before collection, or deliberately use reward-dependent stopping?
- Is the original summed objective retained with scale effects documented? The recommendation here is yes for the first comparison.
- Is cross-question batching necessary now? The recommendation here is to defer it until the one-question methods provide useful evidence.
- What evidence would reject the controller hypothesis, including sparse history, overhead, rescue saturation, or accuracy loss?

## 10. Reading order and implementation priorities

Read **AutoDSPy** for the actual pipeline problem, **DeepSeekMath** for GRPO distinctions, **Reinforce-Ada** and **AERO** for signal recovery, and **VIGOR** for the competing variance-allocation strategy. Then use **GVM-RAFT** to question the allocation statistic, **DPPO** to audit selection/weighting claims, **SAGC** for constrained control and budget normalization, and **DAPO/GRPO-LEAD** for diagnostics and alternative objective choices.

**Adapt first:** bounded K choices, prior-only decisions or explicit capped rescue, uncertainty-aware history, complete cost accounting, and controller checkpointing.

**Evaluate next:** variance-based versus rescue allocation and random-budget controls, under matched protocol and held-out selection rules.

**Defer:** posterior advantage replacement, length-shaped rewards, balanced retention, large auxiliary models, packing, and GPU straggler claims. Each targets a different mechanism or changes the research question.

## 11. Local implementation and evidence references

- [Original reference notebook](../DSPy_GRPO.ipynb): `train_grpo`, original centered/summed loss.
- [GSM8K notebook](../Static_K_GRPO_GSM8K.ipynb): `generate_pipeline`, `group_advantages`, `figure2_loss`, `update_policy`, `train_grpo`, checkpoint functions, `analyze_saved_static_k`.
- [HotpotQA notebook](../Static_K_GRPO_HotpotQA.ipynb): corresponding independent implementations and HotpotQA evaluation.
- [Reward modes](REWARD_MODES.md): fractional/binary/exact modes and protocol identities.
- [Judge investigation](REWARD_JUDGE_INVESTIGATION.md): repeatability limits and compatible-baseline requirements.
- [Static-K results analysis](STATIC_K_RESULTS_ANALYSIS.md), Sections 8-10: limitations, observed prefixes, and future adaptive comparisons. Prefix analysis is descriptive, not a counterfactual training experiment.
- [Phase 1 audit](PHASE_1_AUDIT.md) and [decisions](DECISIONS.md): original-code issues, action approximation, and methodological constraints.
- [Earlier baseline plan](../../../STATIC_K_BASELINE_PLAN.md), Sections 7-8: allocation fairness and separate ablations. Its earlier suggested sizes/horizons are historical proposals, not replacements for the current `{5,8,10,15}` experiments or collaborator's ongoing jobs.

Review verification: all ten paper sections are present; all 24 local Markdown links resolve; code fences are balanced and no trailing whitespace was found. The fixed-K expected summed-gradient identity in Section 3.2 was independently checked by exact enumeration of a tiny Bernoulli policy at K=5,8,10,15. The illustrative mixed-group probabilities were recomputed. These are mathematical/document checks, not Adaptive-K training results. Key method pages visually checked: AERO p. 5, VIGOR p. 6, GVM-RAFT p. 8, Reinforce-Ada p. 8, DPPO p. 5, and SAGC p. 5.

Review completion: all ten local papers are represented above. No training, inference, experiment rerun, notebook edit, or retrospective reward conversion was performed. This document supports the next methodology decision; it does not assert Adaptive-K is implemented or that the ongoing 200-episode studies have finished.
