# Adaptive-K for AutoDSPy: evidence, current status, and research objective

Reviewed 19 October 2026 (Asia/Dhaka). Primary checkout: `experiment/static-k`, commit `54948cf`. This document records an analysis, not a new experimental result or an implemented algorithm. The user's current objective is to implement Adaptive-K; the earlier specification's Static-K-only scope describes the previous stage.

## 1. The central objective

Develop and evaluate a mechanism that chooses the number of candidate DSPy pipelines evaluated for each GRPO training question, reducing training computation while maintaining or improving held-out task accuracy against competitive fixed-K baselines.

The central deliverable is a working, experimentally validated Adaptive-K method. Reproducible notebooks, Static-K studies, reward diagnostics, and stopping-rule simulations provide the foundation. None alone demonstrates the central contribution.

The original AutoDSPy work already identifies this direction: Appendix A.1 compares fixed K=5,10,15 and explicitly proposes adaptive K as future work. Thus neither varying fixed K nor observing that equal rewards yield zero advantages should be presented as a new algorithm. [Original-paper analysis](ORIGINAL_PAPER_RESEARCH.md), local paper Appendix A.1, PDF pp.11–12; [official publication](https://aclanthology.org/2025.emnlp-industry.192/).

## 2. What the original system actually learns

There are two distinct models. The policy model, GPT-2 in the team's experiments, chooses a DSPy program. The execution model, Llama in these experiments, runs that program to produce an answer. The execution model also supplies semantic judgments when exact answer comparison fails. Policy optimization updates GPT-2; it does not fine-tune Llama.

The released concrete grammar is `module -> signature -> stop`. Its modules are `CoT` and `Predict`, with 15 named input/output signatures: 30 nominal programs. It executes one DSPy module per candidate. Richer multistage graphs, retrieval, ReAct, and teleprompter search shown or discussed elsewhere are not implemented by this grammar. K counts sampled candidates, which can repeat; it does not guarantee K different pipelines. [Sources: original notebook cells 15/20/22; current GSM8K notebook cells 12/16/18/20; original-paper analysis.]

At a training episode, select one question and target, collect K candidate trajectories at unchanged policy weights, execute and reward them, then compute:

\[
\bar r = \frac{1}{K}\sum_i r_i,\quad A_i=r_i-\bar r,\quad
L=-\sum_i A_i\sum_t\log\pi_\theta(a_{i,t}\mid s_{i,t}).
\]

One Adam step follows. The current baseline preserves this summed Figure 2 objective: no standard-deviation normalization, K division, PPO clipping, entropy bonus, GAE, KL term, or auxiliary imitation objective. At evaluation, one pipeline is sampled per question. Adaptive training K therefore does not automatically reduce the cost of single-pipeline test inference. [Sources: paper Figure 2; current GSM8K notebook cell 24 `group_advantages`/`figure2_loss`/`update_policy`, cell 26 `train_grpo`, cell 30 `evaluate_module`; DECISIONS D010–D012/D019.]

## 3. Original paper versus our experiments

The paper's main setup uses 200 training examples and 200 episodes; Table 3 gives K=5 and learning rate 1e-4. It studies several policy models and two execution models. GPT-2 plus Llama 3.1 8B is one configuration, reporting 82.4% GSM8K and 73.9% HotPotQA. Its fixed-K ablation reports GSM8K falling to 60% at K=15 while HotPotQA increases to 77.1%. These are published claims, not scores reproduced by this team. [Local PDF Table 1, Table 3, Appendix A.1; ORIGINAL_PAPER_RESEARCH.md.]

The checked-in reference notebook is a demonstration with smaller datasets, K=4 calls, different episode/LR settings, global model references, fragile scoring, and invalid final evaluation arguments. It is a methodological starting point rather than an executable reproduction of every published experiment. The paper also contains incompatible descriptions of training samples per episode and semantic evaluation. Our protocol explicitly follows one sampled question per episode and the Figure 2/released sum loss. [PHASE_1_AUDIT.md §§2–4; ORIGINAL_PAPER_RESEARCH.md.]

The team's comparable four-run study instead uses Llama 3.2 3B, learning rate 1e-5, a 50-question training pool, 30 episodes, 10 validation questions and 30 held-out test questions, seed 42. Only 22 distinct training questions are encountered in those 30 replacement draws. These differences prevent treating the team's 76.7% as a direct reproduction of or controlled comparison with the paper's 82.4%. [Raw run configs and histories; STATIC_K_RESULTS_ANALYSIS.md §1.]

The current GSM8K notebook's top-level learning-rate default is still 1e-4 and its execution-model default is Llama 3.1 8B. The saved successful comparison uses explicit overrides. Future pilots must load or deliberately reproduce the intended configuration rather than assume opening the notebook recreates those four runs. [Current notebook cell 4; saved run configs.]

## 4. What the team has completed

### Experimental infrastructure

The two independent notebooks, `Static_K_GRPO_GSM8K.ipynb` and `Static_K_GRPO_HotpotQA.ipynb`, contain configuration, model initialization, data selection, policy sampling, DSPy execution, rewards, training, checkpoint/resume, validation, explicit test operations, statistics and plots. They preserve the original reference notebook. Fresh starts, independent K run directories, immutable model/dataset identifiers, saved split membership, and a shared replacement schedule make comparisons more reproducible. [PHASE_2_REPORT.md, PHASE_3_REPORT.md, PHASE_4_REPORT.md; current notebooks.]

Sequential differentiable replay accumulates the same summed policy gradient while avoiding retained activations for all K candidates. Checkpoints include optimizer/RNG/history and support durable resume. Validation selects the highest exact-match checkpoint with earliest tie-breaking; held-out outcomes do not select checkpoints. Historical verification reports document synthetic checks and reduced live integration; those reports are prior evidence, not tests rerun during this analysis. [DECISIONS D018–D028; phase reports; notebook cells 24–36.]

HotpotQA uses all supplied distractor context without retrieval and answer EM/F1 evaluation. Its infrastructure and a reduced integration check are documented, but this checkout contains no completed multi-K HotpotQA research study. The present `results/` directory contains GSM8K artifacts only. Historical engineering-check paths in the reports are not present here. [PHASE_3_REPORT.md; current HotpotQA notebook; directory inventory.]

### Stability investigation

An earlier K=5 run at LR=1e-4 concentrated on one pipeline after its first update. The diagnosis establishes this using probabilities on the same question, not just repeated samples. Episodes 2–30 use identical actions within each group; several groups have different rewards but nearly canceling gradients. Adam momentum continues to move parameters despite almost zero current gradient. [docs/diagnostics/k5-20261013/REPORT.md and policy-inspection.json; saved early K=5 run.]

A documented ten-episode pilot at LR=1e-5 retained diversity and more nonzero gradients, motivating the later four-run comparison. This is short-window stability evidence, not a proof of long-run stability. Its report references a pilot directory ending `032749-85f3db7c` that is absent in this checkout; a different ten-episode pilot directory ending `030331-484147ec` is present. Do not silently equate those run identities. [STABILITY_PILOT_REPORT.md; current results inventory.]

### Completed comparable GSM8K study

The four compatible LR=1e-5 runs are:

- K=5, run `20261013T174921-b1ad8988`: 150 training candidates; 23/30 held-out correct; best validation 7/10 at episode 30; 16/30 equal-reward groups; measured training time 668.57 seconds.
- K=8, run `20261013T181104-322720b3`: 240 candidates; 20/30 held-out correct; best validation 5/10 at episode 0; final validation 4/10; 15/30 equal-reward groups; training time 954.30 seconds.
- K=10, run `20261013T183759-77b8ce57`: 300 candidates; 23/30 held-out correct; best validation 6/10 at episode 30; 12/30 equal-reward groups; training time 954.70 seconds.
- K=15, run `20261013T190031-8d8e40ec`: 450 candidates; 21/30 held-out correct; best validation 5/10 at episode 0; final validation 4/10; 13/30 equal-reward groups; training time 1656.42 seconds.

These counts/checkpoint episodes were independently read from raw configs, training and validation JSONL, best pointers and test artifacts in this analysis. Times and additional metrics are preserved in the existing analysis CSV/report; no model inference or training was rerun. The separate early LR=1e-4 K=5 run must not be substituted into this comparison. [results/gsm8k/static_k_analysis/summary.csv; raw directories; STATIC_K_RESULTS_ANALYSIS.md §§1–3.]

Mean unique pipelines per group are 3.67, 4.73, 5.47 and 5.63; unique/K declines from 73.33% to 37.56%. Informative-current-gradient episode counts are 14,15,18,17, using recorded L2 norm >1e-8. There are no recorded execution failures in these four histories. [summary.csv; STATIC_K_RESULTS_ANALYSIS.md §§4–6.]

The validated interpretation is limited: increasing K costs more and does not improve accuracy monotonically in this preliminary study. K=5 and K=10 tie on a small held-out set. K=8 and K=15 test the selected untrained starting checkpoint, so their scores are not training gains. Their identical starting-policy test pipeline choices nevertheless produce one differing correctness outcome, demonstrating execution variability. With one seed, 10 validation questions, 30 test questions and only initial/final validation, optimal-K and significance claims are unsupported. [STATIC_K_RESULTS_ANALYSIS.md §§3/8.]

### Existing offline analysis

Notebook section 24 exports per-candidate/episode metrics, checkpoint and configuration audits, cost/diversity summaries, paired test outcomes, judge disagreements and ordered prefix statistics. On K=15 histories, prefixes of length 5/8/10/15 have reward variation in 15/16/17/17 groups. I independently confirmed these counts from raw reward arrays. Thus candidates 11–15 add no newly reward-varying groups in those observed histories, although they change group means and gradients. This is already useful descriptive groundwork for simulations; it is not an implemented adaptive controller. [STATIC_K_RESULTS_ANALYSIS.md §9; observed_candidate_prefixes.csv; current GSM8K cell 46.]

## 5. Reward consistency: what exists and what remains

Both current notebooks award exact correctness 1 and otherwise invoke a bounded fractional Llama semantic judge. They correct number/text normalization and reject malformed/nonfinite judge scores. GSM8K additionally records raw semantic-judge traces and policy distributions in its current source. Existing four-run histories predate some of that instrumentation, so full true policy entropy and raw judge traces cannot be reconstructed for them. [Current GSM8K cells 16/22/26; HotpotQA cell 22; STATIC_K_RESULTS_ANALYSIS.md §§5–6.]

The three proposed selectable modes `original_semantic`, `binary_semantic`, and `numeric_exact` are not implemented as a mode switch in the current notebooks. A `numeric_exact_match` metric or a configuration field named `semantic_fallback` does not establish a configurable three-mode reward implementation; the current `compute_reward` unconditionally uses semantic fallback for mismatches. [Direct current-source inspection.]

Byte-identical answer strings sometimes received different semantic scores; strict numeric mismatches sometimes received full reward. A semantic judge can recognize a correct answer missed by extraction, but can also introduce false credit. Binarizing judge outputs alone cannot guarantee consistency. Fixing this requires an explicit rule and evidence from stored answer/target/judge inputs. [STATIC_K_RESULTS_ANALYSIS.md §6; reward_disagreements.csv; diagnosis report.]

For a first controlled GSM8K Adaptive-K experiment, deterministic numeric exact reward is a reasonable candidate protocol, subject to auditing extraction limitations. If selected, rerun the relevant fixed-K comparators with exactly that same reward. Keep the semantic-reward histories as a different protocol. For HotpotQA, exact match, semantic equivalence, and F1 address different issues and require a task-specific decision. This paragraph is a recommendation, not an implemented or approved final reward design.

## 6. Important earlier work on other branches

Read-only inspection found `feature/adaptive-k-grpo` at `f4c5ff9`, containing `run_adaptive_k_grpo_v3.py` and older thesis/modification notes. Therefore the statement "no Adaptive-K code exists anywhere" would be incorrect. There is no adaptive controller in the current Static-K notebooks, and the older prototype is not a validated successor to them.

That prototype samples a two-candidate probe, stops on reward contrast or all-success, otherwise expands up to a cost-derived ceiling (default max 6). It also standard-deviation-normalizes advantages, scales them by sqrt(probe/actual K), introduces an all-success imitation loss, clips gradients, cycles a mixed dataset without HotpotQA context, and uses the older answer extractor. It lacks the current validation/checkpoint-selection protocol. The branch has no tracked `adaptive_metrics.json` demonstrating its proposed results. [Script at commit 4f8cbd2; THESIS_PROGRESS_REPORT.md and MODIFICATIONS_FROM_VANILLA.md on that branch.]

Several older claims should not be carried forward as established findings:

- The paper's fixed default is K=5; K=4 belongs to demonstration/older-run settings.
- Equal rewards yield zero centered advantages with or without standard-deviation normalization. Normalization does not discover or fix that property.
- The old 0.9622 final mean reward is a training reward statistic, not 96.22% held-out accuracy.
- An imitation loss changes the objective and policy; it does not guarantee no drift.
- The proposed square-root scaling has no verified variance bound in these artifacts.
- Hardcoded module cost weights are not measured latency/token costs. ReAct has a weight but is absent from its action space.
- Per-episode `max(4-K,0)` hides extra evaluations when K>4; net budget accounting must charge every attempted candidate and all judge calls.
- No exhaustive literature search here establishes the older notes' claim of uncontested novelty.

Other fetched branches contain baseline plots (`adib-baseline-stats`), a separate policy/statistics workflow and K=4,8,16 comparison CSV (`adib-policy-model-comparison`), an additional notebook and execution parsing work (`akash`), and an earlier GSM8K reproduction harness (`experiment/grpo-gsm8k-experiment`). Those inventories were inspected for context; their results were not pooled with the current four-run study. They require protocol audits before scientific aggregation. No branches were switched or merged.

## 7. What Adaptive-K needs to decide

The algorithm must choose when another evaluated pipeline is worth its cost. It may choose K before an episode from policy/history signals, or collect a small probe and decide whether to expand. These are distinct designs. Existing offline logs can help screen the second, while current policy diagnostics support the first.

Useful signals include reward contrast, unique action trajectories, conditional policy entropy, recent informative-gradient frequency, and measured candidate costs. Each has a limitation: reward contrast can reflect judge noise; duplicates can produce varying outputs; entropy does not measure correctness; current-gradient norms require paid replay/backward; cost depends on answer length and fallback judgments. A first design should use a small explicit set of signals, not bundle unrelated interventions.

Design constraints derived from the actual baseline:

1. K must be at least two for a nontrivial centered group. With K=1 the current objective always has zero advantage.
2. Keep policy weights fixed during candidate collection and reward evaluation. Once actual K is final, recompute the mean and advantages over the complete collected group and perform one update.
3. Variable K must propagate through training, updater checks, metrics, checkpoint compatibility, recovery, comparison and plots. Replacing the collection `for` loop alone is insufficient: `update_policy` currently requires exactly configured K and downstream audits assume that budget.
4. A zero-reward probe does not prove the question is unsolvable; an all-success probe does not prove universal mastery. Use bounded decisions and record reasons.
5. Differing rewards are insufficient when all policy action trajectories are identical: their centered score-gradient contributions cancel under deterministic replay. Prefer diagnostics that distinguish reward contrast from action contrast.
6. Outcome-dependent stopping changes the group distribution. A same-group empirical baseline and random stopping time must not be assumed to preserve the fixed-K estimator. Analyze and disclose the resulting algorithm, and compare it empirically.
7. The original summed loss couples K to update scale. Under fixed-K iid assumptions its expected gradient scales with K-1; that derivation does not automatically hold under adaptive stopping. Preserve the primary objective and study any rescaling as a separate ablation if pursued.
8. Preserve the baseline's finite-group Adam-step behavior for a clean first comparison. Skipping steps, adding imitation or entropy, or forcing unique samples are separate changes.
9. Training targets may inform allocation. Held-out targets must not guide inference-time allocation or controller tuning.

These constraints come from current notebook cells 16/24/26/28/36/46, DECISIONS D010–D012/D019–D021, the diagnosis report, and mathematical consequences of the specified loss. They define required implementation work rather than existing Adaptive-K behavior.

## 8. What the literature contributes

Primary sources confirm that adaptive rollout allocation is already an active research area. AERO combines adaptive allocation, selective rejection and posterior estimation. DAPO addresses uninformative groups through dynamic sampling within a larger RL system. GRPO-LEAD focuses on reward design and difficulty-aware advantage weighting. SAGC changes group size to control synchronous rollout stragglers. These are related but different interventions. [AERO](https://arxiv.org/abs/2602.14338), [DAPO](https://arxiv.org/abs/2503.14476), [GRPO-LEAD](https://arxiv.org/abs/2504.09696), [SAGC](https://arxiv.org/abs/2606.02218).

VIP additionally allocates rollout budgets using predicted per-prompt success probabilities and gradient-variance estimates. It is relevant when considering a controller that decides K before collecting the current group. [Official ICLR 2026 paper record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html).

The defensible project question is how to allocate expensive DSPy pipeline evaluations under AutoDSPy's concrete policy/action/reward system. Savings or guarantees reported for token-generation policies or distributed synchronous workers do not automatically transfer to a sequential local pipeline sampler. This analysis verifies the positioning of named papers; it is not an exhaustive novelty assessment or a claim that the team's reported literature reading has been independently audited.

## 9. Practical next work and success criteria

Proceed toward Adaptive-K rather than treating Static-K studies as the endpoint:

1. Freeze the first comparison protocol: execution model/digest, reward, dataset splits/schedule, LR, episode budget, validation cadence and inference controls. Audit extraction and judge consistency as needed. A reward change requires matched static reruns.
2. Screen a few bounded stopping/allocation rules using existing ordered logs. Report actual K, estimated candidate savings, retained reward/action contrasts and stop reasons. Call these offline screening results, not predicted trained-model accuracy. Some smaller-K histories cannot evaluate expansion beyond their recorded K.
3. Implement one minimal Adaptive-K version in the notebook workflow, preserving the specified loss and reproducibility behavior. Start with GSM8K; port to HotpotQA after the mechanism is understood. The old script supplies ideas, not a drop-in controlled implementation.
4. Run a short explicit pilot to establish variable-K control flow, finite/replay-consistent updates, recovery and measured costs; then perform longer experiments, targeting 200 episodes and multiple independent seeds where feasible. Do not launch expensive experiments automatically.
5. Compare against at least the strongest inexpensive current comparator, fixed K=5, and useful larger-K settings. Report equal-episode and equal-computation comparisons. Controller thresholds and checkpoint choices use validation only; evaluate frozen choices on the same held-out questions.

Primary success is lower measured training cost with held-out accuracy within a prespecified acceptable loss, or higher accuracy at comparable cost, across credible repetitions. Define that accuracy tolerance before reading final test outcomes. Beating K=15 on cost while fixed K=5 already achieves the same score would not by itself establish a useful advance.

Measure total attempted pipelines, execution and judge LM calls, input/output tokens, measured training and evaluation time separately, actual-K distribution and stop reasons, reward/action diversity, zero-advantage groups, meaningful gradient counts and policy concentration. Account for controller overhead, failures, retries and discarded work. A lower zero-advantage percentage alone can be produced by selection and does not prove better learning. Preserve full training trajectories for learning statistics and validation-selected checkpoints for final testing.

## 10. Evidence boundaries and housekeeping

This review read the local paper analysis, original and current notebook code, research specifications and phase/diagnostic reports, raw configs/training/validation/test metadata, saved analysis CSVs, and selected other-branch source/documentation through Git. It independently recomputed simple history counts and prefix variation. It did not rerun the historical test suite, materialize checkpoint tensors, execute notebook training cells, invoke Llama, or launch experiments. No implementation file or historical result was changed.

`graphify query` was attempted as AGENTS.md requires; the executable is unavailable in the current shell. Notebook and artifact inspection supplied the code evidence. No graph-backed semantic verification is claimed.

The README's AutoDSPy DOI points to a different 2024 paper. The verified AutoDSPy DOI is `10.18653/v1/2025.emnlp-industry.192`; this review records the correction without editing that historical file. Several progress documents retain earlier completion boundaries and should be read with their dates and later experiment reports, rather than as a current absence-of-results statement.

The project is at a transition: a usable Static-K research foundation and preliminary GSM8K evidence exist; an older Adaptive-K prototype exists elsewhere; a controlled Adaptive-K implementation integrated with the current workflow and convincing experimental validation remain to be completed.
