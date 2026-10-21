# Completed K=5 GSM8K run diagnosis

Run: `results/gsm8k/k5/seed42/baseline/20261013T002408-4024eb86`.

## Finding

Policy concentration after the first update is supported by action probabilities, not merely inferred from duplicate sampling. All episodes 2?30 generated the same pipeline five times: `Predict ? math_problem -> answer ? stop`. Twenty-five groups have equal rewards and exactly zero advantages. Four groups (11, 15, 16, 27) have mixed rewards but identical action sequences and identical per-action log probabilities; their loss and gradient contributions cancel. Their saved gradients are small floating-point residuals, not exactly zero. No evidence here supports changing the Figure 2 objective or repairing detached gradients.

## All episodes

Candidate rewards are in saved order. Unique counts refer to complete action sequences (module, signature, stop); these fully specify the generated pipelines. ZeroVar uses exact equality of saved rewards and agrees with saved reward variance. Loss and gradient are recorded values, not reconstructed experimental results.

```text
Ep  Rewards                 Unique ZeroVar Identical Loss            Gradient
 1  [0.5,0,1,0,0]            5     no      no    1.247992e+00    2.173725e+02
 2  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
 3  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
 4  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
 5  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
 6  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
 7  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
 8  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
 9  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
10  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
11  [1,0,0,0,0]              1     no      yes   0.000000e+00    7.549560e-11
12  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
13  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
14  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
15  [1,0,0,0,0]              1     no      yes   0.000000e+00    1.044456e-11
16  [0,1,1,1,1]              1     no      yes   0.000000e+00    1.862183e-11
17  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
18  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
19  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
20  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
21  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
22  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
23  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
24  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
25  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
26  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
27  [1,0,0,0,0]              1     no      yes   0.000000e+00    1.917880e-11
28  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
29  [0,0,0,0,0]              1     yes     yes   0.000000e+00    0.000000e+00
30  [1,1,1,1,1]              1     yes     yes   0.000000e+00    0.000000e+00
```

## Fractional-mean groups

- Episodes 11, 15, 27: rewards `[1,0,0,0,0]`, mean 0.2, population variance 0.16, advantages `[0.8,-0.2,-0.2,-0.2,-0.2]`.
- Episode 16: rewards `[0,1,1,1,1]`, mean 0.8, population variance 0.16, advantages `[-0.8,0.2,0.2,0.2,0.2]` (stored 0.2 entries differ at floating-point precision).
- All four groups have one unique pipeline and one unique policy log-probability vector. They are NOT zero-advantage groups.

For fixed prompt and identical trajectory, write its summed log probability as s(theta). The prescribed loss is `L = -s(theta) * sum_i A_i`; the derivative is `-grad(s(theta)) * sum_i A_i`. Centering gives sum A_i = 0, so both cancel. Differing Llama execution/judge outcomes do not constitute different GPT-2 actions. This result assumes deterministic replay of the same policy state, as the notebook enforces through evaluation mode and replay matching.

## Probability and entropy evidence

All new GPT-2 forward passes were CPU-only and offline, using the local cached tokenizer/config and saved checkpoints. No Llama calls or GPT-2 optimizer updates were executed. Full distributions are in `policy-inspection.json`.

For the SAME episode-2 question (`gsm8k:train:3812`):

- Before training, checkpoint 0: P(Predict) = 0.6162459084; P(math_problem -> answer | Predict) = 0.0873765055. Module entropy = 0.6658720657 nats; signature entropy conditional on Predict = 2.3625491014 nats.
- After exactly one update, saved episode-2 rollout: P(Predict) = 0.9968154839; P(math_problem -> answer | Predict) = 0.9847486545. Module entropy = 0.0214886722 nats. Signature entropy lies between 0.0789321613 and 0.1191813364 nats.
- Checkpoint 25 on the same question: module entropy = 1.2415559014e-05 nats; signature entropy = 5.0087326153e-05 nats.
- Checkpoint 30 on the same question: module entropy = 1.1352803497e-05 nats; signature entropy = 4.4063591782e-05 nats.

No episode-1 checkpoint exists (only 0, 25, 30). The post-first-update signature distribution cannot be recovered exactly without replaying updates, which was deliberately not done. Its entropy bounds are h(p) <= H <= h(p)+(1-p)*log(14) for 15 signatures and the recorded selected probability p. Selected log probabilities in episode 2 are evidence from that state; checkpoint-0 entropy is an offline recomputation of the initial state on the same question. CPU checkpoint-0 replay of episode-1 selected log probabilities differs from recorded GPU values by at most 2.34e-05, a small device/numerical difference. Cross-device bitwise agreement is not claimed. Deterministic stop has entropy zero and is excluded from the reported module/signature entropy.

## Gradient and optimizer checks

- All 30 groups have five consistent candidate records. Rewards, actions, group centering, variance and loss recomputation pass. The largest difference between recomputed Python scalar loss and saved sequential FP32 loss is 2.98e-08.
- Actual notebook `replay_log_probs`, `figure2_loss` and `update_policy` were exercised with a tiny deterministic categorical model. Different trajectories with episode-1 rewards produce a nonzero gradient; sequential accumulation matches the retained-graph loss/gradient norm. Identical trajectories with all four mixed-reward patterns cancel to FP32 residuals below 1e-6. This validates plumbing at that seam; it is not a new full GPT-2 backward replay of the historical run.
- The saved episode-1 GPT-2 gradient norm is 2.173724654e+02. This proves that the historical first group had a gradient, but alone does not measure the Adam parameter displacement or establish a safe learning rate.
- The notebook calls Adam.step on every finite group. Checkpoints confirm all optimizer parameter states reached step 25 and step 30, with nonzero first-moment norms 1.7339026867 and 1.0238521527. Their ratio is consistent with five decays at beta1=0.9.
- Comparing actual checkpoint 25 and 30 weights gives parameter displacement L2 = 1.5209373307e-01, maximum absolute displacement = 1.7166137695e-05, and 85,167,251 changed elements (tied storage counted once). Thus near-zero current gradient does NOT mean no parameter movement. Adam momentum persists after the first informative gradient. The tiny-model check also reproduces zero current gradient with nonzero Adam movement.

## Other observed issues and limits

1. **Reward instability:** Episode 16's five execution outputs are exactly the same string, `{ (2/5) * 400 - 90 }`, yet the semantic judge scores are `[0,1,1,1,1]`; all numeric EM flags are false. Episodes 11 and 27 are the same question and have the same correct-first/incorrect-rest response pattern. Episodes 11, 15, 27 each have two distinct execution outputs despite identical policy trajectories. No execution or judge errors were recorded across the 150 training candidates. Temperature zero and configured seed do not establish reproducible judge outcomes; raw judge completions are not saved, so the origin of the inconsistency cannot be determined from these artifacts.
2. **Action approximation:** First-token scoring gives 15 signatures only nine distinct token IDs. Six collision groups force equal probabilities within each group; see `policy-inspection.json`. This is an inherited expressiveness limitation, not evidence of detached autograd. The collapsed math signature has a distinct first token, so its concentration is not explained by a collision.
3. **Reporting:** Fixed-point `.4f` hides the four residual gradients. `optimizer_updates=30` counts Adam calls, not 30 informative reward-driven updates. The current gradient norms are measured before Adam.step.
4. **Outcome:** Validation numeric EM decreased from 0.5 at checkpoint 0 to 0.2 at checkpoint 30 (10 questions each). The saved held-out 0.6667 result evaluates the selected checkpoint at episode 0, not the trained episode-30 policy. Small validation samples limit generalization claims.
5. **Run identity:** This run explicitly used Llama 3.2 3B, training pool 50, validation 10, held-out 30 and 30 episodes. These differ from the default research protocol and are recorded in config; this diagnosis does not silently substitute models or reinterpret it as a 200-episode baseline.

## Smallest necessary correction

No GRPO loss correction is supported. Preserve the Figure 2 sum, group-centered advantages and existing replay/gradient logic.

First improve diagnostics only: scientific-notation loss/gradient, unique pipeline count, per-group zero-variance flag, module/signature probabilities and entropy around updates. Record raw semantic-judge completions and request settings to investigate inconsistent identical-response scoring. These do not require changing the objective.

For training behavior, the observed same-question probability/entropy change supports a separately identified fresh short pilot at a smaller learning rate (e.g. 1e-5 instead of 1e-4), holding all other settings and the saved schedule fixed. This is the smallest controlled intervention to test whether the first update is too aggressive; it is not proven sufficient and was NOT run or implemented. Start from checkpoint 0 with a fresh optimizer, rather than resuming the collapsed optimizer state. Compare pre/post-update entropy and unique pipelines before committing expensive inference.

Skipping zero-gradient Adam steps would change the optimizer trajectory. Existing decision D011 preserves those steps for the primary baseline, so it is not proposed as an undocumented correction. Reward-rule changes, entropy regularization, clipping, and resampling for forced diversity are also outside this diagnostic correction.

## Reproduction and preservation

`Diagnosis.ipynb` contains self-contained log audits, small-model gradient checks and optional offline checkpoint probability inspection. `gradient-checks.json`, `checkpoint-movement.json`, `policy-inspection.json`, `episodes.json` and `episodes.csv` retain the evidence.

No original run artifacts, training notebook, learning rate, Figure 2 loss or optimizer behavior were modified. No new Llama inference or GPT-2 training was run. A derived diagnostic report/notebook and evidence files were added.

Training log SHA256: `358e4fcbc84589058bbe823933e68d1ab8e1daa3abbde519fdd80b967bad94c4`.
