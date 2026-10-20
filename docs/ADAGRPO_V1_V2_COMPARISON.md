# AdaGRPO: matched V1 versus V2 comparison

V2 achieved higher observed confirmation accuracy on both matched seeds: **65% mean versus V1's 55%**, with **80 versus 82 total training attempts**. General superiority remains inconclusive because the descriptive paired interval includes zero.

## Design

- Frozen original V1 controller versus frozen V2 controller under common candidate-stream-v2 infrastructure. V1 is a controller ablation, not the historical global-RNG V1 experiment.
- Paired seeds 43 and 44; identical fifty-question training pool, twelve replacement-sampled episodes, twenty shared checkpoint-selection questions, validation at episodes 0 and 12, earliest ties.
- Both methods have a gross training ceiling of 48. Same GPT-2 initialization, Llama digest, numeric reward, original summed loss and Adam settings.
- V2 seed 43 is the existing completed archive, reused read-only after audits. Three training arms and all four confirmation evaluations are new.
- Twenty newly frozen train-split confirmation questions exclude all recorded prior train/validation membership. They never select checkpoints. No controller tuning or official-test access.

## Results

### Seed 43

- Confirmation: V1 **55%**, V2 **70%** (+15 percentage points).
- Paired outcomes: V2 alone correct on 5; V1 alone correct on 2.
- Gross training attempts: V1 **40**, V2 **42**.
- Selected validation: V1 80% at episode 0; V2 85% at episode 12.
- Actual K: V1 `[2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2]`; V2 `[4, 4, 4, 2, 2, 6, 2, 2, 2, 2, 6, 6]`.

### Seed 44

- Confirmation: V1 **55%**, V2 **60%** (+5 percentage points).
- Paired outcomes: V2 alone correct on 2; V1 alone correct on 1.
- Gross training attempts: V1 **42**, V2 **38**.
- Selected validation: V1 70% at episode 0; V2 80% at episode 12.
- Actual K: V1 `[4, 2, 4, 4, 2, 6, 4, 4, 2, 2, 4, 4]`; V2 `[2, 6, 2, 2, 6, 4, 2, 2, 2, 6, 2, 2]`.

Across seeds, V2 minus V1 confirmation accuracy is **+10.0 points**. The descriptive 10,000-resample paired hierarchical-bootstrap 95% interval is **[-10.0, +30.0] points**. The same question resample is shared across seeds; seeds and questions are resampled with replacement.

## Verification and cost

- A fresh-kernel execution of all twenty definition cells passed without launching model work.
- 192 synthetic checks passed, including the retained 163 and exact original V1 controller AST equality.
- All four training artifacts and confirmation journals audited; per-seed initialization, selection, model/runtime, initial validation outcomes and pre-divergence candidate prefixes agree.
- 235 historical notebook/result files retain their pre-study SHA-256 values, including checkpoints.
- New work: 120 training attempts and 200 evaluation attempts; 331 logical model calls.
- New suite wall time: 86.04 minutes within the persisted 120-minute deadline; archived V2 seed43 costs are excluded from this new-work total but included in the controller cost comparison.
- No official GSM8K test split, Colab work or automatic commit.

## Interpretation limits

Two training seeds and twenty confirmation questions have limited power. Checkpoint selection has only two time points and training lasts twelve episodes. The earlier V2 development validation split is reused for selection, while the confirmation set is fresh. Reusing one archived arm saves compute but is not an independent rerun of that arm. Outcome-dependent stopping changes the training estimator even though both controllers preserve the same summed loss; this comparison does not establish estimator equivalence. These results do not establish the formal accuracy/cost research target or general superiority. Do not retune against these confirmation questions.

## Evidence

- [Comparison notebook](../AdaGRPO_V1_V2_Comparison.ipynb)
- [Frozen ledger](../results/adagrpo_v1_v2/comparison_ledger.json)
- [Detailed final audit](../results/adagrpo_v1_v2/final_audit.json)
- [Recorded evidence](audit_artifacts/adagrpo_v1_v2_comparison.json)
- [Figure](../results/adagrpo_v1_v2/comparison.png)
