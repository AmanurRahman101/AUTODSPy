# Research status

**Updated:** 21 October 2026. **Branch during organization:** `amanur`.

## Main conclusion

AdaGRPO implements automatic candidate-group allocation and reproducible local comparisons. We have not demonstrated the primary target: at least 20% fewer gross training attempts than matched Static-K=5 while preserving accuracy within two percentage points under the formal paired uncertainty analysis.

V2 is the main adaptive reference. V3's five-episode exploration warm-up showed no benefit in the latest screen. The immediate research question is why training degraded validation in that shared setup; its cause is not yet established.

## What is implemented

- V1 bounded sequential allocation and budget reservation.
- V2 probe/refinement allocation, deterministic candidate streams, numeric rewards, and partial evaluation recovery.
- Verified V2 study locking, resumable orchestration, compatibility checks, and cost reconciliation.
- V3 early fixed-K exploration followed by frozen V2 allocation.
- Shared original summed centered-reward loss and Adam behavior; sequential GPT-2 updates with the pinned execution model.

V2 preserves the original loss expression but changes collection and sampling. Outcome-dependent stopping is not estimator-equivalent to fixed K. At independent fixed K, the expected summed reward gradient scales with `K - 1`; this does not imply Adam-step equivalence.

## Evidence by study

### V2 development and V1/V2 comparison

The seed-43 development study passed its local screening gate. The later matched V1/V2 comparison observed **65% versus 55% mean confirmation accuracy**, but the paired 95% interval was **-10 to +30 percentage points**. The evidence does not establish general superiority. [Comparison report](../ADAGRPO_V1_V2_COMPARISON.md).

The corrected 100-question evaluation found V2 **68/100** versus **66/100** for each selected static checkpoint. Both static arms selected the same initial policy. This is not a demonstrated advantage over successfully trained static policies. [Evidence](../../results/adagrpo_fresh100/corrected_v2/report.json).

### Matched 30-episode diagnosis

V2 used **108 training attempts** versus Static-K=5's **150**: a **28% reduction**. Accuracy was **16/30 (53.3%) versus 21/30 (70%)**. Initialization was **20/30 (66.7%)**. These were previously examined questions; the result diagnoses a problem rather than confirming the formal target. [Report](../../results/adagrpo_matched30_diagnosis/report.json).

### Latest V2/V3 screen

Both arms completed 30 episodes under one seed, shared fresh membership, and a persisted two-hour limit.

- Training attempts: V2 **92**; V3 **97**.
- Validation in both arms: **7/10 at initialization**, **4/10 after training**.
- Both selected **episode 0**.
- Confirmation: V2, V3, and initialization each **12/20 (60%)**, with identical paired outcomes.
- Wall time: **85.57 minutes**; total new smoke/training/evaluation work: **283 pipeline attempts**.
- **233 engineering checks** passed; artifact audit passed; advancement gate **failed**.

The identical selected initial policies explain the confirmation tie. It is not proof that trained V2 and V3 policies are equivalent. V3 spent five extra training attempts without improving the selected result. This screen did not access the official test split. [Closeout audit](../../results/adagrpo_v3/closeout_audit.json).

## What remains unresolved

- Why both controllers worsened validation in the latest setup.
- Whether first-token action scoring limits learning: 15 signatures map to eight distinct first-token IDs. This is a known representational limitation, not a demonstrated cause of the observed accuracy drop.
- How allocation interacts with update magnitude, stochastic rewards, and Adam momentum.
- Whether a frozen adaptive rule transfers across seeds, tasks, and execution models while reducing manual K tuning.

Concentration and nonzero-gradient counts are diagnostics, not sufficient explanations of accuracy. Changes to action scoring, optimizer behavior, or the controller need a separately frozen protocol and new uncontaminated evaluation membership.

## What has not been established or authorized automatically

The primary research success criterion has not been met. The failed V3 gate does not authorize its conditional static-control continuation or formal cloud study. Existing prepared formal ledgers and workload estimates are preparation artifacts, not completed experiments.

Some earlier retrospective studies used specifically authorized, previously exposed official test questions. This does not turn the latest train-split confirmation into official testing, and no full formal test study has been completed.

## Next step

Analyze the recorded shared training degradation before spending more model compute or proposing another controller. Keep the original loss and current evidence intact; any methodological revision must have its own comparison and acceptance criteria.
