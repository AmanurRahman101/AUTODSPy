# Results catalog

Each namespace represents a distinct protocol or experiment family. Do not combine their accuracies as if all models, rewards, memberships, and checkpoint selections were matched.

## Latest completed experiment

**[adagrpo_v3/](../../results/adagrpo_v3/)** - early-exploration V3 versus frozen V2.

- [Verification report](../../results/adagrpo_v3/VERIFICATION_REPORT.md).
- [Independent closeout audit](../../results/adagrpo_v3/closeout_audit.json).
- [Study ledger](../../results/adagrpo_v3/screen_seed42/study.json).
- [Structured comparison](../../results/adagrpo_v3/screen_seed42/report.json).
- [Candidate progress log](../../results/adagrpo_v3/screen_seed42/training_terminal.log).

Both arms completed; gate failed. V2/V3 used 92/97 training attempts and both selected initialization. Confirmation was 12/20 for each and initialization. This study did not access the official test split.

## Adaptive development and controls

- **[adagrpo/](../../results/adagrpo/)** - original V1 smoke, pilot, recovery, and Static-K=4 control. See [V1 report](../ADAGRPO_IMPLEMENTATION_REPORT.md).
- **[adagrpo_v2/](../../results/adagrpo_v2/)** - seed-43 development study, smoke, static controls, and plots. [Suite ledger](../../results/adagrpo_v2/local_suite.json); [report](../ADAGRPO_V2_REPORT.md).
- **[adagrpo_v1_v2/](../../results/adagrpo_v1_v2/)** - paired V1/V2 controller comparison on seeds 43/44. [Ledger](../../results/adagrpo_v1_v2/comparison_ledger.json); [final audit](../../results/adagrpo_v1_v2/final_audit.json); [interpretation](../ADAGRPO_V1_V2_COMPARISON.md).
- **[adagrpo_verified/](../../results/adagrpo_verified/)** - V2 infrastructure checks, accounting, and prepared study workflows. [Verification report](../../results/adagrpo_verified/VERIFICATION_REPORT.md). Prepared formal work is not completed model evaluation.

## Checkpoint evaluations and 30-episode diagnosis

- **[adagrpo_fresh100/](../../results/adagrpo_fresh100/)** - includes both superseded and corrected evaluation attempts. Use [corrected_v2/report.json](../../results/adagrpo_fresh100/corrected_v2/report.json) for findings; retain superseded work in gross costs.
- **[adagrpo_historical30/](../../results/adagrpo_historical30/)** - V2 against archived Static-K=5; [report](../../results/adagrpo_historical30/report.json) and [audit](../../results/adagrpo_historical30/final_audit.json). Evaluation membership was previously examined, and the archived methods were not fully matched.
- **[adagrpo_matched30_diagnosis/](../../results/adagrpo_matched30_diagnosis/)** - matched new Static-K=5 control plus initialization; [report](../../results/adagrpo_matched30_diagnosis/report.json) and [audit](../../results/adagrpo_matched30_diagnosis/final_audit.json). Not an independent fresh-test confirmation.

## Ahnaf's original Static-K evidence

**[gsm8k/](../../results/gsm8k/)** contains `k5/`, `k8/`, `k10/`, `k15/`, shared selections, extracted analysis, and comparison figures.

- [Statistical analysis artifacts](../../results/gsm8k/static_k_analysis/).
- [Comparison figures](../../results/gsm8k/static_k_comparison/).
- [Research interpretation](../STATIC_K_RESULTS_ANALYSIS.md).
- [30/200-episode benchmark context and unresolved metadata](../ADAGRPO_AHNAF_STATIC_K_CONTEXT.md).

The supplied 200-episode figure bundle has model/test-size questions documented in the context report. It must not be silently treated as the same protocol as the local 3B numeric-reward studies.

## Audit evidence

[docs/audit_artifacts/](../audit_artifacts/) contains engineering checks, source-preservation snapshots, and previous artifact audits. Some newer studies keep their own audits inside their result namespace. Check the date and scope: zero inference in an audit does not mean its preceding study did no training.

## Cost and recovery rules

Read [results/README.md](../../results/README.md) before copying, resuming, or comparing run directories. Gross training attempts include failed/interrupted work; evaluation attempts are accounted separately. Missing token or memory data means unknown, not zero.
