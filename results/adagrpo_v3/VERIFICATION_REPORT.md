# AdaGRPO V3 implementation verification

The self-contained V3 notebook adds five initial K=5 groups and then dispatches to the frozen V2 refinement rule. Both study arms retain the original centered-reward summed loss, policy action scoring, candidate streams, optimizer, numeric rewards, execution model, and hardware configuration.

## Before live execution

- 233 notebook-local checks passed: 163 retained V2 checks and 70 additional checks. These are synthetic engineering fixtures, not model accuracy results.
- Exact original-V2 trajectories, rewards, decisions, losses, gradients, updated parameters, and Adam states agreed under matched fixtures.
- Exhaustive budget checks covered 3,630 episode/gross-cost combinations, including warm-up reservations, pacing debt, interruption costs, and odd allowances.
- Synthetic interruption/recovery exercised collection, validation, the warm-up boundary, multi-arm orchestration, completed-arm reuse, incompatible checkpoints, concurrent runners, and frozen deadlines.
- Fresh-kernel definitions succeeded with model-loading and network factories blocked. No training or inference runs when definition cells execute.
- Static source comparison found 28 of the 32 tracked experimental functions unchanged. Intentional replacements are the controller decision, budget snapshot, collector, and official-test guard. The V2 control still delegates to the original implementations.
- Preparation recorded SHA-256 values for 389 historical notebook/result files and passed a read-only preservation audit. Final preservation is checked again when the study finishes.

## Authorized live study

The frozen seed-42 screen has 30 episodes per arm, one common fresh 50-question training pool, 10 validation questions at episodes 0/30, and 20 confirmation questions. All questions come from the pinned GSM8K training split; the official test split is deferred. Confirmation follows both checkpoint selections. Selection uses highest validation accuracy with earliest ties.

The clean upper workload is 354 pipeline executions, including a deliberately partial two-episode integration smoke. One persisted 7,200-second clock covers all model work and resumes. Static-control continuation requires a passing screen and separate approval.

The live smoke completed two K=5 groups, with exact policy and Adam recovery after each group. It used 10 training and four validation attempts. Its intentional pause uses the inherited wall-limit exception as a control signal; it does not indicate expiry of the study deadline or a discarded committed group.

## Evidence

- `self_checks.json`: complete check names and counts.
- `fresh_kernel_definitions.json`: blocked-factory startup result.
- `experimental_equivalence.json`: unchanged and intentionally changed functions.
- `preparation_audit.json`: initial preservation and preparation audit.
- `screen_seed42/study.json`, `study_clock.json`, and `membership.json`: frozen study and timing identities.
- `screen_seed42/training_terminal.log`: per-candidate progress and operational events.

Accuracy, screening outcome, and final artifact integrity must be taken from the completed study's report and audit. No success claim follows from engineering checks alone.

## Completed local screen

The study completed in 85.57 minutes. Both 30-episode arms audited successfully, and the independent closeout confirmed preservation of all 389 historical files. V2 used 92 training attempts and V3 used 97, with zero discarded training attempts.

Validation fell from 70% to 40% for both methods, so both selected initialization. Initialization and both selected policies scored 12/20 (60%) on confirmation. V3 failed the predefined improvement gate. The identical selected policies explain the zero paired difference; this is not evidence of trained-algorithm equivalence or formal noninferiority.

Gross new cost was 283 pipeline attempts, including smoke and evaluation. Twenty confirmation records were reused across roles. The final two smoke-validation records are durable completed work and included in gross cost; they are outside the partial training run's committed validation history. See `closeout_audit.json` for that accounting distinction. No further model experiments or official testing were launched.
