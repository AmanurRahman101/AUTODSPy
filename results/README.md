# Experimental evidence

**Find a study:** [results catalog](../docs/catalogs/RESULTS.md). **Understand the conclusion:** [research status](../docs/research/STATUS.md).

## Layout

Each existing subdirectory is an independent experiment namespace. Preserve the directory names: notebook configurations, journals, ledger paths, membership manifests, and checkpoint hashes refer to them.

Typical contents vary by protocol:

- Study/suite/comparison ledgers: frozen design, status, arm identities, and deadline.
- Membership/configuration files: question IDs, revisions, seeds, model identities, and schedule.
- Candidate journals: attempted executions and outcomes, including interrupted work.
- Checkpoint references plus `.pt` weights: latest resumable state, optimizer/RNG state, and selected validation policy.
- Evaluation progress and results: policy-bound recovery records and per-question outcomes.
- `report.json`, `final_audit.json`, or `closeout_audit.json`: comparison and integrity evidence.
- `training_terminal.log`: candidate-level progress.

## Reading results

Use the study's final audit to establish completeness before interpreting accuracy. Distinguish training, checkpoint-selection validation, confirmation, and official testing. A selected episode-0 checkpoint is the initial policy; its score does not demonstrate successful training.

Gross training cost includes every charged training attempt, including discarded/interrupted work. Evaluation costs remain separate. A reused evaluation role is not a newly executed pipeline. Missing metrics remain unknown. Logical model calls must not be relabeled as physical HTTP request counts.

## Copying or backing up

Copy the complete run directory, including Git-ignored `.pt` files. A Git clone alone is insufficient to resume local checkpoints. Preserve all manifests, journals, progress files, and checkpoint pointers together. The sibling paper library and local environments are also outside the portable tracked source bundle.

Do not rename or move completed studies, rewrite their status to make a run appear completed, reset deadlines, or migrate old checkpoints to a newer protocol. Superseded/failed work remains part of the research record and cost accounting.

The latest V3 screen completed and failed its advancement gate. No additional training or cloud work is launched by this documentation cleanup.
