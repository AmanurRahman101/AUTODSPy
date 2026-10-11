# Phase 4 — Final experiment management and reproducibility

Completed: 2026-10-12 (Asia/Dhaka), within the lightweight verification scope.
Both independent notebooks now have the final comparison safeguards, resource
summaries and documentation. Full static-K experiments remain for the user to run.
No new Llama inference, pretrained-model training, or Adaptive-K implementation
was performed in this phase.

## Changes and verification

- Updated both notebooks' statistics, comparison, visualization and self-check
  cells. Comparisons require configuration snapshots, selection manifests,
  complete episode/validation histories, durable latest/best checkpoints and
  held-out provenance. They reject duplicate paths and incompatible protocols.
- Added recorded GPU peaks, failure counts, LM time and journal call/token/time
  totals, including observed work discarded during recovery. Missing measurements
  and incomplete historical journals remain unavailable. Interrupted in-flight
  work without a completed journal entry cannot be reconstructed.
- Added markers to episode curves so a one-episode engineering check has a visible
  recorded point. No result is synthesized to fill an empty plot.
- Expanded [the guide](../STATIC_K_EXPERIMENT_GUIDE.md), updated progress, decisions
  and the implementation checklist, and created this report plus
  [machine-readable evidence](audit_artifacts/phase4_verification.json) and
  [clean-kernel outputs](audit_artifacts/phase4_clean_kernels.json).
- Final fresh `autodspy-gpu` kernels passed **111 GSM8K and 135 HotpotQA synthetic
  assertions (246 total)**. Schema, ASTs and all 21 numbered sections passed.
  Checks cover Figure 2 gradients at every K, nonzero synthetic interrupted
  recovery, interrupted validation, metric/input isolation, checkpoint rejection,
  corrupted history/manifest provenance, missing configuration snapshots,
  cross-run protocol drift, all six modes and sequential all-K starts/releases.
- Read-only comparisons passed on the existing Phase 2/3 live engineering runs.
  Generated eleven GSM8K and twelve HotpotQA figures from those actual artifacts.
  The accuracy-versus-training-budget line coordinates match recorded validation
  rows. Final plot paths are recorded in the evidence JSON; these tiny runs are
  integration evidence, not benchmark results. Historical run files were preserved.
- Configuration, service setup, data/schedule, action/policy, execution/reward,
  GRPO/training, checkpoint and evaluation code cells remain identical to their
  committed Phase 2/3 versions. Notebook metadata and empty execution outputs
  remain intact. Original notebook, paper, master specification, AGENTS.md,
  existing runner and baseline report remain unchanged.

## Final delivery — the thirteen requested points

1. **Files.** Modified `Static_K_GRPO_GSM8K.ipynb`,
   `Static_K_GRPO_HotpotQA.ipynb`, `STATIC_K_EXPERIMENT_GUIDE.md`,
   `docs/IMPLEMENTATION_PLAN.md`, `docs/PROGRESS.md` and `docs/DECISIONS.md`.
   Added this report and `docs/audit_artifacts/phase4_verification.json`.
   Added `docs/audit_artifacts/phase4_clean_kernels.json` for executed-kernel evidence.
   The graph is updated separately; generated plot files are engineering artifacts.
   No separate research implementation module was introduced.
2. **Organization.** Each notebook has all 21 required sections, from research
   objective through experiment summary. Each contains its own complete logic and
   tests, with no import of the other notebook or existing baseline runner.
3. **Initialization.** Use the existing `autodspy-gpu` kernel, run definitions with
   `RUN_OPERATION=False`, and explicitly invoke initialization or training.
   Initialization loads pinned pretrained GPT-2 and fresh Adam state, resolves
   official dataset identity, and checks the configured Ollama model. Start Ollama
   and install `llama3.1:8b` beforehand; no model substitution is automatic.
4. **K=5 smoke.** Set `K=5`, `PRESET="smoke"`, `NUM_EPISODES=None`; rebuild
   `CONFIG=build_config()` and invoke `SESSION=run_operation(CONFIG,"train")`.
   This requests five episodes plus initial/final validation, not merely five calls.
5. **Other K.** Release the prior policy, set K to 8, 10 or 15, rebuild configuration
   and start a fresh run. `run_operation(CONFIG,"train",all_k=True)` is an explicit
   sequential option. It trains independently and never invokes final testing.
6. **Pool versus episodes.** Defaults use 200 training examples and 200 episodes.
   Each episode samples one example with replacement and attempts K candidates;
   unique encountered question IDs are counted. Default attempt budgets are
   1,000/1,600/2,000/3,000, excluding evaluation costs.
7. **Shared schedule.** Persisted subset IDs/content hashes and a complete seeded
   replacement schedule exclude K/preset from identity. Matching smoke/pilot runs
   use baseline prefixes. Custom sizes, seeds or schedule length change identity.
8. **Saving/resuming.** Safe immutable checkpoints and atomic pointers store policy,
   Adam, RNG, configuration, data identity, schedule and effective metrics. Resume
   only `latest.json` using the saved configuration. Unsaved work rolls back with
   its statistics; the append-only journal retains observed external costs.
9. **Validation/testing.** Baseline validation is at 0/50/100/150/200 with one pipeline
   per question and no updates. Highest validation EM selects the checkpoint;
   earliest episode breaks ties. Explicit final testing requires completed training
   and the selected checkpoint. HotpotQA uses labeled development, not unlabeled test.
10. **Comparisons/plots.** `compare_runs([...])` checks compatible completed artifacts;
    `plot_comparison(RUNS,save_dir=...)` displays/saves all eleven figures, plus
    HotpotQA EM/F1. Independent repeats stay distinct; repeated tests that disagree
    do not become a selected best test score.
11. **Passed verification.** Final notebook/kernels and 246 synthetic assertions;
    read-only real-artifact comparison; all 23 required plot paths; budget-coordinate
    checks; source preservation; documentation integrity. Prior official-data,
    real-GPT-2 gradient and reduced live inference checks remain separately documented
    in [Phase 2](PHASE_2_REPORT.md) and [Phase 3](PHASE_3_REPORT.md); they were not rerun
    or claimed as new Phase 4 experiments.
12. **Differences from the original.** Existing corrections retain proper evaluation
    model ownership, strict numeric/text EM, fractional judge parsing, failure logging,
    consistent provided-context HotpotQA, reproducible scheduling and durable recovery.
    Sequential replay preserves the Figure 2 sum and centered rewards. First-token
    action collisions, dropout-disabled policy behavior and finite equal-reward Adam
    steps remain. Phase 4 adds artifact checking/reporting without changing methodology.
13. **Limits.** The 6GB GPU and available host RAM constrain joint-model execution.
    CPU/remote Ollama and shorter explicit engineering budgets are documented choices.
    GPT-2 can lose early HotpotQA context at its 1024-token limit. Numeric last-answer
    extraction and semantic-judge failures remain limitations. Exact HTTP counts,
    server truncation, in-flight interrupted usage and total external wall time remain
    unavailable. Artifact paths/checkpoints must remain accessible for comparison.

## Master-requirement coverage and remaining work

- Sections 1–6: scope, independent notebooks, unchanged objective, one-prompt
  protocol, defaults and original action space are covered by audit evidence,
  preserved code and all-K mathematics/schedule tests.
- Sections 7–8: official dataset allocations and benchmark-specific evaluation
  are implemented; prior official-data checks and current leakage/metric regression
  tests support them. Exact published HotpotQA formulation remains unresolved.
- Sections 9–13: presets, independent starts, gradients, checkpoint/recovery and
  validation/test controls are covered by notebook-local checks. Live other-K
  workloads and live induced recovery remain unrun.
- Sections 14–16: requested fields, summaries, modes and all plots are implemented;
  unavailable resource measurements are explicit. Recorded-artifact plot/budget
  verification passed, without inventing absent baseline results.
- Sections 17–19: all numbered sections, hardware controls and notebook-local
  architecture are preserved. Default joint-model hardware feasibility is not established
  by the previous shorter CPU-policy HotpotQA check.
- Sections 20–23: clean-kernel tests, evidence distinctions, complete usage topics
  and all thirteen final-report points are documented here and in the guide.

Full smoke/pilot/baseline workloads, live K=8/10/15, full held-out evaluations,
default joint-model budgets and live induced interruption were **not run**.
No published-score reproduction or learning improvement is claimed. Adaptive-K
is future work. Phase 4 completes the implementation and its reported checks;
the actual research experiments remain explicitly user-triggered.

## Graph and source integrity

`graphify update .` passed with zero LLM/API use: 757 nodes, 758 edges and
64 communities. It skipped unsupported notebook/PDF content; notebook verification
uses actual code and kernel evidence. Some community names were refreshed from hubs;
no semantic extraction or paid relabeling ran. The update backed up its previous
curated graph. Documentation/link/whitespace checks and final preserved-source hashes
passed; notebook checksums are recorded in the machine-readable evidence.
