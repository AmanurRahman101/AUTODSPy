# Phase 1 audit artifacts

These files support the read-only [Phase 1 audit](../PHASE_1_AUDIT.md). They are **not experimental results**.

- `page_4.png`: rendered AutoDSPy.pdf page 4, containing Figure 2 and its caption.
- `page_5.png`: rendered page 5, containing methodology and experimental setup.
- `page_14.png`: rendered page 14, containing implementation details, Table 3, and Algorithm 1.
- `page_15.png`: rendered page 15, containing computational claims and Appendix A.7's episode/fallback descriptions.
- `synthetic_evaluation_fixture.json`: an empty array written by an isolated original evaluation-call fixture with an empty dataset and fake value-head argument. It contains no predictions, scores, or benchmark measurements.

Deletion of temporary audit files was rejected by a tool policy check without a detailed reason. The files were instead moved here using a reversible operation, with their provenance retained. No source or existing user artifact was replaced.
# Phase 2 verification additions

`phase2_clean_kernel.json` records fresh-kernel verification of the final notebook source and its checksum. `phase2_synthetic_checks.json` lists notebook-local checks; these are not benchmark scores. `phase2_full_model_gradient.json` records a synthetic-group update on real pretrained GPT-2 CUDA, separate from research results.

`phase2_live_check.json` is the original reduced-live driver snapshot. The driver completed before the final standalone-validation accounting fix: its snapshot counted 16 logical calls, while the reconciled run summary correctly counts 18 including that extra validation. The final clean-kernel artifact rechecked the actual saved run and all eleven plots. No hidden successful baseline or learning claim is implied. See [PHASE_2_REPORT.md](../PHASE_2_REPORT.md) for truncation failures, controls and limitations.

# Phase 3 verification additions

- `phase3_synthetic_checks.json`: final checksum-matched clean GPU-kernel execution
  and 96 synthetic assertions (87 distinct labels), without Llama inference.
- `phase3_official_data.json`: official labeled distractor splits, 200/50/200
  isolation and persisted cross-K/preset schedule; no model initialization/inference.
- `phase3_real_gpt2_update.json`: real pretrained GPT-2 CUDA update with a synthetic
  long-context group and nonzero gradient; not a benchmark run.
- `phase3_live_check.json`: separately namespaced one-episode live engineering check,
  explicit smaller sizes and CPU/256-token policy budget, final summary, validation,
  selected-checkpoint test, twelve plots and verified post-check Ollama unload.
- `phase3_final_verification.json`: final notebook/evidence checksum agreement,
  preserved source/master hashes, real plot files, reconciled live summary,
  documentation checks and AST graph-update scope.

No fabricated result or completed full baseline is present. The live group had
equal rewards and zero current gradient. See [PHASE_3_REPORT.md](../PHASE_3_REPORT.md)
for observed metrics, hardware/input deviations, cleanup correction and unrun paths.
