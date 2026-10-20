# AdaGRPO GSM8K implementation and verification

Implemented on branch `amanur` from the approved 20 October 2026 plan. The experiment notebook is [AdaGRPO_GSM8K.ipynb](../AdaGRPO_GSM8K.ipynb). The original notebooks and historical experiments are preserved.

## Implemented behavior

The self-contained notebook supports automatic sequential allocation and matched static K=2/4/5 controls. It keeps pretrained GPT-2, the released one-module/signature/stop action grammar, first-token collisions, the summed Figure 2 objective, and one finite Adam step per completed episode. Learning rate is 1e-5. All new arms use deterministic numeric exact reward, with parser provenance and no semantic calls. Old semantic-reward results are not compatible controls.

The controller probes two candidates, expands by two, and uses bounds 2..15 under a gross run budget of four times the episode count. It reserves minimum future groups and stops on noncancelled effective-class credit, four all-correct outcomes, sufficiently concentrated policy support, or the remaining allowance. Effective-class probability and centered-credit calculations account for signature collisions. All completed-group candidates are replayed, without rejection or caching.

Actual-K accounting reaches updates, validation provenance, summaries, artifact audits, plots and checkpoint recovery. Journals distinguish starts and finishes, including incomplete external work. Infrastructure failures abort a partial group without an optimizer update. Discarded work remains charged to the budget; insufficient budget leaves an incomplete run.

Checkpoint `.pt` files are ignored by Git; copy the complete run directory including these files when preserving a run. Latest full-state and best-policy checkpoints use relative references, SHA-256 verification, portable replacement, and rotation restricted to new run directories. Latest durable state repairs an interrupted best/latest publication. Controller decisions can be independently reconstructed from saved probabilities, token IDs, rewards, and gross attempt order. Runtime changes cannot silently resume the same controlled experiment.

The notebook includes an opt-in formal study, 27,828-pipeline workload estimator, full-test guard, plots and paired hierarchical bootstrap acceptance analysis. Final test loading is deferred until an explicitly requested test of a completed formal run. Acceptance audits source artifacts and requires all four arms at seeds 42/43/44. No Colab credits or formal training were used during implementation.

## Local environment

A separate `venv_adagrpo` Python 3.11.9 environment and `AdaGRPO GPU` Jupyter kernel are installed. Core versions: PyTorch 2.6.0+cu124, Transformers 5.17.0, DSPy 3.3.1, datasets 5.0.1, NumPy 2.4.6, Matplotlib 3.11.2. The full version snapshot is [adagrpo_environment.txt](audit_artifacts/adagrpo_environment.txt).

Hardware verified: RTX 3050 Laptop 6GB, 16GB physical RAM, i5-13420H. Llama 3.2 3B Q4_K_M was explicitly installed through Ollama. The live server reports zero model VRAM under CPU offload. GPT-2 uses CUDA FP32. Context is 2048, maximum output 512, generation temperature zero. No model substitution occurred.

## Verification evidence

- [Engineering checks](audit_artifacts/adagrpo_self_checks.json): 93 passing checks, including loss/gradient/Adam equivalence, zero-gradient momentum, collision cancellation, odd group limits, budget reservation, mocked end-to-end runs, interrupted resume equality, staged checkpoint recovery, truncated-journal-tail repair, failure accounting, tamper detection, plots, numerical bootstrap cases, and explicit opt-in guards for cloud smoke and formal training. These fixtures are not benchmark results.
- [Fresh-kernel definitions](audit_artifacts/adagrpo_clean_kernel.json): 19 code cells executed successfully without initiating model/dataset/training/test operations.
- [Preservation manifest](audit_artifacts/adagrpo_preservation.json): SHA-256 verification of all 121 original root notebooks and pre-existing results files; no change.
- [Live smoke](audit_artifacts/adagrpo_live_smoke.json): two episodes per method completed. Static5 used 10 training attempts; adaptive used 8, both with best validation accuracy 0.5 on only two questions. Adaptive K was 4 then 4, stopping for all-correct and useful-contrast reasons. Both arms had a measurable current gradient in at least one episode. No final testing occurred. This demonstrates integration, not a research accuracy result.
- [Artifact audits](audit_artifacts/adagrpo_artifact_audits.json): `verify_run_artifacts` passes on all five recorded runs — both smoke runs, the static pilot, the completed adaptive pilot including its user-approved resumed segment, and the live recovery run.
- [Live recovery](audit_artifacts/adagrpo_live_recovery.json): a real CUDA run was interrupted immediately after its durable episode-1 checkpoint and resumed. Optimizer state contained exactly step {2} after two completed episodes, committed attempts stayed 8/8 with none discarded, gradients remained finite, and no final-test access occurred; independently re-verified from a fresh process. Note recorded: the original launcher process exited 1 after the success record with no traceback, so the evidence was confirmed by independent recheck.
- [Pilot diagnostics](audit_artifacts/adagrpo_pilot_diagnostics.json): reward, parsing-provenance, controller, concentration, and runtime accounting reconciled from pilot journals; committed attempts equal journaled start/finish pairs in both arms.

## Local pilot

The matched 20-episode pilot ran under one shared 60-minute limit. Static-K=5 completed as an independent run: 100/100 training attempts, extraction provenance 72 last-number / 28 bracket with zero parse or candidate failures, best validation **0.95 at episode 10** (checkpoints 0.80 → 0.95 → 0.85), peak policy VRAM ~3.08 GB, 13/20 zero-advantage groups. AdaGRPO was stopped by the shared suite clock 1,153 seconds into its own session with 10 of 20 episodes durable and 36 committed training attempts; the reconciled journal reports zero discarded training attempts. No further episode was committed before the stop. The original suite gate therefore records `incomplete`.

With explicit user approval (D030), the adaptive arm then resumed from its durable episode-10 checkpoint as a named second wall segment. It completed all 20 episodes inside the fresh 3,600-second segment deadline, charged attempts from the journal, beat no checkpoint it had not already reached, and now passes the full completed-run artifact verifier. Final Adaptive facts: **80/80 attempts used with zero discarded**, stops distributed 11 useful-contrast / 5 all-correct / 4 allowance-reached, K chosen automatically across 2, 4, 5, and once the full 15 ceiling (episode 11), policy VRAM ~3.10 GB, 9/20 zero-advantage groups, 11/20 informative gradients, no parse failures, no journal errors, no test access.

**Screening verdict: the pilot is not promising as-is.** Compute met the target exactly — 80 versus 100 matched-window attempts is precisely 20% fewer. Accuracy did not survive: Adaptive's best validation stayed **0.80 at episode 0** across all checkpoints (0.80 → 0.80 → 0.80), versus Static-K=5's 0.95 — a 15-point shortfall, three times the 5-point screening allowance. The diagnostics identify two observations worth investigating; they do not establish the cause of the accuracy gap: (1) policy concentration — the effective class count fell from neff 3–8 during episodes 1–10 to about 2.1–2.9 in most later episodes (with episode 16 at 5.19); (2) budget drain on a question for which all fifteen sampled pipelines failed — episode 11 spent fifteen attempts on an all-failure group with zero gradient, consuming 18.75% of the whole run budget on no credit. Episodes 17–19 were forced into allowance-reached two-candidate groups by the reservation math after that drain. Neither arm accessed the final test set.

Per the standing rule, this failure pattern must be resolved or acknowledged before any Colab spend. The current single-seed, 20-question validation pilot is screening evidence only. These observations motivate controlled follow-up; they do not prove the controller caused the gap or that the question is unsolvable. A matched Static-K=4 control would help distinguish adaptive allocation from simply reducing the budget. Details: [pilot diagnostics](audit_artifacts/adagrpo_pilot_diagnostics.json), [resume record](audit_artifacts/adagrpo_adaptive_resume.json), [artifact audits](audit_artifacts/adagrpo_artifact_audits.json), [suite record](../results/adagrpo/pilot_suite_b7c48a6c.json).

## How to use

Open the notebook with the `AdaGRPO GPU` kernel and execute the definition cells. The operation switch defaults to false. Call `adagrpo_self_checks()` before an intentional operation.

- `run_operation(CONFIG, "train")`: train the selected local preset explicitly; formal training is guarded.
- `run_local_suite("smoke")`: bounded two-episode integration checks for both methods.
- `run_local_suite("pilot")`: 20 episodes per method, 50 training/20 validation questions, shared one-hour deadline, validation-only screening gate.
- `load_run(run_directory)`: restore latest durable training state; incomplete external work stays charged.
- `load_run(run_directory, for_resume=False)`: load the best validation policy. `test_model(session)` additionally requires completed formal training and refuses repeated final-test artifacts.
- `compare_runs(run_directories)`, `plot_comparison(runs)`, `acceptance_report(runs)`: audit before analysis; formal acceptance requires all predefined arms/seeds and explicit full tests.
- The Colab section describes a fresh cloud smoke, durable storage and an explicit `run_formal_study(..., allow_formal_training=True)` operation. Confirm measured runtime and credit budget before using it.

## Research limitations

Adaptive stopping changes the estimator despite preserving the loss expression. K also changes the scale of the summed gradient. The effective-class diagnostic is a proxy for gradient informativeness. Last-number extraction can mistake a reasoning number for the final answer. Fixed seeds cannot guarantee identical external outputs across hardware. CPU inference may prevent a complete local pilot within the time limit.

A capped budget does not prove successful allocation, and a lower zero-advantage fraction does not prove better learning. Formal evidence must compare held-out accuracy and gross cost against static5 and inexpensive static2/static4, with uncertainty reported. If a cheaper static arm performs as well, report that result.

## Knowledge graph

Ran the required `graphify update .` without semantic API calls. The installed extractor indexes code and document headings but does not classify `.ipynb`; notebook functions are therefore inspected directly rather than falsely claimed as graph-indexed. The updated graph contains the AdaGRPO implementation/decision documentation. Previous tracked AST cache entries were preserved.
