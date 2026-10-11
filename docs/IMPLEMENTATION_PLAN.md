# Phased implementation plan

Primary specification: [MASTER_SPEC.md](MASTER_SPEC.md). Current state: [PROGRESS.md](PROGRESS.md). Decisions and open questions: [DECISIONS.md](DECISIONS.md). Phase 1 evidence: [PHASE_1_AUDIT.md](PHASE_1_AUDIT.md).

Created 2026-10-11 (Asia/Dhaka). This is a plan, not evidence that research functionality exists or tests have passed.

## Execution rules and session handoff

Work on one phase at a time. The user authorized Phase 1 on 2026-10-11 after preparation documents were created. Phase 1 is complete; Phase 2 was authorized and is now complete for its reported verification scope. Phase 3 awaits instruction. After each phase, report actual changes, checks, deviations, and unresolved issues, then stop until instructed to continue. The final phase instructions refine the earlier request to proceed autonomously through all stages.

At every new phase or resumed session:

1. Read MASTER_SPEC.md, this plan, PROGRESS.md, and DECISIONS.md.
2. Read applicable AGENTS.md instructions; inspect branch, working tree, existing artifacts, and previous verification evidence.
3. Check the original notebook against the preservation checksum in PROGRESS.md.
4. For codebase inspection, use the installed graphify skill. If graphify-out/graph.json exists, query it first; use path/explain for focused relationships and the wiki index for broad navigation. The graph file was absent during documentation preparation; recheck before auditing. Do not claim graph results that were not obtained.
5. Continue from the last verified milestone. Review existing work before changing it.
6. Update the tracker and decisions with actual results. A completed coding checklist alone does not complete a phase.

Keep executable research logic in each independent notebook. Do not add implementation modules or alter DSPy_GRPO.ipynb. Do not launch expensive training or add Adaptive-K. After code changes, follow the AGENTS.md requirement to run `graphify update .` and record its outcome.

## Preparation — documentation and approval gate

- [x] Inspect branch, initial working-tree state, applicable root instructions, and existing documentation names without auditing research code.
- [x] Preserve the complete master requirements with normalized Markdown formatting.
- [x] Establish plan, progress tracker, decision log, and original-notebook checksum.
- [x] Verify document structure, relative links, and source preservation (see PROGRESS.md).
- [x] Ask permission to begin Phase 1; stop pending the user's response. The user subsequently authorized the audit.

Acceptance at the preparation boundary: four reviewable Markdown documents existed, existing work was preserved, and no research audit, notebook implementation, model execution, or training had started. This milestone passed before Phase 1 authorization.

## Phase 1 — Audit original notebook and research paper

Scope: master sections 3–8, 10–11, 18, 20–21, and the inspection stage of section 22. Read-only research inspection plus documentation; no new experiment notebooks.

Checklist:

- [x] Re-read context documents and inspect current repository state after permission arrives.
- [x] Read the original notebook's Markdown, initialization, dependency assumptions, and executable cells without triggering its training invocation cells.
- [x] Read relevant AutoDSPy.pdf methodology, Figure 2, experimental settings, appendix, and the supplied Figure 2 image. Record page/section and notebook-cell evidence.
- [x] Compare the publication, the released toy notebook, and project-specific settings explicitly; do not force full paper reproduction.
- [x] Trace GPT-2 initialization, tokenizer, model/optimizer ownership, sampling, pipeline generation, execution, answer comparison, reward, loss, gradients, training, and evaluation.
- [x] Inventory MODULES, signatures, action strings, pipeline lengths, conditioning/state transitions, and first-token scoring. Check collisions and whether sampled actions and logged differentiable probabilities agree.
- [x] Audit graph detachment, model references, parameter updates, evaluation model selection, equal-reward groups, exceptions, graph lifetime, and implicit checkpoint loading.
- [x] Resolve or document the one-prompt-per-episode versus appendix wording discrepancy using source evidence. Retain the specified one-prompt protocol regardless of ambiguity.
- [x] Verify original reward and semantic fallback behavior; identify numerical extraction failures and textual-metric requirements without adding reward shaping.
- [x] Investigate HotpotQA dataset configuration and actual context/task formulation. Mark unverified paper or released-code behavior as unknown; do not invent retrieval usage.
- [x] Identify dependency/API and hardware risks requiring later checks, including Ollama call accounting, seeds, quantization metadata, and CPU offloading.
- [x] Run only feasible, targeted audit checks (notebook JSON/cell inspection, isolated synthetic mathematics/functions, cached tokenization, and read-only environment probes). Record exactly what ran; avoid executing the whole original notebook.
- [x] Write docs/PHASE_1_AUDIT.md with evidence, verified issues, proposed correctness corrections, ambiguities, and acceptance criteria for Phase 2. Update decisions and tracker.

Acceptance:

- Evidence distinguishes publication, released implementation, project choices, and unresolved questions.
- Figure 2 sum loss and action behavior are understood well enough to implement a primary baseline; any proposed correction has a stated correctness reason and methodological impact.
- Original notebook and existing work are preserved; no new experiment notebook or expensive run exists.
- Record Phase 1 as completed, partial, or blocked according to actual verification, report, and stop.

Status: **completed as an audit**. Source inspection, targeted synthetic tests, and read-only environment checks support the findings in PHASE_1_AUDIT.md. Published episode/fallback/HotpotQA ambiguities remain explicitly documented. No live model or future-notebook integration success is claimed. Stop here until instructed to begin Phase 2.

## Phase 2 — Implement and validate the complete GSM8K notebook

Dependencies: Phase 1 evidence and instruction to proceed. Deliverable: Static_K_GRPO_GSM8K.ipynb with all 21 numbered sections from master section 17, plus relevant usage documentation. This phase includes the experiment controls and metrics necessary for a complete GSM8K workflow; Phase 4 consolidates and verifies them across both datasets.

Checklist:

- [x] Create readable, self-contained dependency/setup, configuration, hardware, dataset, action-space, policy, pipeline, reward, loss, training, checkpoint, evaluation, metrics, comparison, visualization, and reproducibility cells.
- [x] Keep editable K, seed, preset, mode, pool/episode/validation/test sizes, execution model/host, and checkpoint interval together. Preserve default GPT-2 127M, Llama 3.1 8B, Adam 1e-4, seed 42, and K in {5,8,10,15}.
- [x] Implement official GSM8K main: seeded disjoint train pool and train-split validation, fixed official-test subset, persisted IDs/indices, and access to the full test split.
- [x] Persist one replacement-sampling episode schedule shared across K; smoke/pilot use baseline prefixes. Distinguish controlled defaults from separately named smaller engineering checks.
- [x] Preserve CoT/Predict, original signature pool, and audited action selection. Apply only documented correctness corrections.
- [x] Initialize identical pretrained policy parameters and a fresh optimizer for each K; hold only the necessary model copy. Keep rollout parameters unchanged until the corresponding loss update.
- [x] Execute candidates sequentially; attempt exactly K pipelines on one selected example per episode. Preserve reward behavior and separately report canonical numeric exact match and semantic fallback.
- [x] Implement the Figure 2 summed loss, centered advantages, at most one valid optimizer update, safe equal-reward groups, gradient norms, failure logging, and bounded graph retention.
- [x] Implement initialize/train/resume/validate/test/compare controls and explicit sequential all-K execution. Setup and opening the notebook must not start training.
- [x] Implement separate dataset/K/seed/preset/run artifacts, configuration checks, RNG restoration, safe checkpoint writes, next-unfinished-episode resume, and explicit evaluation checkpoint selection.
- [x] Validate at baseline episodes 0/50/100/150/200 with one pipeline per question and no policy updates. Document validation-only best-checkpoint selection and tie breaking; final testing is a separate explicit action.
- [x] Record all required episode, resource, and evaluation metrics; unavailable values remain unavailable. Generate required Matplotlib plots only from real recorded results.
- [x] Include notebook self-checks for math, gradients, scoring, split isolation, schedules/prefixes, all K values, initialization, checkpoint restoration, resuming, model selection, logs, and plots.
- [x] Verify clean-kernel setup and synthetic paths where feasible, then attempt only a small live smoke test if available resources allow. Document skipped checks and failures accurately.
- [x] Update usage guide, phase report, decisions, tracker, and graph after code changes. Preserve the original checksum.

Acceptance: the GSM8K workflow is self-contained and appropriately verified; source integrity, unchanged objective, deterministic schedule, isolation, correct gradients, and safe resume have evidence. External-service-dependent checks are clearly labeled. Report actual results and stop at the boundary.

Status: **completed** on 2026-10-12 for GSM8K implementation and documented verification. See [PHASE_2_REPORT.md](PHASE_2_REPORT.md) for clean-kernel/synthetic checks, a real GPT-2 update check, reduced live integration, truncation failures and unrun workloads. Stop until Phase 3 is authorized.

## Phase 3 — Implement and validate the independent HotpotQA notebook

Dependencies: validated Phase 2 milestone and instruction to proceed. Deliverable: Static_K_GRPO_HotpotQA.ipynb. Duplicate necessary logic; do not import or execute the GSM8K notebook.

Checklist:

- [x] Re-read requirements/decisions and verify the existing GSM8K milestone before implementation.
- [x] Implement the same complete 21-section workflow and controls independently, preserving all cross-K and GRPO constraints.
- [x] Select and document official HotpotQA configuration with evidence. Prefer distractor if consistent with verified task formulation.
- [x] Persist IDs, contexts, split/source metadata, and available evaluation metadata. Keep gold answers and supporting facts out of model inputs.
- [x] Default to disjoint train-pool/train-validation and labeled development held-out test selections; preserve the authorized disjoint training-split alternative as a documented optional protocol with distinct artifacts and no validation overlap.
- [x] Preserve verified question/context usage and prohibit unverified external retrieval/search. Document unresolved task formulation choices.
- [x] Implement normalized textual exact match and token F1 separately from audited training reward and semantic fallback. Do not report unproduced supporting-fact predictions.
- [x] Include and verify standalone data, mathematics, policy, checkpoint, resume, evaluation, logging, and plotting self-checks, including exact-match/F1 comparison.
- [x] Check clean-kernel independence and perform only a feasible reduced live smoke test. Record actual results without benchmark claims.
- [x] Confirm GSM8K functionality remains intact, update documentation/graph, and verify original-notebook preservation.

Acceptance: standalone HotpotQA workflow verified to the same standard as GSM8K, including textual metrics, leak prevention, and labeled evaluation sources. Report actual results and stop.

Status: **completed** on 2026-10-12 within the documented verification scope. Final fresh-kernel execution, 96 synthetic assertions, official distractor data isolation/schedules, a real long-context GPT-2 update, and a one-episode reduced live checkpoint/validation/test/twelve-plot workflow passed. Source preservation is confirmed. See [PHASE_3_REPORT.md](PHASE_3_REPORT.md) for deviations and unrun workloads. Stop until Phase 4 is authorized.

## Phase 4 — Finalize management, statistics, plots, and reproducibility

Dependencies: both notebook milestones and instruction to proceed. Deliverables: consolidated STATIC_K_EXPERIMENT_GUIDE.md, final verification/report documents, and any necessary in-notebook corrections.

Checklist:

- [ ] Review both complete notebooks against every master section; reconcile protocol drift while preserving validated behavior.
- [ ] Verify all modes, custom-size controls, preset namespaces/prefixes, independent K starts, explicit all-K sequencing, and selected-checkpoint evaluation.
- [ ] Verify checkpoint metadata, compatibility rejection, RNG continuation, interruption safety, log consistency, safe overwrites, and completed-run summaries.
- [ ] Verify every required training/computational/evaluation field and all eleven plots plus HotpotQA exact-match/F1. Keep pipeline attempts distinct from stage/LLM calls and training budgets distinct from evaluation cost.
- [ ] Check validation progression and accuracy versus cumulative training pipeline evaluations using real artifact provenance; enforce comparison compatibility.
- [ ] Document endpoint/CPU offloading, seeds, runtime/quantization metadata, GPU peaks, unavailable measurements, failure behavior, and reproducibility limits.
- [ ] Run targeted regression and clean-kernel checks; perform reduced live tests only when feasible. No automatic baseline training or fabricated results.
- [ ] Complete usage guide covering every section 22 topic and final report covering all thirteen section 23 points.
- [ ] Update graph, preserve original source checksum, and finalize progress/decisions with actual evidence and unresolved limitations.

Acceptance: every master requirement is verified or explicitly identified as unresolved with evidence; both workflows remain independent; documented commands are usable; final artifact/report delivery does not claim unrun benchmarks. Stop after reporting.

## Requirement coverage and verification approach

Master sections 1–2 and 19 constrain all phases. Sections 3–6 and 11 require Phase 1 evidence and synthetic correctness checks in Phases 2–3. Sections 7 and 8 define dataset-specific deliverables. Sections 9–10, 12–13, and 16 define run controls and reproducibility checks. Sections 14–15 and 18 define measurement and plotting checks. Section 17 defines notebook organization. Sections 20–23 define verification, integrity, documentation, and final reporting. The trailing incremental and permission instructions govern phase transitions.

Verification layers:

1. Documentation/source integrity and evidence-backed audit.
2. Notebook format, syntax, fresh-kernel initialization, and self-contained definitions.
3. Small synthetic loss/gradient, action probability, metric, schedule, checkpoint, and interruption tests within the notebooks.
4. Real dataset split/identity checks and installed dependency/service probes.
5. Feasible explicitly invoked reduced live inference/training checks, separately labeled from the controlled research presets.

Each layer records passed, failed, skipped, or not run with reasons. Do not imply that synthetic checks validate Llama behavior, measured runtime, or benchmark performance.
