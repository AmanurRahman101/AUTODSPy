# Decisions, assumptions, and open research questions

Created: 2026-10-11 (Asia/Dhaka).

Authority: [MASTER_SPEC.md](MASTER_SPEC.md). Status: [PROGRESS.md](PROGRESS.md). Evidence: [PHASE_1_AUDIT.md](PHASE_1_AUDIT.md). This log distinguishes user-mandated protocol from implementation proposals and verified or unresolved scientific questions. D001–D009 record preparation decisions; the later entries record Phase 1 audit conclusions.

## D001 — Prepare documents, then request Phase 1 permission

Status: adopted workflow decision, directly required by the user.

The user's final instruction says: “Ask permission to run Phase 1 after creating your necessary documents.” Documentation preparation therefore precedes the audit. Earlier autonomy instructions authorize routine decisions within an approved phase; the later one-phase-at-a-time instruction requires stopping after each phase and awaiting instruction.

Consequence: do not audit the paper or original research code, implement experiment notebooks, or run model experiments before the initial permission. Do not proceed from a completed audit into Phase 2 without instruction.

## D002 — Store persistent context in docs/

Status: adopted organizational decision within the user's authorization.

Use docs/MASTER_SPEC.md, docs/IMPLEMENTATION_PLAN.md, docs/PROGRESS.md, and docs/DECISIONS.md. Keep the eventual STATIC_K_EXPERIMENT_GUIDE.md at repository root as suggested. Add phase reports only when there is actual audit or validation evidence to report.

MASTER_SPEC.md preserves the full requirements with formatting cleanup, relative links appropriate to docs/, and configuration tables expressed as lists with all values retained. It is authoritative over derived documents. Planning documents are not executable implementation files.

## D003 — Preserve the original and existing working-tree changes

Status: mandatory invariant.

DSPy_GRPO.ipynb remains unchanged. Track its initial SHA-256 in PROGRESS.md. Do not edit, delete, stage, revert, or overwrite the pre-existing modified AGENTS.md or untracked AutoDSPy.pdf. Existing baseline scripts and tests are unaudited and must not become hidden dependencies of new notebooks.

No commit, branch switch, or cleanup is required for documentation preparation.

## D004 — Figure 2 objective and episode protocol are fixed

Status: mandated project protocol; paper/released-code agreement pending audit.

Use group mean rewards, advantages `r_i - mean(r)`, and the negative sum of action log probabilities times the corresponding candidate advantage over all actions and all K candidates. No K normalization, reward standard-deviation normalization, clipping, entropy, GAE, KL, additional shaping, or Adaptive-K.

One episode samples one training pair with replacement, attempts K pipelines, and performs at most one valid optimizer update. The default pool and episode count are independently configurable 200 values; neither implies all questions are seen. Count actual unique selected IDs and record gradient norms because the summed loss scales with K.

Audit source claims before describing this protocol as reproduced publication behavior. User descriptions are requirements and hypotheses, not verified evidence of the original implementation's correctness.

## D005 — Independent notebooks and model initialization

Status: mandatory architecture and experiment constraints.

Each notebook owns its complete workflow and must run from a fresh kernel without the other notebook or separate implementation modules. Duplicate necessary code rather than introduce shared imports. Retain responsibilities of the original named functions and the original CoT/Predict action space/signature pool.

Each K starts with identical pretrained GPT-2 parameters and a fresh Adam optimizer. No automatic checkpoint loading, cross-K warm starts, or silent execution-model substitution. Evaluation must explicitly use the selected policy checkpoint.

Any action-scoring correction is undecided until the audit establishes a genuine correctness failure and records its methodological impact.

## D006 — Configurable defaults and reduced engineering tests

Status: mandated distinction; implementation details pending later phases.

Default presets use a shared 200-example pool with smoke/pilot/baseline episode counts 5/20/200. Persist a replacement schedule shared across K and use prefixes for shorter presets. Use distinct run/checkpoint namespaces.

The user additionally authorizes smaller pool/episode/validation/test sizes for implementation checks. Label those as engineering checks with their actual configuration; do not conflate them with controlled baseline presets or published scores. Keep all requested size, model, and endpoint controls easy to edit.

No expensive baseline is authorized for automatic execution.

## D007 — Benchmark splits and unresolved HotpotQA formulation

Status: mandated allocation defaults with audit questions outstanding.

GSM8K uses official main training data for disjoint policy-training and validation subsets, and the official test split for the fixed preliminary held-out subset.

HotpotQA defaults to disjoint training/validation subsets from training and a publicly labeled development subset for held-out testing. The user also permits a disjoint training-split test alternative. Any such alternative must remain disjoint from validation as well as training, be clearly labeled, and use separate protocol metadata; the illustrative 201–400 range is not permission to overlap validation examples. Do not score unlabeled official test data.

Do not decide context formatting, retrieval, or publication consistency before auditing evidence. Preserve question IDs/context/evaluation metadata while excluding gold answers/supporting facts from model inputs.

## D008 — Reward, benchmark metrics, and model failures stay distinct

Status: mandated reporting separation; mechanics pending audit.

Preserve the released training reward as closely as required, including verified semantic fallback; report numerical/textual exact match and HotpotQA token F1 separately. Numeric normalization must support integers, negatives, decimals, and comma-separated numbers. Do not turn F1 or extra shaping into training reward without authority.

Future implementation must define/document failure rewards consistently, maintain exactly K attempted candidates and the group baseline, and log failures. A specific failure policy is not yet chosen. Missing calls/tokens/memory measurements must be marked unavailable.

## D009 — Evidence and completion criteria

Status: adopted reporting convention.

Mark checks passed, failed, skipped, or not run based on actual execution. Preserve paper page/section, notebook cell/function, and installed-version evidence where applicable. Do not claim an audit, complete phase, smoke success, benchmark score, or hardware measurement merely because a plan or code exists.

For later phases, synthetic tests check mathematics and control flow; only actual external-service checks can establish dataset/Ollama integration. Full paper benchmark reproduction is outside the reduced-budget research claim.

## Open questions for Phase 1

- What do Figure 2 and the appendix actually say about samples per episode, and where do they disagree?
- Does the original notebook's loss, gradient path, action likelihood, and model selection match its intended behavior?
- Which action strings collide under first-token scoring, and does preserving that approximation still yield valid sampled-action likelihoods?
- What are the exact original signatures, pipeline construction, reward, answer extraction, and semantic fallback?
- Which settings are published, which belong to the released toy setup, and which are project-specific?
- Is HotpotQA distractor/context-only usage verified, and was any retrieval used in the target experiment?
- Which dependency/service probes and lightweight checks are feasible without changing the user's environment or triggering original training cells?
- Which Ollama call/token/quantization/runtime and GPU-memory measurements can be obtained reliably on the target hardware?

Do not silently resolve these by assumption. Phase 1 findings and remaining uncertainty are now recorded in PHASE_1_AUDIT.md. The following entries refine earlier pending questions without altering MASTER_SPEC.md.

## D010 — Preserve the first-token constrained categorical policy

Status: adopted for the future primary baseline, supported by Phase 1 tests.

The original sampler's selected-entry probabilities and stored log probabilities agree. All action strings tokenize nonempty; 15 signatures have nine distinct first-token IDs. Collision groups cannot learn within-group preferences, and their multiplicities influence aggregate probability. These are expressiveness limitations of the released approximation, not a demonstrated sampled-action likelihood error.

Keep the full signature list and duplicate entries. Do not introduce full-string scores, new action heads, or additional modules. A stable log-softmax implementation may preserve the same categorical mathematics without probability clamping. Retain deterministic stop and its zero log probability.

## D011 — Distinguish zero gradients from Adam steps

Status: adopted baseline-preservation decision, supported by original code and a synthetic momentum check.

The original performs Adam.step even for finite equal-reward groups. After previous nonzero gradients, Adam momentum can move weights despite zero current gradient. Preserve this behavior in the primary baseline; count finite optimizer steps and zero-advantage groups separately. Do not call such groups guaranteed no-update episodes. Nonfinite/invalid groups must not update.

Skipping finite equal-reward steps would require explicit deviation documentation; it is not a behaviorally equivalent memory optimization.

## D012 — Replay must preserve probabilities and model mode

Status: design constraint established by the audit; implementation and validation pending Phase 2.

The original retains policy graphs for all K trajectories until one backward pass. Recomputing differentiable probabilities can reduce this activation cost without changing the sum objective, provided parameters do not change before the episode's one optimizer step, and state construction, valid-action ordering, and mode remain consistent.

Installed Transformers sets pretrained models to evaluation mode by default; original code never explicitly enables training-mode dropout. Keep dropout-disabled behavior explicit while enabling autograd. Do not silently introduce stochastic dropout when adding a training helper. Verify gradients and replay probabilities with notebook-local checks.

## D013 — Correct answer comparison while preserving judge reward semantics

Status: corrections required in later notebooks, with documented methodological impact.

Original comparison mishandles commas, equivalent decimals, and ordinary textual ground truth; evaluation substring and loose judge parsing produce false positives. Implement requested benchmark-specific normalization and strict metrics. Record that these fixes change which outputs receive exact reward 1 or go to semantic fallback.

The original judge returns bounded fractional rewards; keep that behavior rather than silently binarize it. Reject nonfinite values and log failures. Do not add the format bonus mentioned only in the docstring or the embedding-based evaluator described inconsistently in Appendix A.7.

Figure 2/section 4.1/Appendix A.1 LLM fallback and Appendix A.7 embedding fallback are distinct published descriptions. The released LLM judge is the primary-baseline reference. Exact-match/F1 evaluation remains separate from reward, and checkpoint selection uses validation exact match with a documented tie break.

## D014 — HotpotQA configuration is a project protocol choice

Status: Phase 3 proposal to finalize before implementation; exact paper task formulation remains unverified.

Released code loads fullwiki training data without context and later evaluates with raw provided context. It does not implement retrieval, despite the query-related docstring. The inspected AutoDSPy paper does not specify fullwiki/distractor or precise prompt construction.

Prefer the master-specified distractor provided-context setup for a consistent future train/evaluation task; label it a project choice and an input-consistency correction rather than exact paper reproduction. Include all provided context without gold supporting-fact filtering, retain metadata separately, and do not add external search. Document any context/token budget because GPT-2 is limited to 1,024 tokens; preserve the pipeline-state suffix if formatting longer policy inputs.

## D015 — Existing runner/report are preserved but excluded from baseline authority

Status: adopted after inspection.

run_grpo_baseline.py cycles prompts, standard-deviation-normalizes advantages, and clips gradients. These differ from the original notebook and master requirements. Do not import the runner or reuse its reported statistics as this project's results. Preserve all existing files.

## D016 — Environment readiness is narrower than integration success

Status: observation and implementation prerequisite.

The default Python uses CPU-only PyTorch. Existing venv_gpu detects CUDA; use the intended GPU kernel for future hardware checks, with versions recorded. Ollama 0.33.2 responds and the requested Llama 3.1 8B Q4_K_M tag is installed, but no execution model was loaded or queried for generation.

Keep model identity fixed, record digest/quantization, and provide explicit offloading/endpoint controls rather than substituting another model. No runtime/VRAM/token estimate is a measurement until a corresponding live check runs. Caching and retries must be recorded when distinguishing pipeline attempts from actual LM requests.

## Phase boundary after the audit

Phase 1 was authorized and completed on 2026-10-11. Stop after reporting. Phase 2 remains unstarted until the user instructs it. The original notebook, paper, existing runner/report, and user-modified AGENTS.md remain preserved.

## D017 — Phase 2 authorization and boundary

Status: implemented and verified, 2026-10-12. The user's subsequent instruction authorized Phase 2. Create only the independent GSM8K notebook and its necessary controls/statistics/documentation; stop after its verification. HotpotQA and Adaptive-K remain unimplemented. PHASE_2_REPORT.md records actual evidence and unrun workloads; implementation completion does not imply benchmark learning.

## D018 — Immutable identity and explicit operation controls

Status: implemented. Pin GPT-2 to revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; resolve GSM8K main to an immutable commit before loading. Persist subset indices/IDs/content hashes and the full replacement schedule independently of K/preset. Use independent deterministic streams derived from SEED: base seed for train permutation, seed+1 for test, seed+2 for episodes. Changes to sizes/revision/schedule length form a different namespace.

Use unique dataset/K/seed/preset/run directories; require explicit checkpoint selection and compatible controls/model/data/dependency/Ollama versions. No implicit warm start or automatic operation runs from setup. Validation uses fixed split-specific multinomial RNG and restores training RNG. Best numeric EM wins; earliest episode breaks ties. Final test is explicit after training and cannot select checkpoints.

## D019 — Memory-bounded equivalent update

Status: verified. Keep pretrained evaluation mode with autograd enabled. Sample without graphs, then sequentially replay/backpropagate each candidate while weights remain unchanged. Assert replay/rollout log probabilities agree; make one Adam step after accumulating the Figure 2 sum. Tests show equivalence to retained-graph gradients at every K. No K scaling correction or clipping is added.

Use Adam foreach=False to reduce temporary optimizer memory while retaining its mathematical update/default betas/epsilon. The real GPT-2 CUDA update check passed with a synthetic group and changed parameters. Clear any selected-checkpoint label after a policy update so later evaluation cannot misleadingly label modified weights as the earlier checkpoint.

## D020 — Durable periodic recovery and resource journal

Status: verified with meaningful nonzero synthetic training. Immutable checkpoint files and atomic latest/best pointers prevent accidental overwrite. Restore policy/optimizer/RNG/schedule/metrics together. Resume only the latest durable checkpoint; reconcile statistical logs and best pointer to restored state. Interrupted recovery matches uninterrupted final weights with no duplicate effective updates/rows.

Periodic checkpointing cannot preserve unsaved optimizer work. Roll it back together with corresponding statistics, while retaining an append-only external-attempt journal. Resume begins at the next durably committed episode. This limitation is explicit; interval 1 minimizes rollback at high disk-I/O cost. Mid-episode interruptions cannot save potentially partial optimizer state. Completed pilot runs cannot silently become baseline runs.

Final review added recovery for interrupted intermediate/final validation: finish that pending evaluation before advancing and require durable final validation for completion. Tests confirm no extra policy updates. Evaluation-only sessions cannot train or save a checkpoint with optimizer state from another episode; continuing training requires explicit resume.

## D021 — Separate metrics, failures and measured call scopes

Status: implemented. Numeric normalization/finite judge parsing correct the audited scoring errors; last-number extraction remains a known limitation. Preserve fractional Llama semantic reward, exact reward 1 and zero execution/judge-failure reward. Failures can bias groups and must be inspected. Strict numeric EM is the selection metric; semantic means/full-credit counts are separate.

Keep the original DSPy LM provider with cache off, fixed seed/temperature, bounded context/output, zero provider retries and configurable Ollama GPU layers. These are explicit project decoding/resource controls; DSPy adapter retries may still occur. Logical LM.forward attempts and returned usage are instrumented; physical HTTP request counts remain unavailable. Standalone validation/full-test costs join recorded evaluation totals. Journal resources include discarded attempts where instrumented; older engineering journals are marked incomplete rather than retrospectively populated with estimates. PyTorch allocator peaks and Ollama residency are separate.

API support was checked against the installed source and primary documentation: [DSPy LM](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/lm.py), [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md). Version/model fingerprints are persisted and compatibility checked.

## D022 — Reduced live check is evidence of integration, not learning

Status: observed. Retain default research models and budgets in configuration. A separately namespaced live engineering check used pool 2, one episode, K=5, validation/test 1, output 128/context 2048 and 12 Ollama GPU layers because free RAM was limited. Two of five training pipelines returned outputs; three failed after truncation. Rewards and current gradient were zero. Checkpoint/evaluation/plot paths completed, but this run does not demonstrate learning or benchmark performance.

Real full-GPT-2 nonzero-gradient/update correctness was separately verified with a synthetic group. Default CPU-only Llama generation, larger context/output budgets, full presets, live other-K runs and live induced recovery remain unrun. No smaller research model or changed loss was used. The execution model was unloaded afterwards to release resources. Exact results and artifacts are in PHASE_2_REPORT.md.

## D023 — Phase 3 HotpotQA task and labeled evaluation

Status: implemented and verified within the Phase 3 report's scope. The user authorized the next phase.
Adopt D014's explicitly proposed project task: official
[HotpotQA distractor](https://huggingface.co/datasets/hotpotqa/hotpot_qa), all provided
paragraphs in original order, identical input formatting for training and evaluation,
no retrieval. The released notebook's training omission of context is corrected;
the exact published fullwiki/distractor formulation remains unknown. This is not
an exact paper reproduction claim.

Default final held-out examples use labeled development; the authorized optional
training-split holdout is disjoint from training and validation and changes manifest
identity. Preserve IDs, context and gold metadata as dataset artifacts. Input formatting
accepts only question/context, never gold answers or supporting-fact annotations.
Full evaluation explicitly uses labeled development, without fabricated test labels.

## D024 — HotpotQA answer metrics and context limits

Status: implemented and verified within the Phase 3 report's scope. Use the
[official answer evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
normalization, multiset token F1 and special yes/no/noanswer mismatch rule. Extract
the final bracketed answer, otherwise use the whole response. F1/precision/recall
are evaluation metrics only. Training reward retains exact one or the original
fractional Llama judge; validation answer EM selects the checkpoint, earliest tie.
No supporting-fact metrics without predictions.

Keep the original first-token sampler, Figure 2 sum and sequential replay unchanged.
GPT-2's 1024-token limit can drop early context; question/action suffix is retained
by left truncation. Expose MAX_POLICY_TOKENS and log per-state token counts/budget/
truncation flags. Shorter hardware budgets are explicit input deviations. Llama
receives all formatted context within its server budget; actual server truncation
is unmeasured. All budgets must match across compared K runs.

## D025 — Separate long-context correctness from reduced live integration

Status: observed and documented. Real GPT-2 CUDA replay/backward at the default
1024-token budget passed with nonzero gradients and changed parameters, reserving
about 5.36 GiB. The live joint-model check explicitly used CPU policy training,
256 policy tokens and eight Llama GPU layers because local GPU/RAM were constrained.
It retained both research models, the sampler and Figure 2 objective. This input/device
deviation belongs only to a namespaced engineering run, not the default baseline.

All five live candidates succeeded and received equal rewards one, producing zero
current gradient. The one-question validation had EM zero/F1 0.4 but semantic
reward one. This demonstrates why reward and benchmark metrics must remain separate.
The EM tie selected episode zero as documented. One held-out correct example is
integration evidence, not benchmark performance or learning.

Checkpoint restoration, final testing and twelve plots passed; full workloads,
live other K values, default joint GPU/context operation and live induced recovery
remain unrun. Synthetic interrupted recovery has nonzero updates. The driver-only
cleanup argument mistake was corrected by a separate unload request; no loaded
execution model remained. Detailed evidence is in PHASE_3_REPORT.md.

## D026 ? Phase 4 authorization and completion boundary

Status: implemented and verified on 2026-10-12. The user's ?implement the next phase?
instruction authorizes final management/statistics/plot/reproducibility consolidation.
Both notebooks remain independent. Core experimental code is preserved; no full
training, new live inference, or Adaptive-K is implied. Stop after the final report.

## D027 ? Verify recorded artifacts before comparison

Status: verified. Completed summaries alone are insufficient for controlled K
comparisons. Check configuration/model snapshots, manifest hash/schedule/IDs,
complete training/validation progression, centered statistics, candidate budgets,
latest/best checkpoint pointers and held-out checkpoint/question provenance.
Reject duplicate paths, missing metadata and incompatible protocols. Separate
repeats remain separate points; no test-score selection is introduced.

Keep original artifact paths/checkpoints accessible; automatic relocation is not
implemented. Comparison reads evidence without rewriting historical summaries.
Final plots use prior live engineering artifacts, not fabricated or baseline results.

## D028 ? Aggregate only measured resources within explicit scopes

Status: verified. Summaries add recorded training GPU peaks, failures and LM time,
evaluation success/failure/time, and observed journal calls/tokens/time including
rollback. Missing values and incomplete old journals remain unavailable. Journal
entries cover completed recorded attempts; in-flight interrupted usage may be lost.
PyTorch allocator peaks, Ollama residency, LM-forward attempts and physical HTTP
requests retain their separate meanings. No total external wall time is invented.

Fresh-kernel synthetic regression passed 111 GSM8K and 135 HotpotQA assertions.
Read-only previous-live artifact checks and eleven/twelve real-artifact plots passed.
No new external-model integration or benchmark learning claim follows from Phase 4.
