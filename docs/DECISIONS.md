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
