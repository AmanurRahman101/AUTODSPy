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


## D029 - AdaGRPO GSM8K implementation authorization and protocol

The user explicitly approved implementation of the AdaGRPO plan on `amanur` on 20 October 2026. This supersedes the earlier Static-K-only milestone boundary for the new notebook only. Original notebooks and historical results remain preserved.

`AdaGRPO_GSM8K.ipynb` is self-contained and includes adaptive and matched static modes. It keeps the original Figure 2 sum, empirical mean-centered advantages, one finite Adam step per completed group, first-token scoring and signature collisions, and sequential differentiable replay. Default LR is 1e-5. No clipping, normalization, entropy, imitation, Bayesian baseline, or advantage weights are added.

The new reward is binary numeric exact using the audited extractor, with bracket/hash/last-number provenance. No semantic judge is called. Historical semantic-reward experiments are incompatible controls and require matched numeric reruns.

The sequential controller probes two candidates, expands by two, uses bounds 2..15, reserves two attempts for future episodes, and has a gross training-attempt ceiling of four times the episode count. It stops on noncancelled effective-class credit contrast, four all-correct results, four identical effective classes with exact policy Neff <= 1.25, or the allowance. Effective classes aggregate colliding module/signature first-token paths. Every completed-group candidate is replayed. This contrast diagnostic does not guarantee a nonzero gradient. Outcome-dependent stopping changes the estimator despite the unchanged loss expression; no inherited unbiasedness or performance guarantee is claimed.

Started attempts are durably journaled before execution. Failed/rolled-back work remains charged. Service/OOM/timeout failures abort the partial group without an optimizer step; candidate-program failures receive zero. Latest full and best-policy checkpoints use portable replacement and SHA-256 pointers. Latest state resolves interrupted best/latest publication; older referenced files remain until both roles are durable. Insufficient gross budget produces an incomplete budget_exhausted run.

The local target is the verified RTX 3050 Laptop 6GB, 16GB RAM, i5-13420H. Use Python 3.11, CUDA GPT-2 FP32, Adam foreach=False, Llama 3.2 3B Q4_K_M via Ollama with CPU offload locally, context 2048 and max output 512. Do not substitute a different model. Definition cells have no external-model or training operations.

Only bounded smoke/pilot workloads are authorized automatically here. Colab/full training and final testing remain explicit operations. Formal comparison uses seeds 42/43/44, adaptive and static 2/4/5, shared membership, per-seed shared schedules, validation-only selection, and full official testing only after completed formal training. The acceptance target is >=20% fewer gross training attempts than static5 and a paired hierarchical-bootstrap 95% lower accuracy-difference bound >= -0.02. A budget cap alone or a lower zero-advantage rate does not establish a benefit. Report dominance by cheaper static controls and inconclusive uncertainty honestly.

## D030 - Local pilot wall-limit handling and user-approved adaptive resume

The 20 October 2026 local pilot implemented the plan's one-hour limit as a shared suite clock: static5 ran first and consumed 2,438 session-seconds of the 3,600-second budget, so the adaptive arm was stopped 1,153 seconds into its own session with 10 of 20 episodes durable and its partial group discarded. The suite gate records `incomplete`; partial runs were never counted as pilot success. Journal reconciliation at the stop showed 36 committed attempts against an 80-attempt ceiling, with zero parse, candidate, or journaling failures in either arm.

The user explicitly approved a deviation on the same day: resume the wall-limited adaptive arm from its durable episode-10 checkpoint as a named second wall segment using the notebook's own recovery protocol, instead of leaving the pilot incomplete or rerunning both arms. The resumed segment is a dependent continuation, not an independent protocol run: no controller, loss, reward, budget, dataset, or checkpoint behavior changes, the attempt budget remains charged from the journal, and the reported result is screening evidence labeled as resumed rather than a single-shot bounded pilot. The completed-run artifact verifier now audits the arm directly because the resume completes episodes 11-20 with a fresh 3,600-second segment deadline. No other protocol change is authorized; future suites should plan per-arm wall budgets or account for the earlier arm's consumption when sequencing arms.

The approved resume completed on the same day and passes the full completed-run artifact verifier. Its screening outcome and mechanism analysis are recorded in [implementation report](../docs/ADAGRPO_IMPLEMENTATION_REPORT.md) and [pilot diagnostics](audit_artifacts/adagrpo_pilot_diagnostics.json). The compute target was met exactly (80 versus 100 attempts) while validation accuracy did not survive the budget restriction (0.80 best versus 0.95); this is a reported screening failure, not a claim for or against the research hypothesis, and controller-design changes beyond D029 remain unauthorized until explicitly approved.


## D031 - Visible-terminal matched Static-K=4 pilot

On 20 October 2026 the user explicitly requested continued local training and testing in a terminal they can see and open. Run the missing matched Static-K=4 pilot using the existing unchanged AdaGRPO notebook: seed 42, 50 training questions, 20 episodes, 80 training attempts, and 20 validation questions at episodes 0, 10, and 20. The independent arm has the existing 3,600-second wall limit. Match the saved dataset revision, episode schedule, pretrained initialization and selected Llama model identity to the previous Static-K=5 pilot.

Training uses a persistent Codex PTY; a visible PowerShell monitor follows `results/adagrpo/static4_terminal.log`. A temporary launcher executes notebook definitions and adds progress printing only; experimental functions remain self-contained in the notebook. Record the new run directory in `results/adagrpo/static4_active_run.json`. On completion, audit artifacts and compare against the completed adaptive and static5 pilot arms. Preserve partial work if interrupted; do not interpret a partial run as success. No controller redesign, formal/cloud workload or final-test access is included in this operation.


## D032 - AdaGRPO v2 bounded improvement study

The user explicitly approved the complete improvement plan on 20 October 2026. Create `AdaGRPO_GSM8K_v2.ipynb` on amanur and preserve v1/notebooks/results. Protocol `adagrpo-gsm8k-numeric-figure2-v2` retains the original sum loss, empirical centered advantages, numeric reward, first-token grammar/collisions, LR 1e-5, and finite equal-reward Adam steps. No regularizers, rescaling, judge, rejection, uniqueness forcing or training cache are introduced.

The frozen controller probes two; two correct rewards stop at two. Other groups reach four. All-failure or class-credit cancellation stops at four. Useful contrast may refine to six, with bank=max(0,4*completed_episodes-gross_attempts) and allowance=min(6,remaining_budget-2*future_episodes,4+bank). Retain/replay all candidates. Outcome-dependent stopping changes the estimator; fixed independent-K expected summed gradient scales as K-1, not an Adam step guarantee.

Versioned per-candidate streams use seed/episode/candidate; evaluation uses seed/split/question. Partial validation is bound to policy weights, data, runtime, execution options and sampling identity. Finished journal outcomes recover without re-execution; incompatible progress is archived and gross costs retained. Before-evaluation durable checkpoints make interruption recovery possible.

Authorize one development smoke plus one matched seed-43 study: same preserved 50 training questions, twelve replacement episodes, twenty fresh validation questions excluding all prior membership, validation at 0/12, adaptive/static4/static5, ceilings 48/48/60. A single persisted 7200-second deadline covers all new local model setup, smoke, training, validation and interruptions/resume. Never reset it. The smoke deliberately reloads after a finished validation attempt to verify no repeated execution. Keep CUDA GPT-2 and CPU-offloaded Llama3.2:3b unchanged across arms. Use Codex PTY and persistent progress log; do not open another external monitor.

Correctness gates precede all model work. Screening requires complete audited arms, >=20% gross training savings vs static5, <=5-point validation gap, improvement over initialization, strictly beating static4, finite updates and variable K. Do not retune against confirmation questions or claim formal success. Cloud/full tests stay gated and explicit; five-arm formal default adds shuffled-K (clean maximum 34,935 executions), with frozen choices, 3 seeds and the paired 10,000-resample accuracy interval. Failure stops cloud progression.

## D033 - Direct matched V1 versus V2 controller comparison

On 20 October 2026 the user explicitly requested a test of whether V2 is better than V1, using matched seeds where appropriate. Compare the frozen original V1 controller and frozen V2 controller under common V2 infrastructure in `AdaGRPO_V1_V2_Comparison.ipynb`, protocol `adagrpo-gsm8k-controller-ab-v1`. This is a controller ablation, not a rerun of the original global-RNG V1 notebook. Preserve both source notebooks and all historical artifacts.

Use paired seeds 43 and 44, the existing fifty-question training pool, twelve replacement-sampled episodes per seed, the same saved twenty-question checkpoint-selection split, and validation at episodes 0 and 12 with earliest ties. Share pretrained initialization, candidate-stream-v2 streams, schedule, binary numeric rewards, original summed loss, Adam and model identity. Both adaptive controllers have a gross training ceiling of 48 attempts. Reuse the completed V2 seed-43 arm read-only after artifact, checkpoint, runtime, initialization and selection verification; all other three arms are newly trained.

Freeze twenty new train-split confirmation questions disjoint from every recorded prior training/validation membership before model work. Evaluate each validation-selected best policy exactly once on this common confirmation set. Do not tune controllers on these outcomes or access the official test split. Report paired per-seed outcomes, training cost and descriptive hierarchical-bootstrap uncertainty; two seeds and twenty questions cannot establish general superiority or formal success.

One new persisted 7,200-second deadline covers all new model setup, training, selection evaluation, confirmation, failures and resumes. It is separate from the completed D032 suite and cannot reset automatically. Maximum clean new work is 344 pipeline executions. Use a Codex PTY with a persistent progress log and request the integrated terminal panel; if attachment remains queued, provide the integrated-terminal tail command without opening an external window. No cloud launch, model substitution, repeated tuning or automatic commit is authorized by this comparison.

D033 outcome (completed 21 October 2026 Asia/Dhaka): all paired arms and four fresh confirmation evaluations completed within 86.04 minutes. V1/V2 confirmation was 55%/70% at seed 43 and 55%/60% at seed 44; gross training attempts were 40/42 and 42/38. The observed mean difference is +10 points with a descriptive paired 95% bootstrap interval of [-10,+30] points. This supports a promising observed improvement but leaves general superiority and the formal target unproven. All 192 engineering checks and training/confirmation audits passed; 235 pre-study notebook/result files retained their SHA-256 values. New work was 120 training and 200 evaluation attempts, with 331 logical model calls. No official test split or cloud work was accessed. See `docs/ADAGRPO_V1_V2_COMPARISON.md` and `results/adagrpo_v1_v2/final_audit.json`.


## D034 - Verified V2 infrastructure, no new model work

On 21 October 2026 the user approved the verified-reliability implementation with no new training or external inference. `AdaGRPO_GSM8K_v2_verified.ipynb` uses protocol `adagrpo-gsm8k-numeric-figure2-v2-verified-v1` and a separate results namespace. Frozen V2 controller, action scoring, numeric reward, learning rate, summed loss, optimizer behavior and sampling streams remain unchanged. All 32 experimental function fingerprints match the frozen source, with matched synthetic trajectories, gradients and Adam state.

Study-level reserved arm paths, sealed configurations, durable statuses, process-owner locking, explicit resume, original deadlines and source-bound shuffled schedules now protect multi-arm execution. Reporting distinguishes gross/committed/discarded and evaluation work, preserves unknown measurements, deduplicates historical references and rejects unsupported schemas instead of assigning zero cost. Historical V1 repeated-work classification uses a labeled episode/split proxy where no exact evaluation identity was recorded.

The prepared fifteen-arm study retains five methods, seeds 42/43/44, 200 episodes, 200 training and 50 validation questions, the 34,935 clean upper workload and unchanged formal acceptance criteria. All arms remain pending; no model work or official tests were launched. Preparation requested only the train split and excludes prior evaluation membership. Two old smoke-validation IDs (5308,1534) were removed from the reusable development training pool; this is the approved cross-experiment exclusion safeguard, not a change to historical runs.

Correction to D010/earlier audit wording: the recorded fifteen-signature map has **eight**, not nine, distinct first-token IDs. Source: `results\adagrpo_v2\gsm8k\adaptive\seed43\screen\20261020T154041-268c0802/config.json`, SHA-256 `ec7c2c97940008129e8e485bfc857521e2ea3efa675726a7e5ef6990b3ca64ab`. The six collision groups and two singleton signatures yield 3+2+2+2+2+2+1+1=15 actions across eight token IDs. Historical reports remain preserved; this correction does not alter action scoring.

Verification: 227 verified checks (163 retained plus 64 new), the existing 192 comparison checks, fresh-kernel no-model/no-HTTP startup, Windows child-process locks, and 364 historical SHA-256 comparisons passed. Tiny synthetic optimizer steps are engineering fixtures, not GPT-2 training or experimental accuracy results. See `results/adagrpo_verified/VERIFICATION_REPORT.md`.

## D035 - Fresh 100-question checkpoint evaluation

On 21 October 2026, the user explicitly authorized an evaluation-only comparison of the validation-selected best checkpoints from the completed seed-43 AdaGRPO V2, Static-K=4, and Static-K=5 development runs. Use the same 100 new questions selected from the pinned GSM8K training split. Before inference, freeze the question IDs, source run paths, best-checkpoint roles and SHA-256 digests. Exclude all recorded historical and reserved GSM8K training/validation memberships, including the earlier V1/V2 confirmation set and prepared formal-study membership. The official GSM8K test split is not loaded.

Use one generated pipeline per question with the frozen V2 evaluation sampling protocol, numeric exact-match reward, unchanged GPT-2 and Llama 3.2 3B identities, and the existing CUDA/CPU-offload settings. Do not train, update an optimizer, select checkpoints from these outcomes, or modify the source runs. Journal each arm in a new `results/adagrpo_fresh100/` directory and persist per-question results for recovery. A single persisted 7,200-second deadline covers model initialization and evaluation and cannot reset. Report repeated/interrupted evaluation attempts as gross evaluation cost. With one seed and 100 train-split questions, paired question bootstrap results are descriptive evidence only and cannot establish the formal multi-seed test-set criterion. No cloud or official-test operation is included.

The self-contained workflow is `AdaGRPO_GSM8K_Fresh100.ipynb`. The study is only complete after all 300 question evaluations and the source/artifact audit pass.

## D036 - Correct fresh-question overlap using Ahnaf's attached study metadata

On 21 October 2026, after D035 evaluation had begun, the user provided Ahnaf's earlier static-K GSM8K study figures and source metadata and asked that they be analyzed while the evaluation continued. The attached `sources.json` records the same pinned GSM8K revision, seed 42 for data selection, 200 training questions and 50 validation questions. The Static-K notebook's selection rule is a NumPy `default_rng(data_seed).permutation` followed by contiguous train/validation slices. Reconstructing the first 250 indices revealed two overlaps in the initial fresh100 set (one prior training and one prior validation question). The initial set was therefore invalidated before either static arm ran. Its 71 started adaptive evaluation attempts (70 successful, one interrupted) remain preserved and are excluded from the corrected accuracy comparison.

The corrected protocol is `adagrpo-fresh100-v2` in `AdaGRPO_GSM8K_Fresh100_v2.ipynb`, under `results/adagrpo_fresh100/corrected_v2/`. It excludes the full reconstructed 200/50 membership plus all other recorded and reserved IDs. It uses the same frozen V2 seed-43 best checkpoints, per-question candidate-stream-v2 evaluation, 100 common fresh GSM8K train-split questions per arm, numeric reward, and no policy/optimizer updates. The corrected ledger inherits the original D035 7,200-second deadline; no time budget is restarted. The official test split remains unopened.

The attached 200-episode figure bundle reports fixed-episode-200 test scores K=5/8/10/15 of 78.5/74.5/72.5/82%, respectively, and validation-selected scores 78.5/79/72.5/82%. Its `sources.json` says `test_size=200` (the plotted 0.5-point increments also indicate 200 questions), while the user's note says 50 test questions; report this discrepancy and verify the raw evaluation manifest before citing the test-set size. The older 30-episode figure bundle reports 76.67/66.67/76.67/70% on 30 held-out questions, 150/240/300/450 training pipeline evaluations, and 40-53% all-equal-reward groups. These results are exploratory, use one seed, and should inform the motivation and diagnostics rather than be pooled with the corrected V2 comparison: model identity, reward fallback, pool/split sizes, run length, and validation/test membership differ. Their evidence reinforces measuring both cost and accuracy and keeping static K=4 as a matched-budget control. No success claim follows from historical plots or the fresh100 evaluation alone.
