# GSM8K semantic judge inconsistency investigation

This investigation inspected the current notebook, original released notebook and paper reward sections, experiment guide, saved analysis, configurations and candidate/journal records. It introduced no real LLM inference, GPT-2 retraining or evaluation of historical checkpoints. Existing results, checkpoints, plots, summaries, and the original notebook remain unchanged.

## What the historical records establish

The local GSM8K Arrow cache was read directly, without downloading data. Reconstructing the saved 50-question selection reproduced its recorded content hash, `99aa5f2f4c9c016ccbba241fca6d5abd81d59078c82a4cde3ed9d52bffacafd2`. This verifies the recovered questions and canonical targets. The [saved fixture](audit_artifacts/reward_disagreement_fixture.json) preserves exact candidate output strings and observations.

- **K=8, `gsm8k:train:4423`:** target 6. The same Predict / `question -> reasoning` output concludes 13.5 servings, having omitted the cake expense. Episode 1 candidate 4 has reward/judge score 0; episode 18 candidate 3 has 0.5.
- **K=10, the same question and identical output:** episode 1 candidate 4 has 0; episodes 5 and 18 candidate 4 have 0.5.
- **K=15, `gsm8k:train:2477`:** target 52. The same Predict / `question -> hypothesis` output formulates `x + 108 = 160` without presenting the solved final value. Within episode 8, candidates 2, 10 and 14 have scores 0.5, 0 and 0.

All these entries record `numeric_exact_match=False`, `semantic_used=True`, and no judge error. Their recorded reward equals their recorded judge score. These are genuine contradictions in the saved *judgments*, not a difference between numeric reward and logged advantage. Original-semantic partial-credit grading can legitimately assign a fractional score to incomplete reasoning, but identical requests should not silently alternate between scores in a controlled experiment.

The saved configurations agree on Llama 3.2 3B identity/digest, temperature 0, seed 42, 512 output tokens, context 4096, cache disabled and provider retries zero. They do not contain raw judge request/response traces or a per-run source revision. The recovered question/target and saved candidate output would form the same prompt under the inspected function; the actual transmitted historical request cannot be independently certified from those artifacts.

## Reproduction and ranked explanations

A small deterministic mock drove the actual pre-change `compute_reward()` with identical question, response and target. A judge returning `[0]` and then `[0.5]` yielded 0 and 0.5 in two calls. An equality assertion failed before implementation. This proves that uncached independent judgments expose the exact inconsistency pattern; it does not prove which server behavior produced the historical responses.

The evidence supports the following assessment:

1. **Repeated judge-response variation is plausible and directly reproducible with a mock.** Temperature zero and a seed do not prove server-level determinism. The exact model/kernel/server cause remains unidentifiable from saved logs.
2. **Different effective request/context or parameters are not demonstrated.** Snapshots and recovered inputs match, and the K=15 discrepancy occurs within one episode. Missing per-call requests/provider responses prevent excluding hidden context, request overrides or unsupported settings.
3. **Parsing is deterministic for a fixed raw response under the inspected parser.** Complete `[0]` and `[0.5]` parse consistently. Historical raw responses and exact source versions are absent, so their formatting or past parser behavior cannot be reconstructed.
4. **Numeric extraction is not a sufficient explanation for these same-output disagreements.** The candidate string and canonical target are identical, and every observation took semantic fallback. The historical last-number convention can still create other false exact matches; it is retained in the original mode and avoided by the new conservative modes.
5. **Missing caching is confirmed as an enabling mechanism, not the fundamental model root cause.** Identical requests were independently judged rather than bound to one recorded result. This permits drift even when recorded controls match.

**Root cause conclusion:** the precise historical cause is unresolved. The implementation can reproduce and prevent the repeated-input reward drift through immutable request caching, and now captures evidence needed to distinguish future response, parsing and configuration differences. It does not claim to make the LLM judge perfectly deterministic or correct.

## Changes that address the problem

- `original_semantic` remains the default and keeps the existing fractional rubric, parser and extractor. Its improved reproducibility controls are explicitly versioned separately from the uncached legacy behavior.
- `binary_semantic` adds a strict final-answer equivalence rubric and parser returning only 0/1. This is a paper-aligned implementation choice: section 3.3 and Figure 2 describe binary rewards, while the released notebook permits fractional fallback. The paper does not prescribe all implementation details of this new judge.
- `numeric_exact` provides deterministic numerical rewards without semantic calls. Its conservative parser rejects arbitrary reasoning numbers; written-out or algebraic answers can be false negatives under this mode.
- Semantic clients pin model/digest, model context, runtime and generation controls. No model is substituted or pulled automatically. Execution settings and the GPT-2 policy remain unchanged.
- Full request identity, versioning and immutable persistent entries make a repeated request return the recorded result, including a distinctly recorded failure. Safe publication never overwrites the winning entry. Integrity errors/frozen misses abort rather than silently regrade.
- Private caches avoid mutable shared-cache dependence on which K ran first. A common complete frozen snapshot gives stronger cross-run control; incomplete coverage fails explicitly. Private caches alone do not guarantee matching first judgments in independent runs.
- Logs retain full raw requests/responses, effective parameters, identifiers, attempts, parse/provider failures, measured calls, timing and cache origin. Failures after the explicit attempt budget still return the existing zero fallback, distinguished from a valid incorrectness judgment.
- Checkpoint loading verifies reward/judge/cache compatibility before loading policy/optimizer state. Old snapshots are classified as legacy and restored without retroactive configuration additions.
- Summaries and both comparators carry reward identity. Statistics reject mixed reward protocols and new exports use separate protocol/hash directories; existing research exports are not regenerated.

See [REWARD_MODES.md](REWARD_MODES.md) for configuration examples, snapshot export, error policy and legacy checkpoint instructions.

## Verification and preservation

The new tests cover numerical normalization/malformed answers, fractional and binary parsing, numeric-only no-call behavior, persistent repeated-input consistency, mode/parameter/version isolation, strict client identity, logging without credentials, retries, frozen snapshot hit/miss, cache corruption, receipts, historical configuration compatibility, and replay of all three saved disagreement patterns.

They also exercise the production initialization, training, validation, testing, checkpoint and analysis paths using a **32-parameter CPU policy and mocked execution/judge responses**. These transient fixtures are engineering checks, not experimental results. No GPT-2 or Llama model is loaded for those tests.

The exact Figure 2 implementation source hash remains `3ced73a45e50796d6e5ef6cacf96424a0dd3963238652cf5d0d2539d43d22e24`. Centered-advantage and summed-loss gradient checks pass. **280 reward checks**, the prior **111 notebook synthetic checks**, and **13 saved-data analysis checks** pass on the restored notebook: **404 checks total**, including the individual preservation checks. The current pipeline sampler is preserved unchanged; the four earlier diagnostic-sampler checks are not included in this restored-version total. Notebook schema validation and code compilation also pass. Named checks are in [reward_protocol_tests.json](audit_artifacts/reward_protocol_tests.json) and [reward_legacy_regression_checks.json](audit_artifacts/reward_legacy_regression_checks.json); the combined outcome is [reward_validation_summary.json](audit_artifacts/reward_validation_summary.json).

The preservation baseline covers **116 existing artifacts**, including the four completed run directories, analysis exports/PDFs, original notebook and paper. Small artifacts are checked with SHA256, size and modification time; checkpoint files are guarded by size/modification time and are not rehashed or evaluated. The baseline is [reward_artifact_preservation.json](audit_artifacts/reward_artifact_preservation.json). Pre-existing deleted historical files and untracked work were left alone.

## Remaining limits and next research step

Caching preserves the first recorded decision; it does not prove that decision is correct. It also preserves recorded error zeros, so failure counts must be examined. Partial-credit rubric ambiguity remains intentionally present in original-semantic mode. Server-level response nondeterminism and unsupported seed behavior cannot be diagnosed retrospectively. Any live repeatability diagnostic should be separately authorized and save complete traces under fixed model/settings; none was launched automatically.

Use the same reward mode/version, extractor, cache policy/snapshot identity, execution/judge models, generation controls, selections and schedule in future Static-K and Adaptive-K comparisons. For any improved cached, binary or numeric experiment, train **new matching Static-K baselines** before claiming Adaptive-K accuracy/cost gains. Prefer all four K values with independent seeds and larger evaluation sets. The old runs remain untouched evidence about the legacy uncached protocol; converting their stored rewards cannot reproduce the policies a different reward method would have trained.

## Saved-notebook repair

The missing implementation was recovered from the exact earlier edit commands and merged into the current notebook. The cause of its absence cannot be established from the available evidence. The current experiment controls (K=15, seed 42, Llama 3.2 3B, learning rate 1e-5, baseline 30 episodes), sampler, Figure 2 implementation and user operation/plot cells were retained. `REWARD_MODE` now appears on the first line of the configuration cell. A full pre-repair backup and saved-file verification record are linked from [reward_notebook_recovery.json](audit_artifacts/reward_notebook_recovery.json). The notebook was reopened from disk for the mock checks and schema validation; no real training or inference was launched.
