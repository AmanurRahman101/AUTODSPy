# GSM8K reward modes and reproducibility

The GSM8K implementation is self-contained in `Static_K_GRPO_GSM8K.ipynb`: configuration in section 3, reward/cache functions in section 12, and mock tests in section 25. Its source and the original `DSPy_GRPO.ipynb` remain unchanged by the HotpotQA adaptation described below. No training or inference starts when defining these functions.

## Three modes

- **`original_semantic` is the default.** Keep the existing numeric extractor, exact-match reward 1, and byte-identical fractional semantic-judge rubric. The existing parser accepts a complete finite numeric response, optionally in brackets, and preserves its original clamping to [0,1]. Fractional scores remain fractional. There is no format bonus or binary conversion.
- **`binary_semantic`** uses the conservative numeric extractor for an immediate reward 1. Otherwise a versioned LLM prompt asks whether the *final answer* is correct or semantically equivalent. Partial reasoning without a correct final answer earns no credit. The parser accepts only complete 0/1 numeric responses, optionally bracketed; a fractional or prose response is a logged judgment failure, not a thresholded score. Rewards are always 0 or 1. This follows the paper's stated binary formulation; the exact rubric, parsing and caching details are implementation choices.
- **`numeric_exact`** uses the conservative numeric extractor and returns only 0 or 1. It never calls or creates a separate semantic judge. Pipeline execution still uses the configured execution LLM. A failed or unparseable final answer receives 0.

The conservative `explicit-final-number-v1` extractor accepts a standalone number, a final numeric square-bracket answer, the suffix after `####`, or a clearly marked answer. It normalizes commas, decimals, negative values, common currency/unit formatting and simple Markdown formatting. It rejects multi-number arithmetic expressions, malformed comma groups, NaN/Infinity, and arbitrary last numbers in reasoning. It does not solve expressions or recognize written-out numbers: binary-semantic evaluation can judge those; numeric-only evaluation cannot. It is deliberately conservative and may reject an otherwise correct free-form answer. The original-semantic extractor keeps the historical last-number convention, including its known limitation, to preserve that methodology.

Validation/test checkpoint selection remains strict numeric exact-match accuracy. Original-semantic runs retain the historical extractor for that metric; the other two modes use the conservative extractor. Semantic credit is always reported separately. These extractor versions are part of reward identity and cannot be mixed as one comparison.

## Switching modes

For a new run using the notebook's other current controls:

```python
REWARD_MODE = "original_semantic"  # or "binary_semantic" or "numeric_exact"
LEGACY_REWARD_PROTOCOL = False
JUDGE_CACHE_ENABLED = True
JUDGE_LOGGING_ENABLED = True
CONFIG = build_config()
```

Alternatively:

```python
CONFIG = build_config(reward_mode="original_semantic")
CONFIG = build_config(reward_mode="binary_semantic")
CONFIG = build_config(reward_mode="numeric_exact")
```

Numeric mode automatically disables the judge-result cache. These examples only construct configuration; they do not start a run. The saved-notebook repair preserves the current configuration controls, including baseline 30 episodes, learning rate 1e-5, seed 42 and Llama 3.2 3B. For a new study, check these controls explicitly. To derive controls and saved selections directly from a historical run while introducing a **new** reward protocol:

```python
BASE_RUN = Path("results/gsm8k/k5/seed42/baseline/20261013T174921-b1ad8988")
SAVED = json.loads((BASE_RUN / "config.json").read_text(encoding="utf-8"))["config"]
BASE = {key: value for key, value in SAVED.items() if key not in REWARD_SETTING_KEYS}
CONFIG = build_config(**dict(BASE, reward_mode="original_semantic"))
# This would create a fresh independent experiment only when explicitly invoked.
# SESSION = run_operation(CONFIG, "train")
```

Never load an old trained policy as the starting policy of a new reward-protocol baseline. Changing reward mode does not require editing internal reward functions.

## Explicit legacy option and old checkpoints

```python
CONFIG = build_config(**dict(BASE, reward_mode="original_semantic",
                            legacy_reward_protocol=True,
                            judge_cache_enabled=False))
```

Or set `LEGACY_REWARD_PROTOCOL=True` before `build_config()`. Legacy mode inherits the original execution client's generation settings, makes one judge attempt, disables caching, keeps the original rubric/parser/extractor, and returns zero on judge failure. Enabling logging only adds observational records. Merely setting caching to false on an improved run does **not** select the legacy protocol: its standardized client and other controls still differ.

Old snapshots without reward fields are inferred as `legacy-original-semantic-v1`. They are never interpreted as improved defaults. For any later explicitly authorized checkpoint load, recover the exact saved configuration:

```python
CONFIG = json.loads((BASE_RUN / "config.json").read_text(encoding="utf-8"))["config"]
POINTER = json.loads((BASE_RUN / "latest.json").read_text(encoding="utf-8"))
# SESSION = run_operation(CONFIG, "resume", checkpoint=POINTER["checkpoint"])
```

The completed four runs remain completed: they cannot be resumed as longer/new experiments. Changing reward mode, cache policy/snapshot, relevant generation settings, or judge identity is rejected by compatibility checks. Historical configuration is restored unchanged when loading an old checkpoint. New checkpoint reward/cache checks occur before loading policy or optimizer state. Four-argument `compute_reward(...)` calls remain a legacy compatibility seam for the prior notebook fixtures; production training/evaluation always pass the configuration and run context explicitly.

## Standardized semantic judge

Improved semantic modes use a separate measured **client**, not another GPT-2 policy or a duplicate copy of the execution model. By default `JUDGE_MODEL=None` inherits the explicitly configured execution model. Selecting another judge model must be explicit; missing models or digest mismatches raise an error. No model is pulled or substituted automatically.

New-run initialization resolves and pins the Ollama model digest before snapshots or grading. `JUDGE_MODEL_DIGEST` can require a previously known SHA256 digest. The model template, system text and model parameters are captured from Ollama metadata; their digest and dependency/Ollama identity are pinned too. Temperature zero and a fixed seed improve control, but do not establish server-level determinism.

Defaults are temperature 0, seed inherited from the experiment seed, maximum output 512 tokens, context 4096, top-p 1.0, top-k 40, repetition penalty 1.1, and the configured execution endpoint/GPU allocation/timeout. They are explicit judge controls and are recorded separately from execution controls. Improved mode checks that the active client matches the pinned model, endpoint, chat format and generation settings, and rejects unversioned provider overrides. It explicitly uses the legacy chat-output contract and verifies captured forwarded requests against the pinned request before cache publication or an update. Unexpected request drift aborts the episode. DSPy/provider caching and hidden retries are disabled.

`JUDGE_MAX_ATTEMPTS=1` and `JUDGE_RETRY_DELAY_SECONDS=0` are defaults. Larger explicit attempt budgets retry the same request after a provider or parser failure. Every attempt is logged. After exhaustion the original zero-reward error fallback is retained. Cache entries distinguish such **judge failures** from valid decisions that an answer is incorrect; an error score has `judge_score=None` and a non-null `judge_error`. Failure entries are cached too, so a repeated request does not silently flip from a failed zero to a later score. This can preserve a transient failure: inspect failures and use a fresh declared run/cache protocol for a corrected experiment rather than editing cache entries mid-run.

## Persistent cache and common snapshots

`JUDGE_CACHE_ENABLED=True` is the new semantic default. The key hashes the full question, ground truth, exact candidate response, exact judge prompt/chat messages, reward mode/protocol, prompt/parser/extraction/cache versions, model name/digest/context, runtime identity, effective generation settings, and retry/error rules. It excludes K, episode, candidate index, pipeline choice and question ID because those do not affect this judgment rubric. Identifiers are logged separately. The implementation matches **exact evaluation requests**; it does not infer that differently worded responses are paraphrases.

Entries live in each new run's `judge_cache/`. An optional `JUDGE_CACHE_ROOT` still creates one namespace per run, rather than a shared mutable store. Entries have immutable exclusive publication, SHA256 request/content checksums, finite/range/mode validation, flushed safe writes, and no overwrite path. Concurrent publication returns the first valid published decision; a different locally observed response is logged as a publication race. Unknown/corrupt cache versions, changed contents and invalid binary entries fail closed rather than being silently regraded.

Checkpoint metadata contains checksums of cache entries that the checkpoint used. Loading verifies those entries still exist and match. Extra immutable entries from interrupted work may remain, just as the external-attempt journal remains; their measured calls are not erased. Retain the cache alongside checkpoints. Snapshot export also refuses to overwrite an existing file.

Private caches avoid the order effect in which the first K run fills a mutable store for the next K. They **do not guarantee identical first judgments across independent runs**. For a stronger controlled comparison, precompute a common set of full requests under one standardized judge, then publish a single immutable snapshot and require that snapshot in all runs:

```python
# No LLM call: explicitly export already recorded cache entries.
CACHE = SESSION["reward_context"]["cache"]
SNAPSHOT_ID = CACHE.export_snapshot("results/gsm8k/judge_snapshots/study_v1.json")

CONFIG = build_config(**dict(BASE,
    reward_mode="original_semantic",
    judge_cache_enabled=True,
    judge_cache_policy="frozen",
    judge_cache_snapshot="results/gsm8k/judge_snapshots/study_v1.json",
    judge_cache_snapshot_id=SNAPSHOT_ID))
```

All compared K values must use the same snapshot content ID, judge controls and reward protocol. Frozen mode makes no judge call on a cache miss; it raises `FrozenJudgeCacheMiss`. A snapshot from one run may not cover another run's outputs, especially when execution itself varies. Do not describe partial coverage as a deterministic complete benchmark. Precomputation/repeatability inference is a separate explicitly authorized experiment; none was launched here. Do not tune checkpoint selection or allocation using held-out test scores.

## Logging and scientific grouping

With `JUDGE_LOGGING_ENABLED=True`, each new run adds `judge_evaluations.jsonl`. Training/evaluation candidate records also carry mode, protocol, cache status and semantic traces. Records include full requests/responses, model/digest/context, effective whitelisted parameters, raw and parsed outcomes, cache source/hit/miss, attempt failures and timing, measured forward calls, and question/candidate/episode identifiers. Cache hits distinguish the original grading trace from current zero-call work. Authentication keys are excluded. Numeric/exact-match bypasses are recorded without claiming a semantic call occurred.

The existing measured ledger includes both execution and judge clients, so a cached judge hit reduces measured judge calls without removing a pipeline execution. Pipeline counts remain K per training episode. Provider/server/physical HTTP details not actually observed remain unavailable. Log/cache I/O failure aborts the episode instead of producing an unrecorded update.

Configurations, summaries, checkpoints, validation/test artifacts and analysis carry reward identity. Comparators reject mixed protocols and incompatible judge/cache controls. Sequential all-K execution carries the first run's resolved settings forward and rejects execution/judge protocol drift before training another K. Standalone analysis labels every derived data table and writes into protocol/run-set hash subdirectories beneath `static_k_analysis/` and `static_k_comparison/`, protecting existing exports from a different selected run set. Server residency snapshots are recorded as observed latency context rather than being treated as an immutable model identity.

Keep separate cohorts for:

1. Legacy uncached original semantic: `legacy-original-semantic-v1`.
2. Improved original semantic: `original-semantic-repro-v2`, with cache enabled/disabled and private/frozen policies distinguished.
3. Binary semantic: `binary-semantic-v1`, with its cache/judge identity.
4. Numeric exact: `numeric-exact-v1`, with its conservative extraction version.

Prompt, parser or grading changes require version changes. Even an unchanged rubric with new caching/settings is a different training protocol. Original Figure 2 loss, group centering, optimizer stepping, policy/action generation and scheduling remain unchanged; old rewards and checkpoints are never retroactively converted.

## Tests and next experiments

Run `reward_protocol_self_checks()` after loading notebook definitions. Tests use mocks, saved disagreement fixtures and a tiny CPU policy fixture, not GPT-2 retraining or Ollama inference. Outcomes are saved in `docs/audit_artifacts/reward_protocol_tests.json`; prior regression outcomes are in `reward_legacy_regression_checks.json`. See the [investigation report](REWARD_JUDGE_INVESTIGATION.md) for the actual verification counts and remaining uncertainty.

For Adaptive-K under the improved cached/default, binary, or numeric protocol, **new Static-K baselines under that same protocol are necessary**. Prefer the four K values and multiple seeds with identical schedules/selections and prespecified budgets. The existing runs remain evidence about the legacy protocol, not evidence about policies that a new reward method would train. Reusing them is defensible only for an Adaptive-K experiment that exactly retains their legacy reward protocol and other controls, with the existing small-sample limitations stated.

## HotpotQA adaptation

`Static_K_GRPO_HotpotQA.ipynb` now carries the same judge controls, private immutable
cache/snapshot machinery, logging, checkpoint receipts, protocol checks and saved-data
analysis. It uses full provided-context inputs and normalized textual answers.
Its modes are `original_semantic`, `binary_semantic`, and `answer_exact`; the latter
never calls a judge. EM selects checkpoints and answer F1/precision/recall remain
separate metrics. Original semantic retains bracket-or-text extraction; the other
modes use versioned explicit-final-text extraction. Historical snapshots without
reward fields retain the uncached legacy HotpotQA interpretation.

HotpotQA reward/cache versions use a `hotpotqa-` prefix and record
`answer_extraction_version`; they must not be pooled with GSM8K protocols or
their numeric extractor. Cache keys include the complete context/question,
response and ground truth. The [verification record](audit_artifacts/hotpotqa_parity_verification.json)
reports 135 notebook assertions and 167 reward/integration/analysis assertions,
all using mocks and tiny CPU policies. These checks overlap in subject matter;
they do not establish benchmark accuracy or live HotpotQA feasibility.
