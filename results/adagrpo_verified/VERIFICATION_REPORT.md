# AdaGRPO V2 verified implementation

Implemented reliability and evaluation infrastructure while preserving the frozen V2 estimator. No GPT-2 training, Ollama inference, cloud launch or official-test access occurred. Optimizer verification used tiny synthetic policies only.

## Changes

- Self-contained [verified notebook](../../AdaGRPO_GSM8K_v2_verified.ipynb), artifact protocol `adagrpo-gsm8k-numeric-figure2-v2-verified-v1`; controller `credit-refine-v2` and sampling `candidate-stream-v2` retained.
- Fifteen ordered arms have reserved directories, sealed configurations, dependencies and durable statuses. Resume restores incomplete arms, audits completed arms, retains original deadlines and charges interrupted work. OS locks prevent concurrent owners and release after process death.
- Shuffled K references its adaptive arm by key and binds the source summary, checkpoint, K sequence and permutation seed. Source/schedule changes are rejected before initialization.
- Read-only auditing and accounting separate training, validation, confirmation, repeated/interrupted/discarded work, and new versus historical costs. Missing usage stays unknown. Logical calls are never presented as physical HTTP requests.
- Old V1 repeated-evaluation classification uses an explicitly labeled episode/split proxy where exact policy identities were unavailable. Older Static-K schemas are listed as unsupported rather than counted as zero.
- Recorded signature map confirms 15 actions and **8** distinct first-token IDs; the historical reports saying nine remain preserved.

## Verification

- **227 verified checks passed**: retained 163 V2 checks plus 64 new checks.
- **192 comparison regression checks passed** independently; these overlap the retained V2 checks and are not an additional 192 unique scenarios.
- All **32 frozen experimental function fingerprints** match V2, including initialization, execution, sampling, evaluation, both training definitions, numeric extraction, updates, bootstrap and acceptance logic. Configurations match except artifact protocol/namespace.
- Matched synthetic rollouts, rewards, gradients, learned parameters and Adam state agree exactly, including equal-reward steps.
- Synthetic recovery covers initialization, partial validation, partial collection, and interruption between arms. Finished validation is not repeated; lost training attempts remain charged. All fifteen synthetic study arms completed and passed artifact audits.
- Fresh-kernel execution passed for 22 definition cells with pretrained model/tokenizer initialization and HTTP blocked.
- Separate Windows child-process tests confirmed concurrent-run rejection, normal lock reuse, and lock release after abrupt process exit. POSIX locking is implemented but was not exercised on this Windows host.
- SHA-256 preservation passed for **364** historical notebook, result, checkpoint and documentation files.

## Prepared evaluation — not launched

- [Frozen study ledger](studies/formal-v1/study.json): V2, Static-K=2/4/5 and shuffled K; seeds 42/43/44; 200 training questions, 200 replacement-sampled episodes, 50 validation questions at episodes 0/50/100/150/200.
- Membership and per-seed schedules are frozen across methods. Two old smoke-validation questions (`gsm8k:train:5308`, `gsm8k:train:1534`) were removed from the reusable development training pool and replaced with unused questions. This prevents cross-experiment evaluation reuse in the new cohort; old results remain unchanged.
- Status is **prepared**; all 15 arms are **pending**, no study clock has started, no run checkpoints exist, and only the GSM8K training split was requested.
- Clean upper workload: **34,935 executions** (11,400 training, 3,750 validation, 19,785 separately authorized official tests).
- Recorded local mean latency implies approximately **147.9 pipeline-execution hours** for that upper workload. This excludes initialization, checkpoint/update overhead and failures; it is not a cloud runtime estimate.
- Two earlier unlaunched preparations are preserved in `preparation_history/` as development provenance. They contain no started arms or inference work and are superseded by the final frozen ledger.

## Use

Run the notebook definitions in the existing AdaGRPO GPU kernel, then:

```python
prepared = prepare_study()  # idempotently returns the frozen preparation
audit_study(prepared['ledger_path'])
```

Future model work requires a separate explicit operation:

```python
# Do not run as part of this implementation.
run_study(prepared['ledger_path'], allow_training=True)
# After interruption: same ledger, same budgets and deadline.
run_study(prepared['ledger_path'], resume=True, allow_training=True)
```

Final testing remains a separate explicit best-checkpoint `load_run(..., for_resume=False)` / `test_model()` operation. `formal_acceptance(path)` refuses incomplete studies or missing official-test records. The original >=20% gross-training saving and >=-2-point paired lower-bound criteria remain unchanged, including cheaper-static/shuffled dominance reporting.

Keep the notebook with the reference notebook and recorded reference metadata in the project layout. Runtime/model identity is pinned to the archived reference; a changed runtime fails clearly and requires a separately reviewed study rather than a silent resume. No new Python modules or dependencies were introduced.

## Evidence

- [Verified checks](self_checks.json)
- [Comparison regression](comparison_regression.json)
- [Kernel check](clean_kernel.json)
- [Process-lock checks](lock_process_checks.json)
- [Historical preservation](preservation.json)
- [Prepared study audit](prepared_study_audit.json)
- [Existing evidence accounting](existing_evidence_accounting.json)
- [Signature correction](signature_audit.json)

These changes establish engineering correctness under the recorded checks. They make no new accuracy, speedup or formal research-success claim.
