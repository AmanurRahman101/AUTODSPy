# Static-K experiment guide

Independent notebooks: `Static_K_GRPO_GSM8K.ipynb` and `Static_K_GRPO_HotpotQA.ipynb`. Each contains its own complete workflow and tests. The [master specification](docs/MASTER_SPEC.md), [audit](docs/PHASE_1_AUDIT.md), and [progress](docs/PROGRESS.md) distinguish published methods, released code, project settings and actual verification.

## Environment and initialization

Use a Jupyter kernel with PyTorch, Transformers, DSPy, datasets, NumPy and Matplotlib. On this laptop, the existing `venv_gpu` has CUDA PyTorch; the default system Python has CPU-only PyTorch. Register the intended kernel if necessary:

```powershell
venv_gpu/Scripts/python.exe -m ipykernel install --user --name autodspy-gpu --display-name "AutoDSPy GPU"
```

Start Ollama and ensure `ollama pull llama3.1:8b` has installed the requested execution model. Edit `OLLAMA_HOST` for another server. The notebook probes model availability, records its digest/quantization and refuses silent substitutions. `OLLAMA_NUM_GPU=0` explicitly runs Llama on CPU to leave GPU room for GPT-2; automatic or partial GPU loading can be requested explicitly. Use identical settings across K comparisons.

Open the chosen notebook and run its definition cells top to bottom with `RUN_OPERATION=False`. Optional dependency installation is controlled by `INSTALL_DEPENDENCIES`; it does not reinstall PyTorch. Run `notebook_self_checks()` to verify synthetic mechanics. No model training or final testing starts from setup.

`run_operation(CONFIG, "initialize")` explicitly loads official data, checks Ollama,
and initializes pretrained GPT-2 with a fresh Adam optimizer. It creates a run but
does not train. To train that initialized session, call `train_grpo(SESSION)`.
Alternatively, `run_operation(CONFIG, "train")` initializes and trains a fresh run.
Release an existing session with `release_experiment(SESSION)` before starting another.

## A K=5 smoke run

Edit the configuration cell, then run these commands in a new notebook cell after all definitions:

```python
K = 5
PRESET = "smoke"
NUM_EPISODES = None
CONFIG = build_config()
SESSION = run_operation(CONFIG, "train")
```

The default smoke uses five episodes, a 200-question training pool, 50 validation questions and the same persisted schedule as pilot/baseline. Episode-zero and final validation may cost more than the five training episodes. For engineering checks only, override the pool, validation/test sizes and episode count explicitly. Keep such runs separate from research results.

An initialization-only call is `SESSION = run_operation(CONFIG, "initialize")`; then explicitly call `train_grpo(SESSION)` when ready. It initializes the immutable GPT-2 revision with fresh Adam state, checks Ollama, loads the corresponding official dataset and persists split/schedule metadata. It never searches for an existing checkpoint.

## Other K values and presets

Before replacing an active session, call `release_experiment(SESSION)`. Set K to 8, 10 or 15, rebuild CONFIG, and start a new train operation. Every new run reloads identical pretrained parameters and a fresh optimizer. Never load K=5 weights into K=8.

`PRESET="pilot"` selects 20 episodes; `"baseline"` selects 200. `NUM_EPISODES`, `TRAIN_POOL_SIZE`, `VALIDATION_SIZE`, `TEST_SIZE`, `VALIDATION_INTERVAL` and `CHECKPOINT_INTERVAL` are editable. `SCHEDULE_EPISODES` is the shared schedule length, at least the largest planned run. Keep it consistent across presets/K. The default smoke/pilot sequences are prefixes of the persisted 200-episode schedule.

Explicitly run all four K values sequentially with `run_all_k_sequential(CONFIG)`. This performs training and validation, not final testing. No full experiment is started by this guide or notebook setup.

## Questions, episodes and results

The default training pool has 200 examples. Each of 200 episodes selects one question with replacement and attempts K pipelines. Repeats are allowed; logs record actual unique questions encountered. All K runs use exactly the same saved question at each episode. Validation comes from disjoint official training examples; testing comes from official test examples. Gold rationales never enter model inputs.

Artifacts use `results/gsm8k/k<K>/seed<seed>/<preset>/<unique-run>/`. Shared selection manifests live in `results/gsm8k/selections/`. Each run stores configuration, immutable checkpoints, latest/best pointers, JSONL training/validation logs, an attempt journal, summary and separately invoked evaluation artifacts.

## Saving and resuming

`save_checkpoint(SESSION)` explicitly saves a safe between-episode state. Training also saves at its configured interval, every validation and completion. Read the path from that run's `latest.json` and resume with the **same configuration**:

```python
release_experiment(SESSION)
SESSION = run_operation(CONFIG, "resume", checkpoint="results/.../checkpoints/episode_....pt")
```

Model, Adam, Python/NumPy/Torch/CUDA RNG, metrics and the entire selection schedule restore together. K, preset, episode budget, data, model identity and decoding/resource controls must match. Completed pilot checkpoints cannot become baseline runs. Historical checkpoints may be evaluated, but resume must use the latest durable checkpoint.

Recover the exact resolved configuration from the run snapshot rather than relying
on current notebook defaults or the latest dataset revision:

```python
RUN_DIR = Path("results/.../your-run")
CONFIG = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))["config"]
LATEST = json.loads((RUN_DIR / "latest.json").read_text(encoding="utf-8"))["checkpoint"]
SESSION = run_operation(CONFIG, "resume", checkpoint=LATEST)
```

Interrupted validation is retried before the next training episode, including after the final update. Completion requires final validation. Evaluation-only sessions cannot train or save training checkpoints; explicitly resume to restore matching optimizer state.

If interrupted within an episode, restore the latest checkpoint before continuing. Periodic saving can lose unsaved updates: their metrics are rolled back too, so the restored effective history has no duplicate updates or rows. The append-only journal retains external attempts that cannot be undone. Interval 1 minimizes rollback but writes large model/optimizer files every episode. Check disk capacity; keep only deliberately chosen artifacts after a run.

## Validation and final testing

Baseline validation runs at 0, 50, 100, 150 and 200. Each question gets one pipeline, never K, and no updates. Validation numeric EM selects the checkpoint; earliest episode wins ties. Use the best checkpoint path from `best.json`:

```python
release_experiment(SESSION)
SESSION = run_operation(CONFIG, "validate", checkpoint="results/.../selected.pt")
TEST = test_model(SESSION)  # requires the best checkpoint and completed training
```

Alternatively use `run_operation(CONFIG, "test", checkpoint=...)`. Final testing is never called by training. `test_model(SESSION, full_official=True)` explicitly evaluates the full official test split later. Do not tune using held-out scores.

GSM8K now supports `REWARD_MODE="original_semantic"` (default), `"binary_semantic"`, and `"numeric_exact"`. The default keeps the existing fractional rubric and original numeric extractor, with new runs using versioned judge controls, private persistent caching, and detailed logging. This improved cached protocol differs from the completed uncached experiments. Set `LEGACY_REWARD_PROTOCOL=True` to preserve the uncached historical reward behavior; old snapshots without reward fields are recognized as legacy automatically. Merely disabling the cache on an improved configuration does not select the legacy protocol.

Binary-semantic mode accepts only correct/equivalent final answers as 1, with no partial credit. Numeric-only mode uses a conservative final-answer extractor and makes no judge calls. Both return only 0/1. Strict numeric EM and semantic fallback are reported separately; the extractor version is recorded. Full settings, snapshot procedures, error policy and checkpoint compatibility are in [REWARD_MODES.md](docs/REWARD_MODES.md). The evidence and limits of the disagreement investigation are in [REWARD_JUDGE_INVESTIGATION.md](docs/REWARD_JUDGE_INVESTIGATION.md). No reward method is retrospectively applied to old results.

## Comparisons, plots and resource accounting

```python
RUNS = compare_runs(["results/.../completed-run-1", "results/.../completed-run-2"])
FIGURES = plot_comparison(RUNS, save_dir="results/gsm8k/comparison_plots")
```

Only compatible completed runs compare. All eleven figures display inside the notebook and contain only recorded results; missing tests/measurements have no invented points. Repeated runs remain distinct. The first-token categorical sampler retains collisions; the Figure 2 objective sums over K, so gradient magnitude is an explicit limitation.

Comparison checks the configuration/selection manifest, complete episode history,
exactly K attempts per episode, reward centering, validation progression and cumulative
training budget, latest/best checkpoint pointers, and held-out checkpoint/question
provenance. Missing checkpoints/manifests or inconsistent logs are rejected before
plotting. Each run path may appear once; independent repeats at the same K are allowed.
Keep recorded paths and checkpoint files available: automatic artifact relocation is
not supported. GSM8K and HotpotQA, different seeds, presets, subsets, decoding budgets,
model digests or runtime versions cannot be combined as one controlled K comparison.

Reward modes/protocols, judge controls and cache policies/snapshot identities must also match. GSM8K section 24's standalone saved-data analysis labels its tables and places new exports under protocol/hash subdirectories of `results/gsm8k/static_k_analysis/` and `results/gsm8k/static_k_comparison/`. It leaves the existing root exports unchanged. Section 25 provides `reward_protocol_self_checks()` using mocks and saved fixtures only.

The compact `summary.json` distinguishes effective training/evaluation costs from
observed journal costs including rollback. It also reports recorded GPU peaks,
training/judge/LM failures, evaluation success/failure counts and LM time. Missing
measurements or incomplete historical journals stay `null`. Journal totals cover
completed entries; interrupted in-flight calls can remain unrecorded.

Sequential rollout/replay bounds activation retention without changing the loss. Finite equal-reward groups still make Adam steps, whose momentum may move weights despite zero current gradients. PyTorch peak allocated/reserved memory and Ollama residency are distinct measurements. Calls mean measured `LM.forward` attempts, including judges/adapter retries; exact physical HTTP requests are unavailable. Token totals require returned usage. Training time excludes validation/checkpoint I/O. Fixed seeds do not guarantee bitwise remote inference across hardware or server versions.

Actual checks and unresolved limitations are reported in [PHASE_2_REPORT.md](docs/PHASE_2_REPORT.md). Clean-kernel checks and a real GPT-2 update passed. A reduced live run completed with truncation failures and zero rewards; it does not demonstrate learning. No expensive baseline result is implied by synthetic tests or reduced live engineering checks.

## HotpotQA configuration and use

Use the same K, preset, mode and checkpoint commands inside the HotpotQA notebook;
it never imports GSM8K. Results are under `results/hotpotqa/k<K>/seed<seed>/<preset>/<run>/`.
The default official `distractor` allocation is 200 training examples, 50 additional
training-split validation examples and 200 labeled development examples for final
testing. `TEST_SOURCE_SPLIT="train"` explicitly selects a third disjoint training
slice after both training and validation. This alternative creates a different
manifest and cannot be compared as the same protocol. Full evaluation explicitly
uses all labeled development examples, never unlabeled official test questions.

All provided paragraphs enter the model input, in their original order, followed
by the question. Gold answers/supporting-fact annotations remain metadata. The
paper's exact task configuration was not verifiable; consistent provided-context
distractor QA is a documented project choice. There is no search or retrieval.

Answer EM uses official HotpotQA normalization; token F1 follows its answer
evaluator, including the yes/no/noanswer rule. Training reward remains exact-match
one followed by the original fractional Llama semantic judge on mismatches.
F1 never enters the reward or checkpoint selection. Highest validation answer EM
selects the checkpoint, with the earliest episode winning ties. Testing reports
both EM and F1, and the notebook displays twelve figures, including their comparison.

GPT-2 left-truncates long inputs to `MAX_POLICY_TOKENS=1024`, retaining the question
and action suffix while potentially losing earlier context. Trajectories record
token counts and truncation flags. Lower this control explicitly only when needed
and record the input deviation; keep it fixed across K. Llama receives the complete
formatted input subject to `OLLAMA_NUM_CTX`. Server truncation is not directly
measured. CPU offload requires enough host RAM; a remote Ollama endpoint may be
more practical on a 6GB laptop. Never silently change the execution model.

For official-data checks without initializing either model:

```python
CONFIG, DATASETS = load_hotpotqa(build_config())
DATA = prepare_selection(CONFIG, DATASETS)
CHECKS = dataset_self_checks(CONFIG, DATASETS, DATA)
```

Phase 3 checks, actual integration outcomes and unrun workloads are recorded in
[PHASE_3_REPORT.md](docs/PHASE_3_REPORT.md). Final consolidation and verification are
reported in [PHASE_4_REPORT.md](docs/PHASE_4_REPORT.md).

## Changing K, budgets and presets

Release the prior policy, change `K` to 8, 10 or 15, and rebuild `CONFIG` before
starting `run_operation(CONFIG, "train")`. Every call starts independently from
the same pinned pretrained GPT-2 revision; it never inherits the earlier K's weights.
Set `PRESET="pilot"` for 20 episodes or `"baseline"` for 200. Keep `NUM_EPISODES=None`
to use that preset's default. Validation defaults to 0/5 for smoke, 0/10/20 for
pilot, and 0/50/100/150/200 for baseline.

Change `TRAIN_POOL_SIZE`, `VALIDATION_SIZE`, `TEST_SIZE`, `NUM_EPISODES`,
`VALIDATION_INTERVAL`, or `CHECKPOINT_INTERVAL` explicitly when needed, then call
`build_config()`. `build_config(num_episodes=..., train_pool_size=...)` also permits
per-call overrides. Size/revision/seed/schedule changes produce a different selection
namespace. Smoke and pilot use baseline schedule prefixes only when those settings
match. Changing the execution model is an explicit protocol change requiring matching
settings in every compared K run; it is never done automatically.

An optional invocation trains all four K values sequentially:

```python
# Explicitly starts four runs; choose the preset and sizes first.
RUN_DIRS = run_operation(build_config(), "train", all_k=True)
```

This releases each policy before initializing the next and checks initial-parameter
and schedule identity. It does not trigger final testing. Default 200-episode budgets
are 1,000/1,600/2,000/3,000 candidate attempts for K=5/8/10/15: 7,600 per dataset.
Failures still count as attempts. These budgets exclude validation/testing and differ
from measured LM calls.

## Monitoring GPU memory and interpreting evidence

Inspect `training.jsonl` for per-episode `peak_allocated_gpu_bytes`,
`peak_reserved_gpu_bytes`, `ollama_memory`, `out_of_memory_failures`, and failure
details. `summary.json` aggregates recorded training peaks. CPU-policy runs have
`null` PyTorch GPU peaks. Ollama residency is a separate server snapshot; neither
measurement is total system GPU usage. `nvidia-smi` can inspect current device use.

Sequential rollout/replay retains one candidate's graph at a time. The 6GB laptop
still needs room for policy parameters, gradients and Adam state; Llama CPU offload
needs sufficient host RAM. The previous HotpotQA 1024-token CUDA check reserved about
5.36 GiB without concurrent live Llama training. Its reduced live integration instead
used a CPU policy and an explicitly shorter input budget. Checkpoint files include
policy and optimizer state, so frequent saves need substantial disk space.

Phase 4 reruns lightweight notebook checks and reads previous live engineering
artifacts; it does not execute new Llama inference or baseline training. The saved
Phase 4 plots under `results/engineering_checks/phase4-plots/` visualize those tiny
engineering runs only. Full presets, live K=8/10/15, full held-out datasets, default
joint-model hardware operation and live induced recovery remain unrun.
