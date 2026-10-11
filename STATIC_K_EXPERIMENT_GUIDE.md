# Static-K experiment guide

Current deliverable: `Static_K_GRPO_GSM8K.ipynb`. HotpotQA belongs to Phase 3 and is not implemented yet. The [master specification](docs/MASTER_SPEC.md), [audit](docs/PHASE_1_AUDIT.md), and [progress](docs/PROGRESS.md) distinguish published methods, released code, project settings and actual verification.

## Environment and initialization

Use a Jupyter kernel with PyTorch, Transformers, DSPy, datasets, NumPy and Matplotlib. On this laptop, the existing `venv_gpu` has CUDA PyTorch; the default system Python has CPU-only PyTorch. Register the intended kernel if necessary:

```powershell
venv_gpu/Scripts/python.exe -m ipykernel install --user --name autodspy-gpu --display-name "AutoDSPy GPU"
```

Start Ollama and ensure `ollama pull llama3.1:8b` has installed the requested execution model. Edit `OLLAMA_HOST` for another server. The notebook probes model availability, records its digest/quantization and refuses silent substitutions. `OLLAMA_NUM_GPU=0` explicitly runs Llama on CPU to leave GPU room for GPT-2; automatic or partial GPU loading can be requested explicitly. Use identical settings across K comparisons.

Open the GSM8K notebook and run its definition cells top to bottom with `RUN_OPERATION=False`. Optional dependency installation is controlled by `INSTALL_DEPENDENCIES`; it does not reinstall PyTorch. Run `notebook_self_checks()` to verify synthetic mechanics. No model training or final testing starts from setup.

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

An initialization-only call is `SESSION = run_operation(CONFIG, "initialize")`; then explicitly call `train_grpo(SESSION)` when ready. It initializes the immutable GPT-2 revision with fresh Adam state, checks Ollama, loads official GSM8K main and persists split/schedule metadata. It never searches for an existing checkpoint.

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

Strict numeric EM and fractional semantic fallback are separate. Exact comparison fixes comma/decimal/negative-number bugs. The original last-number extraction convention can still confuse reasoning with the answer. Reward preserves the same Llama judge, not an embedding evaluator or extra shaping.

## Comparisons, plots and resource accounting

```python
RUNS = compare_runs(["results/.../completed-run-1", "results/.../completed-run-2"])
FIGURES = plot_comparison(RUNS, save_dir="results/gsm8k/comparison_plots")
```

Only compatible completed runs compare. All eleven figures display inside the notebook and contain only recorded results; missing tests/measurements have no invented points. Repeated runs remain distinct. The first-token categorical sampler retains collisions; the Figure 2 objective sums over K, so gradient magnitude is an explicit limitation.

Sequential rollout/replay bounds activation retention without changing the loss. Finite equal-reward groups still make Adam steps, whose momentum may move weights despite zero current gradients. PyTorch peak allocated/reserved memory and Ollama residency are distinct measurements. Calls mean measured `LM.forward` attempts, including judges/adapter retries; exact physical HTTP requests are unavailable. Token totals require returned usage. Training time excludes validation/checkpoint I/O. Fixed seeds do not guarantee bitwise remote inference across hardware or server versions.

Actual checks and unresolved limitations are reported in [PHASE_2_REPORT.md](docs/PHASE_2_REPORT.md). Clean-kernel checks and a real GPT-2 update passed. A reduced live run completed with truncation failures and zero rewards; it does not demonstrate learning. No expensive baseline result is implied by synthetic tests or reduced live engineering checks.
