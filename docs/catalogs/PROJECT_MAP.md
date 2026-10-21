# Project map

## Source entry points

The root notebooks remain at stable paths for reproducibility and frozen source checks. [Notebook catalog](NOTEBOOKS.md) classifies them as controller implementations, evaluations, historical baselines, and upstream references.

`run_grpo_baseline.py` is a legacy baseline runner. `BASELINE_RESULTS.md`, `baseline_metrics.json`, and `gpt2_trained_policy_model_grpo.pt` are legacy baseline artifacts, not the latest AdaGRPO result. The metrics file and model weights are excluded from Git. `STATIC_K_EXPERIMENT_GUIDE.md` documents the earlier baseline workflow.

## Documentation layout

- `docs/README.md` - navigation and report authority.
- `docs/catalogs/` - notebooks, results, and this project map.
- `docs/guides/` - local operation and environment.
- `docs/research/` - current research interpretation and paper library.
- `docs/reference/` - exact documentation preserved before organization.
- `docs/audit_artifacts/` - checks and preservation records.
- `docs/diagnostics/` - earlier diagnostic notebook and evidence.
- `docs/agents/` - workflow instructions for automated agents.
- Existing `docs/*.md` - reports, decisions, original specifications, and historical handoffs. Keep their dates and experimental scope in mind.

## Experimental artifacts

`results/` has separate namespaces for original Static-K, V1, V2, verified infrastructure, comparisons, and V3. See the [results catalog](RESULTS.md). Do not move or merge these directories: their internal paths, hashes, and identities are used in recovery and integrity checks.

Original benchmark figures are in `results/gsm8k/static_k_comparison/`; extracted statistics are in `results/gsm8k/static_k_analysis/`. Some teammate-supplied material remains external; the context report documents provenance and unresolved metadata.

## Papers and figures

`AutoDSPy.pdf` and `assets/` are the checked-in foundational paper and original figures. The wider nine-paper library is `../papers/`, outside Git; see the [paper catalog](../research/PAPERS.md).

## Tooling and local-only directories

- `.git/` - version control; never reorganize by hand.
- `.codex/`, `.agents/`, `AGENTS.md`, and `skills-lock.json` - agent instructions and installed skill metadata.
- `graphify-out/` - generated knowledge graph and cache; dirty output is expected after updates.
- `venv_adagrpo/` - current local research environment.
- `venv/` - earlier environment, preserved.
- `__pycache__/` - generated Python cache, ignored by Git.

Environments, generated caches, and checkpoint weights are not a substitute for portable setup and complete run backups.

## Preservation policy

This organization pass adds navigation documents, replaces the root README with an accurate entry point, and corrects citation metadata. It preserves all existing notebooks, experimental artifacts, model checkpoints, paper files, scripts, and historical reports byte-for-byte. The preceding encoding repair is recorded separately in `docs/audit_artifacts/notebook_encoding_repair_20261021.json`.

Original filenames stay stable. No training, inference, notebook test suite, checkpoint migration, run deletion, or Git push is part of this organization pass. New source work should have its own protocol/results namespace and a documented entry in the navigation.
