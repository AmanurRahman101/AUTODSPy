# GPT-2 / GSM8K / Llama 3.1 8B reproduction guide

Read [the evidence audit](REPRODUCTION_AUDIT.md) and [deviations](DEVIATIONS_FROM_PAPER.md) first. The original `DSPy_GRPO.ipynb` remains accessible and unchanged. **Do not run its cleanup cell**, which deletes older results. Use `DSPy_GRPO_Reproduction.ipynb` or the commands below from the repository root. No branch change is needed.

## Installation

The existing `venv_gpu` uses Python 3.13 and was inspected with torch 2.6.0+cu124, transformers 5.14.1, DSPy 3.3.0, datasets 5.0.1 and PyYAML 6.0.3. Prefer it for this checkout. A fresh environment can be installed with PowerShell:

```powershell
py -3.13 -m venv .venv-reproduction
.\.venv-reproduction\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv-reproduction\Scripts\python.exe -m pip install -r requirements-reproduction.txt
```

Substitute `.\.venv-reproduction\Scripts\python.exe` for `.\venv_gpu\Scripts\python.exe` below if using the fresh environment. No pytest, sentence-transformers, TRL, PEFT or Ollama Python package is required. PDF readers used for the audit are not training dependencies. Installed transitive versions are recorded in every run manifest.

For notebook use, optionally install `ipykernel` into the selected environment, register it with `python -m ipykernel install --user --name autodspy-reproduction`, and select that kernel in Jupyter/VS Code. The CLI does not require a notebook server.

## Model and dataset setup

On the machine hosting the execution LLM, install Ollama and start its server, then install the exact model:

```powershell
ollama serve
# In a separate terminal:
ollama pull llama3.1:8b
```

Default endpoint is `http://localhost:11434`. To use a different machine, configure its Ollama listening address and access appropriately, then pass `--endpoint http://HOST:11434` to all train/evaluation commands. Only Ollama-compatible endpoints are supported. No alternative model is loaded automatically.

YAML defaults send `num_gpu=0` to the execution server, so Llama runs on CPU while GPT-2 trains on CUDA. A remote server can use a positive explicit `execution_num_gpu` count in a copied YAML to place Llama layers on its own GPU. Set `device: cpu` or pass `--device cpu` if the policy should run on CPU. Changes to endpoint, execution model digest, quantization or generation settings must stay consistent across training and evaluation.

`gpt2` and official `openai/gsm8k`, subset `main`, download automatically through Hugging Face at pinned revisions already present in this checkout's cache. Training uses only `train`; evaluation uses `test`. Raw answers and canonical text after `####` are retained. The target config draws 200 indices with replacement each episode; log files show the actual indices and unique counts. Check [deviations](DEVIATIONS_FROM_PAPER.md) for the unresolved pool ambiguity.

For offline use after files are cached:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
```

Unset these variables when downloads are needed. Dataset fingerprints, selected indices and content hashes are recorded. Do not select checkpoints or tune settings using held-out results; run independent training seeds if assessing variance.

## Static validation and unit tests

```powershell
.\venv_gpu\Scripts\python.exe -m compileall -q autodspy_reproduction tests
.\venv_gpu\Scripts\python.exe -m autodspy_reproduction validate --config configs/grpo_gsm8k.yaml
.\venv_gpu\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

Tests cover first-token collisions, generation/termination, behavior/current probabilities, centered and standardized advantages, equal rewards, notebook loss, both clipping signs, entropy, finite nonzero gradients, full-graph versus accumulated gradients, numerical answers, split guards, checkpoint/RNG round trips, exact resume and four-way evaluator wiring. Test fixtures are never benchmark results.

## Live probe and smoke training

```powershell
.\venv_gpu\Scripts\python.exe -m autodspy_reproduction preflight --config configs/grpo_smoke_test.yaml
.\venv_gpu\Scripts\python.exe -m autodspy_reproduction execute --config configs/grpo_smoke_test.yaml
.\venv_gpu\Scripts\python.exe -m autodspy_reproduction train --config configs/grpo_smoke_test.yaml
```

Smoke uses **the real GPT-2 and Llama 3.1 8B**, one episode, two train samples and K=5: ten pipeline evaluations, plus any LLM judge/adapter calls. Check the printed unique run path, `status.json`, `metrics.jsonl`, `trajectories.jsonl`, `manifest.json`, `initial_policy.pt` and `episode_0001.pt`. All-equal rewards can legitimately produce zero reward gradients; inspect `zero_advantage_groups` instead of treating zero loss as success or failure by itself. Finite loss/gradients, real completed calls and saved checkpoints establish the live smoke milestone.

A separate GPU engineering check uses the real pretrained policy and official train data with a **scripted execution fixture**:

```powershell
$env:PYTHONPATH=(Get-Location).Path
.\venv_gpu\Scripts\python.exe tests/policy_smoke.py
```

Its artifacts live under `runs/engineering-policy-smoke`, and its artificial rewards/latencies are not accuracy measurements or a passing live Llama smoke. It tests policy memory, optimizer integration and checkpoint loading only.

## Full training (explicit command only)

```powershell
.\venv_gpu\Scripts\python.exe -m autodspy_reproduction train --config configs/grpo_gsm8k.yaml
```

This performs 200 episodes x 200 sampled prompts x 5 candidates = **200,000 pipeline evaluations**, potentially more LLM calls for fallback judging or DSPy adapters. One Adam update occurs per episode after collecting all 200 prompt groups; gradients average prompt-group losses but retain the released sum across candidates/actions. No textbook clipping, entropy, GAE or KL is silently introduced. Gamma/lambda and the paper's shared entropy setting are evidence metadata, not active GRPO settings.

Every run gets a unique directory. Initial policy is saved once; full checkpoints save every ten episodes and at episode 200. Checkpoints include model, Adam, sampler/Python/PyTorch/CUDA RNG, complete config, data/model/runtime identity and completed episode count. Budget estimates should use actual smoke latency; do not extrapolate the paper's A6000 two-hour claim to a laptop. At 10 seconds per pipeline, 200,000 evaluations alone take about 23 days; at 60 seconds, about 139 days, before judging/training overhead. These are arithmetic examples, not measurements. Full checkpoints require roughly 30GB across twenty saves, plus logs/initial weights; actual sizes are recorded on disk.

## Resume and checkpoint loading

```powershell
.\venv_gpu\Scripts\python.exe -m autodspy_reproduction train --config configs/grpo_gsm8k.yaml --resume runs/reproduction/RUN/episode_0010.pt
```

Resume requires identical config and data/policy/execution provenance. It creates a new run directory and continues at episode 11 without overwriting prior artifacts. Interrupted unsaved episodes must be repeated from the last completed checkpoint. Exact local deterministic resume is unit tested; bit-identical LLM output across servers/runtime versions is not promised.

For inference-only Python loading:

```python
from autodspy_reproduction.config import load_config
from autodspy_reproduction.policy import load_policy
from autodspy_reproduction.runtime import load_checkpoint
config = load_config('configs/grpo_gsm8k.yaml')
policy = load_policy(config)
metadata = load_checkpoint('runs/reproduction/RUN/episode_0200.pt', policy)
trajectory = policy.generate('What is 5 + 3?')
```

Legacy root-level `.pt` weights are not automatically loaded because they do not establish provenance or initialization. Keep the associated `initial_policy.pt`; evaluation locates it through checkpoint metadata. If moving run artifacts, preserve that path or explicitly repair its recorded location after documenting the move.

## Matched evaluation

For a completed full checkpoint:

```powershell
.\venv_gpu\Scripts\python.exe -m autodspy_reproduction evaluate --config configs/grpo_gsm8k.yaml --checkpoint runs/reproduction/RUN/episode_0200.pt
```

For evaluator validation following a smoke run, use **the smoke YAML** and its `episode_0001.pt`. The evaluator rejects mismatched training/execution settings. It processes the same selected official held-out examples for static Predict, static CoT, saved initial untrained policy and trained policy, using one policy object sequentially. Static signatures are `question -> answer`; all methods receive identical bracket instructions and execution generation settings.

Results save after each method in `results.json`; all predictions, pipelines, scores and latency components save in `predictions.jsonl`. Exact numerical accuracy, LLM-fallback accuracy, execution/judge errors and pipeline frequencies are separate. Inference latency includes policy generation and DSPy execution, excludes judge time, and includes any model loading unless it was already warm. Sequential method order and warm/cold conditions are recorded by runtime/call logs; use an explicit common warm-up protocol for publication timing and do not compare CPU laptop timing to paper GPU timing.

Default evaluation uses all 1,319 official test cases. A separately named YAML with `evaluation_limit: 1300` creates a recorded seeded subset for approximate paper comparison; unknown paper indices prevent exact subset matching. The appendix's embedding-based semantic subset cannot be reproduced without its unidentified embedding model. No results are generated as placeholders.

## GPU memory and troubleshooting

```powershell
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv -l 2
```

Run this in a separate terminal during smoke/training. Metrics report PyTorch **policy-process** peak allocated/reserved bytes; this excludes Ollama, display and other processes. Use nvidia-smi for total-device usage. Float32 Adam/GPT-2 weights, gradients and moments alone require about 2GB before activations/workspace. The real engineering smoke measurements are recorded in [validation](VALIDATION_RESULTS.md); a scripted policy-only run is not full-experiment memory measurement.

- **Connection refused:** start Ollama on the configured host; check `/api/version` and `/api/tags`. The CLI preflights before allocating policy weights and aborts on missing target models.
- **CPU Llama memory/timeout:** Q4_K_M weights are roughly 4.9GB, with additional context/runtime RAM. Close memory-intensive applications yourself, increase timeout in a copied config, or use a remote server with sufficient RAM. Do not substitute a smaller model.
- **CUDA out of memory:** ensure the execution LLM is CPU/remote, inspect other GPU use, or set `gradient_checkpointing: true`. Alternatively run the policy on CPU explicitly. Do not reduce the final episodes/K/model size. Checkpointing may change runtime but not the intended loss.
- **Non-finite logits/gradients or replay mismatch:** run stops before Adam updates; inspect the failing settings. Do not hide this with uniform action fallback. Use FP32 and consistent zero dropout.
- **Invalid DSPy output/judge:** inspect per-candidate error/raw judge logs; malformed numeric judge scores are zero and flagged. Infrastructure errors abort. Empty/invalid pipelines are zero reward and flagged. Ollama/DSPy adapter retries can add calls beyond nominal pipeline counts.
- **Config mismatch on evaluation/resume:** use the exact training YAML/endpoint. Use an independent, labeled run for methodological changes. Resume does not silently change settings.
- **Cache/download errors:** use the pinned revisions; remove offline environment variables when downloading. Authenticate only if your environment requires it; the chosen policy and dataset are public.

The isolated notebook starts with static validation and opt-in controls for execution, smoke, full training and evaluation. Its default top-to-bottom execution never launches the full experiment or deletes artifacts.
