# Local environment and operation

## Recorded research setup

- Windows laptop: Lenovo LOQ, Intel Core i5-13420H, 16 GB RAM.
- Verified GPU: RTX 3050 Laptop, 6 GB VRAM.
- Python 3.11.9 in `venv_adagrpo/`; Jupyter kernel **AdaGRPO GPU**.
- Recorded PyTorch: `2.6.0+cu124`.
- GPT-2 on CUDA in FP32; one policy resident; sequential collection and replay.
- Execution model: **`llama3.2:3b`**, Ollama CPU offloading.
- Latest matched/V3 runtime: context **4096**, output limit **512**. Earlier protocols used different context settings; do not silently treat them as matched.

These are recorded experiment settings, not a fresh verification of whatever environment is installed when this guide is read. Exact dependency versions, model digest, quantization, and runtime belong to each run's configuration/preflight artifacts.

## Existing environment

From the repository root in Codex's integrated PowerShell terminal:

```powershell
.\venv_adagrpo\Scripts\Activate.ps1
python --version
```

Select **AdaGRPO GPU** when opening a notebook. If the environment/kernel is missing, follow the setup section in the notebook for the intended protocol. Rebuilding an environment is a separate action; dependency upgrades can change the experimental runtime.

## Ollama

The explicitly selected model can be installed with:

```powershell
ollama pull llama3.2:3b
```

If the desktop service is not already running, start it with `ollama serve`. Do not start a second server on an occupied port. Never substitute a different model automatically. Notebook configuration controls CPU offloading; the latest comparison must not be mixed with a separate GPU-offloading experiment.

## Inspecting the completed V3 study

Open `AdaGRPO_GSM8K_v3.ipynb`, execute only its documented definition cells, then use its read-only operation:

```python
audit_study('results/adagrpo_v3/screen_seed42/study.json')
```

The existing JSON report and closeout audit can also be read directly without importing model dependencies or executing notebook cells.

To view the saved log inside Codex:

```powershell
Get-Content .\results\adagrpo_v3\screen_seed42\training_terminal.log -Tail 30
```

This is a completed study log. Adding `-Wait` watches future writes; it does not start training, and closing the viewer does not stop a worker.

## Model-work boundary

The V3 screen is complete and failed its gate. Do not run another screen or reset its deadline from a setup guide. New training, altered controller experiments, or formal cloud work require a new approved experiment. Original reference notebooks may execute model work from top-level cells; use the [notebook catalog](../catalogs/NOTEBOOKS.md) before choosing Run All.

## Recovery and sharing

Use only a matching notebook/protocol with its frozen ledger, configuration, code identity, and original deadline. Do not resume historical weights under a revised controller. Copy complete run directories, including Git-ignored `.pt` files, when transferring machines. [Artifact handling](../../results/README.md).
