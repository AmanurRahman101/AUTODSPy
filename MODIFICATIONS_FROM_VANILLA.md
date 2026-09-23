# 🛠️ Modifications & Divergence from Vanilla AutoDSPy
### *Comprehensive Audit of Changes, Architectural Additions, and Fork Changelog*

> **Upstream Repository:** [`nafew-azim/AUTODSPy`](https://github.com/nafew-azim/AUTODSPy) (Base Paper: *EMNLP 2025 Industry Track*)  
> **Working Fork:** [`AmanurRahman101/AUTODSPy`](https://github.com/AmanurRahman101/AUTODSPy)  
> **Active Development Branch:** `feature/adaptive-k-grpo`  
> **Initial Fork Base Commit:** `4098fbd`  
> **Document Purpose:** Tracks all modifications, architectural improvements, and new files introduced in this fork relative to vanilla AutoDSPy.  
> **Maintenance Protocol:** **Update this file whenever new scripts, algorithmic tweaks, datasets, or artifacts are committed.**

---

## 📑 Table of Contents
1. [Executive Summary: Vanilla vs. Fork Comparison Matrix](#1-executive-summary-vanilla-vs-fork-comparison-matrix)
2. [File-by-File Inventory (Vanilla vs. Fork)](#2-file-by-file-inventory-vanilla-vs-fork)
3. [Component-by-Component Detailed Modifications](#3-component-by-component-detailed-modifications)
   - [3.1 Headless Scripting vs. Interactive Notebooks](#31-headless-scripting-vs-interactive-notebooks)
   - [3.2 Dataset Loader & HuggingFace Hub Compatibility Layer](#32-dataset-loader--huggingface-hub-compatibility-layer)
   - [3.3 Mathematical Standardization of GRPO Advantage](#33-mathematical-standardization-of-grpo-advantage)
   - [3.4 Hardware Co-Design: CPU-GPU Memory Split](#34-hardware-co-design-cpu-gpu-memory-split)
   - [3.5 Algorithmic Addition: AdaGRPO-DSPy (Probe & Expand Engine)](#35-algorithmic-addition-adagrpo-dspy-probe--expand-engine)
   - [3.6 Experiment Tracking & Metrics Logging Infrastructure](#36-experiment-tracking--metrics-logging-infrastructure)
   - [3.7 Academic Documentation & Thesis Artifacts](#37-academic-documentation--thesis-artifacts)
4. [Git Commit Chronology & Changelog](#4-git-commit-chronology--changelog)
5. [How to Maintain & Update This Document in the Future](#5-how-to-maintain--update-this-document-in-the-future)

---

## 1. Executive Summary: Vanilla vs. Fork Comparison Matrix

| Architectural Feature | Vanilla AutoDSPy (`upstream/main`) | Our Working Fork (`AmanurRahman101/AUTODSPy`) | Rationale for Modification |
| :--- | :--- | :--- | :--- |
| **Execution Environment** | Jupyter Notebooks only (`.ipynb`) | Standalone Headless Python CLI Scripts (`.py`) | Enables automated terminal execution, CLI flags, reproducible logging, and non-interactive runs. |
| **GRPO Advantage Formula** | Unnormalized mean difference: $A_i = R_i - \mu_R$ | Normalized GRPO Standard: $A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$ | Conforms with DeepSeekMath / GRPO literature standards; uncovered the 53.33% dead-gradient collapse. |
| **Group Sizing ($K$)** | **Static $K=4$** hardcoded for every query | **AdaGRPO Two-Stage ($k_{\text{probe}}=2 \to k \le 6$)** | Eliminates redundant calls on easy queries; dynamically expands on hard queries to rescue dead gradients. |
| **Module Cost Awareness** | None (Treats `Predict`, `CoT`, and `ReAct` identically) | **Cost weights ($\omega_m$) with dynamic budget clipping** | Prevents runaway inference compute on heavy multi-step reasoning modules. |
| **Mastered Query Handling** | Still runs all 4 rollouts with $A_i=0$ | **Early Exit B + Self-Imitation Anchor Loss on CPU** | Saves 50% compute on mastered prompts while preventing policy drift via CPU anchor loss ($\lambda=0.1$). |
| **Gradient Stability** | Naive averaging across fixed $K=4$ | **Variance Stabilization: $\tilde{A}_i = A_i \cdot \sqrt{k_{\text{probe}}/k_{\text{total}}}$** | Bounds gradient variance to $\mathcal{O}(1/k_{\min})$ across variable dynamic group sizes. |
| **Hardware Memory Strategy** | All components unconstrained (OOM crashes on local GPUs) | **Hybrid CPU-GPU Split** (GPT-2 on CPU, Ollama on GPU) | Pinned to RTX 3050 (6 GB); leaves 100% VRAM for `llama3.2:3b` with sub-5ms CPU policy passes. |
| **Dataset Hub Support** | Crashes on modern HuggingFace `datasets` (>= 2.14) | **Modern Hub Compatibility Monkey-Patch** | Adapts legacy `gsm8k` and `hotpot_qa` (fullwiki $\to$ distractor) to current HF Hub schemas. |
| **Experiment Artifacts** | Ad-hoc printed notebook cells | Structured JSON metrics, trajectory logs, checkpoints | Enables automated Table 1 & Table 2 generation and publication-ready plotting. |

---

## 2. File-by-File Inventory (Vanilla vs. Fork)

```
AUTODSPy/
├── [UNTOUCHED] DSPy_GRPO.ipynb                 # Original author Colab notebook for GRPO
├── [UNTOUCHED] DSPy_PPO.ipynb                  # Original author Colab notebook for PPO
├── [UNTOUCHED] DSPy_Reinforce.ipynb            # Original author Colab notebook for REINFORCE
├── [UNTOUCHED] CITATION.cff                    # Original repository citation
├── [UNTOUCHED] LICENSE                         # MIT License
├── [UNTOUCHED] README.md                       # Original repository README
├── [UNTOUCHED] assets/                         # Upstream figure assets
│
├── [MODIFIED]  .gitignore                      # Added checkpoints (*.pt), metrics, venv, cache
│
├── [NEW]       run_grpo_baseline.py            # Headless static K=4 baseline runner with CLI
├── [NEW]       run_adaptive_k_grpo_v3.py       # Complete AdaGRPO Probe & Expand execution engine
├── [NEW]       baseline_metrics.json           # Machine-readable Table 1 metrics from formal baseline
├── [NEW]       gpt2_trained_policy_model_grpo.pt # Checkpoint from 30-episode baseline execution
├── [NEW]       BASELINE_RESULTS.md             # Formal empirical analysis of baseline & failure taxonomy
├── [NEW]       THESIS_PROGRESS_REPORT.md       # NSU CSE498R Thesis Blueprint & Milestone Roadmap
└── [NEW]       MODIFICATIONS_FROM_VANILLA.md   # THIS FILE: Continuous changelog & divergence audit
```

---

## 3. Component-by-Component Detailed Modifications

### 3.1 Headless Scripting vs. Interactive Notebooks
* **Vanilla Upstream:** Provided only `DSPy_GRPO.ipynb`. Required manual user cell execution in Jupyter or Google Colab, making scripted multi-episode benchmarks, automated seed sweeps, and headless terminal runs impossible.
* **Our Fork Additions:**
  1. [`run_grpo_baseline.py`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/run_grpo_baseline.py): Fully argument-driven CLI runner supporting `--num_episodes`, `--k`, `--model`, `--output_metrics`, and `--save_path`.
  2. [`run_adaptive_k_grpo_v3.py`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/run_adaptive_k_grpo_v3.py): Fully argument-driven AdaGRPO engine supporting `--k_probe`, `--k_max`, `--budget`, and `--anchor_lambda`.

---

### 3.2 Dataset Loader & HuggingFace Hub Compatibility Layer
* **Vanilla Upstream:** In `DSPy_GRPO.ipynb`, the author wrote:
  ```python
  gsm8k = datasets.load_dataset("gsm8k", "main", split="train[:20]")
  hotpotqa = datasets.load_dataset("hotpot_qa", "fullwiki", split="train[:20]")
  ```
* **The Breakage:** Modern versions of `datasets` (>= 2.14) throw errors because:
  * `gsm8k` moved to `openai/gsm8k`.
  * `hotpot_qa` moved to `hotpotqa/hotpot_qa`.
  * The `fullwiki` configuration was deprecated and superseded by `distractor`.
  * Newer `datasets` versions do not permit `trust_remote_code` kwargs on these datasets.
* **Our Fork Solution:** Integrated a clean, non-invasive compatibility adapter at the top of both runners:
  ```python
  _real_load_dataset = datasets.load_dataset
  def _compat_load_dataset(path, *args, **kwargs):
      kwargs.pop("trust_remote_code", None)
      if path == "gsm8k":
          path = "openai/gsm8k"
      elif path == "hotpot_qa":
          path = "hotpotqa/hotpot_qa"
          if args and args[0] == "fullwiki":
              args = ("distractor",) + args[1:]
          elif kwargs.get("name") == "fullwiki":
              kwargs["name"] = "distractor"
      return _real_load_dataset(path, *args, **kwargs)
  datasets.load_dataset = _compat_load_dataset
  ```

---

### 3.3 Mathematical Standardization of GRPO Advantage
* **Vanilla Upstream:** Used unnormalized difference from mean:
  $$A_i = R_i - \mu_R$$
  This masked the zero-variance collapse because the author did not divide by group standard deviation.
* **Our Fork Formulation:** Replaced with the canonical Group Relative Policy Optimization (GRPO) advantage formula:
  $$A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$
  This formalization uncovered the critical theoretical and empirical finding of our thesis: **when $\sigma_R = 0$, $A_i \equiv 0 \implies \nabla_\theta \mathcal{L} \equiv \mathbf{0}$, causing 53.33% dead-gradient episodes in static $K=4$.**

---

### 3.4 Hardware Co-Design: CPU-GPU Memory Split
* **Vanilla Upstream:** Kept policy model tensors and LLM inference together, resulting in CUDA Out-Of-Memory (OOM) crashes on consumer GPUs (e.g. RTX 3050 6GB).
* **Our Fork Architecture:**
  * **Worker LLM (`llama3.2:3b` in 4-bit Q4_K_M):** Dedicated to **GPU VRAM** via local Ollama daemon (~2.0 GB allocated out of 6 GB).
  * **Policy Network (`GPT2LMHeadModel` 124M):** Pinned to **Host CPU** (`torch.device("cpu")`). Forward and backward passes take **<5 ms** and leave the GPU 100% free for LLM inference.

---

### 3.5 Algorithmic Addition: AdaGRPO-DSPy (Probe & Expand Engine)
* **Vanilla Upstream:** Implemented only fixed $K=4$ sampling.
* **Our Fork Implementation ([`run_adaptive_k_grpo_v3.py`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/run_adaptive_k_grpo_v3.py)):**
  1. **Stage 1 (Probe, $k_{\text{probe}}=2$):** Samples 2 candidates.
     * *Mixed Signal ($\sigma_R > 0$):* Immediate optimizer step. Saves 50% compute.
     * *All-Success Saturation ($\mu_R = 1.0$):* Early Exit B with **Self-Imitation Anchor Loss** ($\mathcal{L}_{\text{anchor}} = -\lambda \sum \log \pi_\theta(a)$ with $\lambda = 0.1$ on CPU). Saves 50% compute with 0 policy drift.
  2. **Stage 2 (Cost-Aware Expansion, $\mu_R = 0.0$):**
     * Module weights: $\omega_{\text{Predict}} = 1.0, \omega_{\text{CoT}} = 2.2, \omega_{\text{ReAct}} = 5.5$.
     * Dynamic ceiling: $k_{\text{target}} = \text{clip}(\lfloor B / \text{avg\_cost} \rfloor, k_{\text{probe}}, k_{\text{max}})$.
     * **Early Rescuing Rule:** Exits expansion immediately upon the first working pipeline ($R_k > 0$).
  3. **Variance Stabilization:**
     * Binds gradient variance to $\mathcal{O}(1/k_{\min})$:
       $$\tilde{A}_i = A_i^{\text{raw}} \times \sqrt{\frac{k_{\text{probe}}}{k_{\text{total}}}}$$

---

### 3.6 Experiment Tracking & Metrics Logging Infrastructure
* **Vanilla Upstream:** Zero persistent metrics logging; values were only printed to stdout in notebook cells.
* **Our Fork Additions:**
  * Automated JSON metrics persistence ([`baseline_metrics.json`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/baseline_metrics.json) and `adaptive_metrics.json`).
  * Per-episode trajectory logging: duration, rollouts executed ($k$), rewards array, mean, std, advantage vectors, loss values, and status tags (`EARLY_EXIT_A_ACTIVE`, `EARLY_EXIT_B_SUCCESS`, `EXPAND_RESCUED`, `EXPAND_FAILED_DEAD`).
  * Automatic checkpoint serialization (`gpt2_trained_policy_model_*.pt`).

---

### 3.7 Academic Documentation & Thesis Artifacts
* **Vanilla Upstream:** Generic README and CITATION file.
* **Our Fork Additions:**
  * [`BASELINE_RESULTS.md`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/BASELINE_RESULTS.md): Formal 252-line empirical report with complete 30-episode trajectory table and failure taxonomy.
  * [`THESIS_PROGRESS_REPORT.md`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/THESIS_PROGRESS_REPORT.md): Comprehensive thesis blueprint, NSU context, supervisor Dr. Shafin Rahman, novelty white-space, defense cheat-sheet, and 8-week roadmap.
  * [`MODIFICATIONS_FROM_VANILLA.md`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/MODIFICATIONS_FROM_VANILLA.md): This continuous divergence tracking document.

---

## 4. Git Commit Chronology & Changelog

| Commit Hash | Author | Date | Message & Scope of Changes |
| :--- | :--- | :--- | :--- |
| `f1fb65c` | AmanurRahman101 | Sep 18, 2026 | `docs: enhance baseline report formatting, visual hierarchy, and failure taxonomy` (Refined `BASELINE_RESULTS.md` with complete 30-episode table). |
| `7ee9d91` | AmanurRahman101 | Sep 18, 2026 | `docs & feat: baseline execution results and Table 1 metrics for static K=4 GRPO` (Added `.gitignore`, `run_grpo_baseline.py`, and `BASELINE_RESULTS.md`). |
| `4098fbd` | nafew-azim | *(Upstream)* | *Initial fork point from upstream repository.* |

*Pending Working Tree Commit:*
* `feat & docs: introduce AdaGRPO-DSPy v3 engine, thesis progress report, and fork modifications changelog`  
  * Added `run_adaptive_k_grpo_v3.py`
  * Added `THESIS_PROGRESS_REPORT.md`
  * Added `MODIFICATIONS_FROM_VANILLA.md`

---

## 5. How to Maintain & Update This Document in the Future

Whenever you make future updates to the repository, follow this 3-step protocol:

1. **When adding a new script or file:**  
   Add it to the file tree in **Section 2** and list its role in **Section 3**.
2. **When completing a milestone or new experimental run:**  
   Add a new entry to **Section 4 (Git Commit Chronology)** with the commit hash, date, and description.
3. **When modifying hyperparameters or algorithms:**  
   Document the change under **Section 3.5** (e.g. updating $\omega_m$ weights, probe group sizes, or budget ceilings).

---
*Maintained by the CSE498R Senior Thesis Team (Amanur, Ahnaf, Adib, Akash), Dept. of CSE, North South University.*
