# 📌 Project Blueprint: AdaGRPO-DSPy — Adaptive Group Sizing for Compound AI System Compilation

## 1. Executive Summary & Context
* **Institution:** Dept. of CSE, North South University (NSU), Dhaka
* **Course:** CSE498R Senior Undergraduate Thesis
* **Supervisor:** Dr. Shafin Rahman
* **Team:** Amanur Rahman, Ahnaf, Adib, Akash
* **Base Paper:** *"AutoDSPy: Automating Modular Prompt Design with RL for Small and Large Language Models"* (EMNLP 2025 Industry Track)
* **Base Repo:** [`nafew-azim/AUTODSPy`](https://github.com/nafew-azim/AUTODSPy) | **Branch:** `feature/adaptive-k-grpo`
* **Core Objective:** Replace the static candidate group rollout size ($K=4$) in AutoDSPy's GRPO with **AdaGRPO-DSPy**: an online, uncertainty- and cost-aware dynamic group allocator. It eliminates zero-advantage gradient updates, cuts worker LLM inference calls by >30%, and maintains task convergence accuracy.

---

## 2. Problem Statement: The Static K=4 Bottleneck
AutoDSPy frames prompt/pipeline engineering as RL. A trainable Policy Network (GPT-2, 124M) samples module topologies executed by a frozen Worker LLM (`llama3.2:3b`), updated via Group Relative Policy Optimization (GRPO).

* **Compute Waste:** On mastered queries, all 4 rollouts succeed ($R=1.0$); rollouts 3 & 4 offer 0 new gradient signal.
* **Exploration Failure:** On complex queries, $K=4$ is too small to find valid topologies ($R=0.0$).
* **Cost Asymmetry:** Compound pipelines vary in execution cost (1 call for `Predict` vs. 5–10 calls for `ReAct`). Static $K=4$ treats every candidate identically.

### ⚠️ Dead-Gradient Mathematical Proof
In GRPO:
$$A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}, \quad \nabla_\theta \mathcal{L}(\theta) = -\frac{1}{K} \sum_{i=1}^K \left[ A_i \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \right]$$

When all rollouts yield identical rewards ($R_1 = \dots = R_K = C \in \{0.0, 1.0\}$):
$$\mu_R = C, \quad \sigma_R = 0 \implies A_i = 0 \implies \nabla_\theta \mathcal{L}(\theta) \equiv \mathbf{0}$$

*Outcome:* Over 50% of LLM calls in static setups burn compute for **zero** weight updates.

---

## 3. Current Milestones & Empirical Proof

### 3.1 Uncontested Novelty White-Space
* **Token-Level Generation (Saturated):** AERO (Feb 2026), SAGC (Jun 2026), DAPO (May 2025) focus on token-level LLM math/code outputs on large clusters.
* **Compound Pipeline Space (Uncontested):** Nobody has applied dynamic group allocation with module execution cost-awareness ($\omega_m$) to compound AI system compilation.

### 3.2 Hardware Co-Design
* **Worker LLM (GPU):** `llama3.2:3b` in 4-bit (`Q4_K_M`, 2.0 GB) pinned to dedicated VRAM (2.4 GB allocated out of 6 GB GDDR6 on RTX 3050 Laptop).
* **Policy Network (CPU):** GPT-2 (124M, ~500 MB) pinned to `torch.device("cpu")` (forward/backward passes <5 ms; leaves GPU VRAM entirely free).

### 3.3 Verified Baseline Metrics (Table 1: 30 Episodes, Seed 42)
| Metric | Static Baseline ($K=4$) | Academic Implication |
| :--- | :---: | :--- |
| **Total Worker LLM Calls** | 120 calls | Fixed $30 \times 4$ |
| **Active Learning Episodes ($\sigma_R > 0$)** | 14 / 30 (46.67%) | Meaningful updates |
| **Dead Gradient Episodes ($\sigma_R = 0$)** | **16 / 30 (53.33%)** | **Over half of training produced zero learning** |
| ↳ *All-Failure Collapse ($\mu_R = 0$)* | 11 / 30 (36.67%) | Under-exploration on hard queries |
| ↳ *All-Success Saturation ($\mu_R = 1$)* | 5 / 30 (16.67%) | Redundant sampling on trivial queries |
| **Wasted Inference Calls** | **64 / 120 (53.33%)** | Zero training utility |
| **Final Mean Reward (Last 5)** | 0.9622 | Baseline convergence target |

*Baseline reference artifacts:* [`BASELINE_RESULTS.md`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/BASELINE_RESULTS.md) · [`baseline_metrics.json`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/baseline_metrics.json) · [`gpt2_trained_policy_model_grpo.pt`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/gpt2_trained_policy_model_grpo.pt)

---

## 4. The Solution: AdaGRPO-DSPy Workflow

1. **Stage 1 (Informative Probe, $k=2$):**
   * If $\sigma_R > 0$ ➔ **Early Exit A:** Normal update, saves 50% compute.
   * If $\mu_R = 1.0$ ➔ **Early Exit B:** Mastered query. Run self-imitation anchor loss ($\lambda=0.1$) on CPU; 0 extra LLM calls.
2. **Stage 2 (Cost-Aware Iterative Expansion, if $\mu_R = 0.0$):**
   * Compute architecture cost: $\omega_{\text{Predict}}=1.0, \omega_{\text{CoT}}=2.2, \omega_{\text{ReAct}}=5.5$.
   * Calculate budget: $k_{\text{target}} = \text{clip}(\lfloor B / \text{avg\_cost} \rfloor, k_{\text{probe}}, k_{\text{max}})$.
   * Resample candidates up to $k_{\text{target}}$, **exiting on the first successful pipeline** to immediately rescue the zero-gradient state.
3. **Variance Stabilization:**
   * Scaled advantage $A_i = \text{raw\_advantage} \times \sqrt{k_{\text{probe}} / k_{\text{total}}}$ to bound gradient variance to $\mathcal{O}(1/k_{\min})$ across dynamic sizes.

---

## 5. Comparative Targets & Road Ahead

### Projected Comparison (Table 2)
* **LLM Calls:** 120 ➔ **~74 to 88 (~30%+ reduction)**
* **Dead Gradient Episodes:** 16 / 30 (53.3%) ➔ **$\le$ 5 / 30 (<17%)**
* **Final Accuracy:** Maintained at **$\ge$ 0.96**

### Next Steps
* [x] **Milestone 1:** Baseline verification & hardware co-design ([`BASELINE_RESULTS.md`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/BASELINE_RESULTS.md)).
* [ ] **Milestone 2 (Weeks 1–2):** Run [`run_adaptive_k_grpo_v3.py`](file:///c:/Users/rahma/OneDrive/Documents/Projects/University%20Projects/CSE498R/AUTODSPy/run_adaptive_k_grpo_v3.py) on seed 42 to log Table 2 metrics.
* [ ] **Milestone 3 (Weeks 3–5):** Scale to $N=100$ across 3 seeds (42, 123, 999) on GSM8K + HotpotQA.
* [ ] **Milestone 4 (Weeks 6–8):** Compile ACL/EMNLP template paper and defend under Dr. Shafin Rahman.

---

## 6. Quick Defense Cheat-Sheet
* **Why not fixed $K=2$ or $K=8$?:** Static $K=2$ collapses on hard prompts ($R=[0,0]$); static $K=8$ doubles compute costs on trivial queries. Only dynamic sizing handles both extremes.
* **Novelty vs. AERO/DAPO?:** Those optimize token-level LLM generation. None account for compound multi-module execution costs ($\omega_m$) in prompt/pipeline compilers.
* **Gradient Stability?:** The scaling factor $\sqrt{k_{\text{probe}} / k_{\text{total}}}$ mathematically dampens advantage spikes on large expansion groups.
