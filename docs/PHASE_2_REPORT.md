# Phase 2 — GSM8K implementation and verification

Completed 2026-10-12 (Asia/Dhaka). Scope: the complete independent GSM8K notebook, its experiment controls, documentation and lightweight verification. Phase 3 has not started. No expensive baseline was launched.

## Deliverables

- [Static_K_GRPO_GSM8K.ipynb](../Static_K_GRPO_GSM8K.ipynb): 21 numbered sections, 21 code cells, all executable research logic and tests inside the notebook. Saved outputs are empty; opening/running setup does not train.
- [STATIC_K_EXPERIMENT_GUIDE.md](../STATIC_K_EXPERIMENT_GUIDE.md): initialization, K switching, presets, recovery, validation, final testing, plots and resource limitations.
- Updated plan, progress and decisions. MASTER_SPEC.md remains unchanged.
- Verification JSON in [audit_artifacts](audit_artifacts/README.md), actual reduced-run artifacts under `results/engineering_checks/phase2-live/`, and the default official subset/schedule manifest under `results/gsm8k/selections/`.
- `graphify-out/` refreshed through the required AST-only command. The extractor skipped notebooks and lacked its PDF extra; its graph does not represent the new notebook logic. No semantic/API extraction was run.

Original DSPy_GRPO.ipynb SHA-256 remains `a787bc87b6e652d30ef483f6c3ef8525df351c0f5572a3ee5ac4c1455d863f98`. Original paper, existing notebooks/runner/reports, user instructions and skills were preserved. No HotpotQA notebook or implementation module was created. Changes in this phase have not been committed automatically.

## Workflow and research protocol

The notebook follows the requested order: objective, dependencies, controls, hardware/Ollama, official data, sampling, original actions/signatures, GPT-2, action generation, pipeline construction/execution, reward, Figure 2 loss, training, checkpoints, validation, testing, statistics, comparisons, plots and reproducibility.

Defaults retain GPT-2, Llama 3.1 8B, Adam 1e-4, seed 42, K=5/8/10/15, train pool 200, validation 50 and official-test subset 200. Smoke/pilot/baseline select 5/20/200 episodes. The one-prompt-per-episode replacement schedule is persisted independently of K/preset. Independent deterministic streams use seed, seed+1 and seed+2 for training permutation, test permutation and episode draws respectively. Every K uses the same episode question. The default 200-episode schedule contains 130 distinct training IDs; this describes the planned schedule, not 200 executed training episodes.

Each new run loads immutable pretrained GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` and fresh Adam. The official GSM8K main revision verified here is `740312add88f781978c0658806c59bc2815b9866`. Selected IDs/indices and content hashes persist; train/validation indices are disjoint and test comes from the official test split. Full test data remains accessible without automatically evaluating it.

The loss is exactly the negative sum of action log probabilities times centered rewards. No K division, reward standard-deviation normalization, clipping, entropy, GAE, KL, gradient clipping or adaptive selection is added. Rollouts have no autograd graphs; sequential differentiable replay accumulates equivalent gradients before one Adam step. Evaluation-mode dropout matches the released pretrained model behavior. Finite equal-reward groups retain the original Adam step, which may still move weights through momentum. Gradient norms explicitly expose K scaling.

## Starting, saving and evaluating

Select the `AutoDSPy GPU` Jupyter kernel, start Ollama and ensure `llama3.1:8b` is installed. Run definition cells with operation switches false. Set K=5 and PRESET="smoke", then explicitly call `SESSION = run_operation(build_config(), "train")`. Release an existing session before changing K; start K=8/10/15 with fresh initialization. `run_all_k_sequential(CONFIG)` is an explicit all-K request, never setup behavior.

Run directories include dataset/K/seed/preset/unique run ID. Checkpoints are immutable; latest/best pointers are atomic. Save explicitly between episodes with `save_checkpoint(SESSION)`. Resume an interrupted run using its explicit latest checkpoint and identical configuration. Policy, optimizer, all RNG states, episode schedule and metrics restore together. Historical checkpoints and completed pilot configurations cannot silently resume as baseline training.

Recovery starts from the next durable episode. Periodic saving cannot recover unsaved optimizer work: those updates and their statistical rows roll back together. The attempt journal preserves externally executed work. Interval 1 minimizes rollback at significant disk-I/O cost. Synthetic interruption tests demonstrate identical final parameters and no duplicated effective rows/steps, including interruptions during intermediate and final validation. Resume finishes any pending validation before advancing; a run is complete only after final validation is durable. Evaluation-only sessions cannot train or save inconsistent model/optimizer checkpoints; use explicit resume to continue training.

Baseline validation is at episodes 0/50/100/150/200, with one pipeline per validation question, no gradients and isolated/restored training RNG. Highest validation numeric EM wins; earliest episode breaks ties. Load the selected path in best.json to explicitly validate/test after training. Final testing is a separate operation and never selects checkpoints. Standalone evaluation and full-test costs are included separately in summaries.

`compare_runs([...])` rejects incompatible protocols/data/model/runtime identities. `plot_comparison(RUNS)` displays all eleven required figures from real logs. It can save figures; unavailable metrics have no fabricated points. No synthetic benchmark curves were saved.

## Verification actually performed

- Notebook JSON, Python AST and formal nbformat validation passed.
- Clean Jupyter kernel execution passed with the CUDA virtual environment. Setup performed no training or test operation. All 72 notebook-local synthetic checks passed; the exact final count/list and notebook checksum are in [phase2_clean_kernel.json](audit_artifacts/phase2_clean_kernel.json) and [phase2_synthetic_checks.json](audit_artifacts/phase2_synthetic_checks.json). Installed nbformat 5.11.1 and nbclient 0.11.0 in the existing virtual environment and registered its AutoDSPy GPU kernel. Final CRLF-to-LF serialization cleanup left the parsed notebook unchanged; execution and final serialization checksums are both retained.
- Tests cover numeric negatives/decimals/commas and false substring matches, canonical GSM8K targets, finite fractional judges/failures, centered rewards, Figure 2 loss/gradients at all K, replay-versus-retained gradients, intended parameter updates, equal groups, stop probabilities, independent initialization, splits/schedule prefixes/repeats, RNG restoration, selected-policy evaluation, checkpoint overwrite/config guards, durable interruption recovery, logs, comparison and eleven empty-data plots without invented results.
- Actual official data loading and default 200/50/200 selection checks passed. All four K settings produced identical selection identities. Cached tokenizer collisions match the audit.
- A real pretrained GPT-2 CUDA check used a synthetic K=5 reward group and the production update function. It produced loss 0.623698, gradient norm 137.126492, changed parameters through Adam and used approximately 2.16 GiB peak allocated / 2.42 GiB reserved PyTorch GPU memory. This is a correctness check, not benchmark learning. See [phase2_full_model_gradient.json](audit_artifacts/phase2_full_model_gradient.json).
- A live reduced engineering run used the actual GPT-2 policy, DSPy and Llama 3.1 8B Q4_K_M. Settings: two training examples, one episode, K=5, one validation example, one test example, 128 output tokens, context 2048 and 12 Ollama GPU layers. This uses neither a smaller policy nor a smaller execution model. Training attempted exactly five pipelines: two returned outputs and three failed after output truncation/adapter parsing. Ten logical training LM calls reported 2,377 prompt and 920 generated tokens. Training episode time was 129.86 seconds; the complete driver including setup/evaluations/plot saving took 234.47 seconds. These timings are scoped measurements, not estimates for a baseline.
- That live group had rewards all zero, loss/gradient norm zero, one preserved finite Adam step, one encountered training question and no OOM. It therefore did **not** demonstrate learning. One-question validation/test EM was zero and is not a useful benchmark estimate. All failure details remain recorded.
- Live immutable checkpoint creation, selected-checkpoint restoration, standalone validation, explicit testing, artifact comparison and eleven saved real-data plots succeeded. Final summary reconciliation counts 18 logical LM calls including the separately invoked validation; the initial driver snapshot counted 16 before that accounting correction. Physical HTTP counts remain unavailable. Live recovery was not induced; the meaningful nonzero recovery test is synthetic.
- PyTorch peak allocated/reserved memory in the live episode was approximately 2.34/2.61 GiB. Ollama separately reported approximately 1.96 GiB VRAM residency with 12 GPU layers/context 2048. These are separate allocators, not a summed reliable device peak. The model was unloaded after verification to release test resources.
- Graph update succeeded: 732 nodes, 728 edges, 64 communities, zero LLM token/API cost. It explicitly skipped notebook/PDF content; the primary evidence remains notebook execution and the source audit. Community labels were partly refreshed from hubs; semantic label refresh was not invoked.

## Corrections and remaining limitations

Evaluation now receives the intended policy explicitly, has no undefined value-head dependency, and uses numeric exact match rather than substring scoring. Numeric normalization and strict finite judge parsing correct mis-scoring; these can change rewards relative to the buggy released code. Fractional semantic fallback remains the released Llama judge, with no extra shaping or embedding replacement. Failures receive logged zero rewards and can bias groups.

The original first-token action selection remains, including fifteen signatures but only nine distinct first-token IDs. This limits within-collision discrimination. Generated pipelines still contain one CoT/Predict stage followed by stop. Last-number extraction can still confuse reasoning with the final answer; signatures such as summary remain even though they are not naturally suited to math.

Deterministic decoding, cache-off, zero provider retries, immutable revisions, CPU/partial offload controls and strict runtime compatibility are documented project reproducibility choices. Adam foreach=False reduces temporary memory without changing its mathematical update. Long policy inputs retain their suffix through left truncation, a documented boundary behavior beyond the original tiny examples.

The default CPU-only Ollama offload option is exposed but was not tested with a live generation; the live run used 12 GPU layers because free host RAM was limited. Larger output budgets and the default 4096 context were not validated by that reduced run. GPU co-residency, RAM paging, long runtime and large optimizer checkpoints remain practical concerns on 6GB VRAM/16GB RAM. Physical HTTP requests, omitted usage and complete external wall time are unavailable rather than invented. Remote inference can change across devices/server versions despite fixed seeds. Native Jupyter emitted Windows event-loop/transport warnings, but execution passed.

Paper-versus-appendix episode/fallback inconsistencies remain as audited; no published score reproduction is claimed. Full smoke/pilot/main workloads, live training at K=8/10/15, full official evaluation, CPU-only execution-LM generation and live interruption recovery were not run. Future Adaptive-K remains outside scope.

## Phase boundary

Phase 2 is complete for implementation and the reported verification scope. Preserve this GSM8K milestone. Stop here until the user authorizes Phase 3, then implement only the independent HotpotQA notebook using the master and audited task-formulation decisions. Phase 4 will consolidate cross-dataset checks.
