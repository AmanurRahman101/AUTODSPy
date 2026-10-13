# AutoDSPy Static-K Research — Master Specification

Source: the user's master specification in this session, preserved on 2026-10-11 (Asia/Dhaka). Markdown formatting, escaped line breaks, and equations have been normalized for readability; requirements, qualifications, notes, and phase instructions are retained. The final phase and permission instructions at the end govern the order of execution. This document is the primary source of truth for subsequent sessions.

## Role

Act as a Senior Machine Learning Research Engineer specializing in PyTorch, DSPy, reinforcement learning, GRPO, Jupyter notebooks, experimental reproducibility, and GPU-constrained machine learning.

You are working inside my existing AutoDSPy repository.

Repository: Current;
Branch: experiment/static-k

Original implementation: [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb) (this is a toy setup do not fully follow the paper)

Research paper: [AutoDSPy.pdf](../AutoDSPy.pdf)

## 1. Research Objective

I am conducting research on the **limitations of static group-size selection (K) in GRPO**, using the AutoDSPy framework.

My research consists of two stages:

**Stage 1 — Static-K Baseline (Current Task)**

Train the GPT-2 policy network using different fixed GRPO group sizes:

- K = 5
- K = 8
- K = 10
- K = 15

Conduct these experiments separately on GSM8K and HotpotQA.

Collect performance, learning, and computational statistics for every K.

**Stage 2 — Adaptive-K GRPO (Future Research)**

Develop an adaptive group-size selection algorithm and compare it against the static-K baselines.

**Do not implement Adaptive-K in this task.**

The objective now is to build a reliable, reproducible, properly organized static-K experimental environment.

## 2. Critical Requirement — Notebook-Centric Implementation

I want two independent Jupyter notebooks:

1. `Static_K_GRPO_GSM8K.ipynb`
2. `Static_K_GRPO_HotpotQA.ipynb`

Keep the original [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb) unchanged.

Each notebook must contain its own complete experimental workflow:

- Dependency installation and verification
- Environment initialization
- Ollama connection and model configuration
- Dataset loading and preparation
- GPT-2 policy-network initialization
- DSPy action-space definition
- Pipeline generation
- Pipeline execution
- Reward calculation
- GRPO loss
- Training
- Checkpointing and resuming
- Validation
- Final testing
- Experimental statistics
- Visualization and comparison

Follow [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb) structure and initializations to create these new two notebooks.

**Everything required to run the experiment must be inside the corresponding notebook.**

Do not create separate Python implementation modules such as:

- `__init__.py`
- `dataset.py`
- `policy.py`
- `pipeline.py`
- `trainer.py`
- `reward.py`
- `evaluation.py`
- `metrics.py`

Do not introduce a complicated software architecture.

The notebooks should follow the original [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb) Py_GRPO.ipynb organization, with clearer Markdown sections and additional research functionality.

Duplicating necessary functionality across the two independent notebooks is acceptable.

Do not require one notebook to import or execute code from the other.

External files are permitted only for artifacts such as datasets/cache metadata, logs, checkpoints, plots, experiment results, and documentation.

## 3. Preserve AutoDSPy's GRPO Methodology

Read Figure 2 and the relevant methodology sections of the [AutoDSPy.pdf](../AutoDSPy.pdf) paper.

Inspect the original `DSPy_GRPO.ipynb` implementation before making changes.

The paper's Figure 2 defines the group-relative policy-gradient loss as follows.

**Group mean reward**

\[
\bar r=\frac{1}{K}\sum_{i=1}^{K}r_i
\]

**Group-relative advantage**

\[
A_i=r_i-\bar r
\]

**GRPO policy loss**

\[
\mathcal L_{\mathrm{GRPO}}=-\sum_{i=1}^{K}\sum_{t=1}^{T_i}\log\pi_\theta(a_{i,t}\mid s_{i,t})A_i
\]

Use this formulation as the primary training objective. [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb) implemented these accurately as far I know.

Do not replace it with a different textbook GRPO implementation.

Do not introduce:

- PPO-style clipping
- Entropy regularization
- Standard-deviation-normalized advantages
- GAE
- KL penalties
- Additional reward shaping
- K-normalized loss
- ReAct
- Teleprompters
- Adaptive-K mechanisms

The paper reports certain PPO-related hyperparameters elsewhere. Do not automatically incorporate them into the Figure 2 GRPO loss.

Preserve the mathematical behavior of the released implementation while correcting software errors that prevent proper training or evaluation.

**Important:** Since the loss sums over K candidate pipelines, increasing K may change gradient magnitudes. Do not silently normalize the loss by K. Instead, record gradient norms and explicitly document this limitation for later analysis.

Note: Figure 2 can be found in [Figure_2 (F)-1.png](<../assets/Figure_2 (F)-1.png>).

## 4. Correct Interpretation of Training Episodes

This section is particularly important.

[Figure_2 (F)-1.png](<../assets/Figure_2 (F)-1.png>) describes one training iteration as randomly selecting a single prompt–ground-truth pair from the training dataset.

The original notebook follows a one-prompt-per-episode training loop.

The paper's appendix contains potentially conflicting wording about samples per episode. Audit this discrepancy and document it rather than assuming every episode processes the entire dataset.

For this project, explicitly use the following training protocol:

**Training dataset pool: 200 examples**

**Total training episodes: 200**

**Samples selected per episode: 1**

**Sampling method: Random sampling with replacement**

**Candidate pipelines per episode: K**

**Policy updates: At most one per episode, consistent with the original training procedure**

One episode must execute the following:

1. Randomly select one prompt–ground-truth pair `(x, y)` from the 200-example training pool.
2. Generate K candidate DSPy pipelines using GPT-2.
3. Execute those K pipelines using Llama 3.1 8B.
4. Compare outputs against the ground-truth answer.
5. Calculate the rewards.
6. Compute the mean group reward.
7. Calculate group-centered advantages.
8. Calculate the [Figure_2 (F)-1.png](<../assets/Figure_2 (F)-1.png>) GRPO loss.
9. Perform the corresponding policy update when valid.
10. Record statistics and proceed to the next episode.

Random sampling with replacement means the same question may be selected multiple times across different episodes.

Therefore:

**200 episodes do not necessarily mean 200 unique training questions were encountered.**

Record the actual number of unique questions sampled.

### Consistent sampling across K

For a scientifically controlled static-K comparison, generate the random episode-to-question selection sequence once using a fixed seed.

Persist this sequence.

All four K experiments must use exactly the same episode-to-question sequence.

For example:

- Episode 1: Question 47
- Episode 2: Question 12
- Episode 3: Question 47
- Episode 4: Question 189
- ...

K=5, K=8, K=10, and K=15 must encounter the same question at the same episode number.

Do not independently generate a different training-question sequence for each K.

Note: make the code cleaner in this section, so I can change dataset size, episodes with my choice later.

## 5. Main Experiment Configuration

Use the following initial configuration in both notebooks.

- Policy network: GPT-2 127M
- Execution LLM: Llama 3.1 8B via Ollama
- RL algorithm: AutoDSPy Figure 2 GRPO
- Dataset: GSM8K / HotpotQA
- Training pool size: 200
- Training episodes: 200
- Samples per episode: 1
- Sampling: Random with replacement
- Group sizes: 5, 8, 10, 15
- Learning rate: 1e-4
- Optimizer: Adam, matching released notebook
- Random seed: 42
- Validation questions: 50
- Preliminary test questions: 200
- GRPO loss: [Figure_2 (F)-1.png](<../assets/Figure_2 (F)-1.png>) / implemented in [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb)
- PPO clipping: Disabled
- Entropy regularization: Disabled

These settings define our controlled research experiment.

Verify relevant paper settings against the original publication and code before implementing them.

Clearly document any settings that are project-specific rather than directly reproduced from the paper.

Note: make the code clean so that I can easily change training pool size, episode size, validation size and test question size or Execution LLM model. (I might change those number depending on my hardware constrain.) Look into the [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb) for inspiration.

## 6. Preserve the Original DSPy Action Space

Keep the original AutoDSPy module pool:

`MODULES = ["CoT", "Predict"]`

Preserve the original signature pool and the intended pipeline-generation process.

Do not add ReAct or other DSPy modules.

Maintain the responsibilities of the existing notebook functions, including:

- `generate_pipeline()`
- `execute_pipeline()`
- `compare_answers()`
- `compute_reward()`
- `train_grpo()`
- `test_model()`
- `evaluate_module()`

You may reorganize their implementations into cleaner notebook cells, but preserve the original experimental logic.

Audit the existing first-token action-scoring approximation.

Check whether different action strings can map to the same first token and whether the calculated log probabilities correspond to the sampled actions.

Preserve the released action-selection behavior in the primary baseline unless a genuine correctness failure requires a change.

Document any correction and explain its methodological impact.

## 7. Notebook A — GSM8K

Create:

`Static_K_GRPO_GSM8K.ipynb`

Use the official GSM8K dataset, `main` configuration.

### Dataset allocation

**Training pool**

Select 200 questions from the official training split. (Cleaner code so I can change it later on my own.)

**Validation set**

Select 50 additional questions from the official training split, with no overlap with the training pool. (Cleaner code so I can change it later on my own.)

**Preliminary test set**

Select 200 questions from the official test split. (Cleaner code so I can change it later on my own.)

Preserve access to the complete official test split for later research evaluation.

Use reproducible sampling with seed 42.

Persist selected question indices and identifiers.

Do not allow validation or test examples to enter policy training.

### GSM8K reward and evaluation

Preserve [AutoDSPy.pdf](../AutoDSPy.pdf) original reward mechanism as closely as possible, including exact-match evaluation and semantic fallback when applicable.

Implement reliable numerical answer normalization for GSM8K.

Support:

- Integers
- Negative numbers
- Decimal numbers
- Comma-separated numbers

Use canonical ground-truth answers.

Report numeric exact-match accuracy separately from semantic fallback results.

Do not alter the main training reward without documenting the change.

## 8. Notebook B — HotpotQA

Create:

`Static_K_GRPO_HotpotQA.ipynb`

Use the official HotpotQA dataset.

Verify and document the selected benchmark configuration, preferably `distractor` if consistent with the paper's experimental setup.

Inspect how the original AutoDSPy implementation provides HotpotQA questions and contextual information to the execution LLM.

Preserve the paper's actual task formulation whenever it can be verified.

Do not introduce external search or retrieval tools unless those tools are confirmed to have been used in the target experiment.

### Dataset allocation

**Training pool**

Select 200 questions from the official HotpotQA training split. (Cleaner code so I can change it later on my own.)

**Validation set**

Select 50 different examples from the training split. (Cleaner code so I can change it later on my own.)

**Preliminary test set**

Select 200 examples from a disjoint, publicly labeled development/validation split.

HotpotQA's official test annotations are not generally available for local evaluation. Do not attempt to score unlabeled official test examples using fabricated ground truth.

Preserve question IDs, context, and available evaluation metadata.

Avoid leakage of gold answers or gold supporting-fact annotations into model inputs.

Or you can get training dataset for testing. Suppose if I take 200 training split use 201-400 dataset for testing splits. (Cleaner code so I can change the number of testing set later on my own.)

### HotpotQA reward and evaluation

HotpotQA uses textual answers.

Implement suitable evaluation:

- Normalized exact match
- Token-level F1
- Training reward
- Semantic fallback, if used by the original implementation

Keep training reward and evaluation metrics separate.

Use identical answer extraction and evaluation protocols across all K experiments.

Do not report supporting-fact accuracy unless the model actually produces supporting-fact predictions.

## 9. Experiment Presets

Both notebooks must support three workload presets.

### Smoke Test

- Training pool: 200 examples
- Episodes: 5
- Purpose: Check execution and training correctness

### Pilot Experiment

- Training pool: 200 examples
- Episodes: 20
- Purpose: Estimate runtime and inspect preliminary statistics

### Main Static-K Baseline

- Training pool: 200 examples
- Episodes: 200
- Purpose: Main experimental comparison

All presets should use the same persisted 200-example training pool.

The 5-episode smoke-test question sequence and 20-episode pilot question sequence should be prefixes of the 200-episode sequence for the same seed.

However, smoke-test, pilot, and baseline runs must use separate run identifiers and checkpoint namespaces unless a run is intentionally being resumed.

Do not accidentally resume a completed pilot as if it were a baseline without validating its configuration and checkpoint compatibility.

(You can alter the smoke test or pilot test training set depending on my VRAM or RAM limitation while running the test by yourself.)

## 10. Independent K Experiments

Conduct separate experiments for:

- K=5
- K=8
- K=10
- K=15

Each experiment must start with identical pretrained GPT-2 parameters and a fresh optimizer.

Do not fine-tune K=8 using a K=5 checkpoint.

Keep all other parameters unchanged.

The main experiment's candidate pipeline budgets are:

- K=5: 200 episodes; 1,000 pipeline evaluations
- K=8: 200 episodes; 1,600 pipeline evaluations
- K=10: 200 episodes; 2,000 pipeline evaluations
- K=15: 200 episodes; 3,000 pipeline evaluations

Total: 7,600 candidate pipeline evaluations per dataset, assuming every candidate completes.

These counts do not necessarily equal the number of execution-LLM calls because some DSPy pipelines may contain multiple stages.

Measure actual LLM calls and token usage separately when possible.

## 11. Training Correctness

Inspect the current [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb) code carefully.

Verify that:

1. GPT-2 is initialized correctly.
2. The intended policy model receives gradients.
3. Policy action probabilities correspond to sampled actions.
4. The correct prompt and ground truth are used for every episode.
5. Exactly K candidate pipelines are attempted per episode.
6. Each candidate receives the appropriate reward.
7. The group baseline is calculated using the K rewards.
8. Advantages equal reward minus the group mean.
9. [Figure_2 (F)-1.png](<../assets/Figure_2 (F)-1.png>) loss is computed correctly.
10. Backpropagation updates the intended parameters.
11. Equal-reward groups are handled safely.
12. Failed pipeline executions are logged consistently.
13. No unexpected checkpoint is automatically loaded.
14. The correct trained policy is used during evaluation.
15. Training does not accumulate unnecessary GPU computation graphs.

If efficient implementation requires recomputing differentiable action log probabilities after sampling, ensure the policy parameters remain unchanged between rollout collection and gradient calculation.

Explain any correctness-related changes compared with the original notebook.

Do not silently change the research methodology.

## 12. Checkpointing and Resuming

Implement checkpoint saving and restoration inside each notebook.

Every checkpoint should record:

- Policy model state
- Optimizer state
- Dataset
- K
- Training episode
- Learning rate
- Experiment configuration
- Random seed
- Python, NumPy, and PyTorch RNG states
- Training-question selection sequence or identifier
- Training metrics
- Execution-model configuration

Support:

- Starting a new experiment
- Saving a checkpoint
- Resuming interrupted training
- Loading a selected checkpoint for evaluation
- Comparing multiple completed runs

Checkpointing must be safe against accidental overwrites.

Use separate checkpoints and logs for each dataset, K, seed, and experiment preset.

A resumed run must continue from the next unfinished episode without duplicating optimizer updates or statistics.

Validate configuration compatibility before resuming.

Save periodic checkpoints, with a configurable checkpoint interval.

## 13. Validation and Testing

Do not evaluate on the held-out test data during training.

### Validation

Use 50 fixed validation questions per dataset.

For the 200-episode baseline, evaluate at:

- Episode 0
- Episode 50
- Episode 100
- Episode 150
- Episode 200

Allow configurable validation intervals for smoke and pilot experiments.

Validation should generate one policy-selected pipeline per question, rather than K candidate pipelines.

Make inference sampling and decoding reproducible as far as the execution model permits.

Do not update GPT-2 during validation.

Save validation accuracy and other benchmark-specific metrics.

Select the best checkpoint using validation performance with a documented tie-breaking rule.

### Final Testing

Final testing should be explicitly triggered after training.

GSM8K: Use the selected 200-question official test subset.

HotpotQA: Use the selected 200-question labeled development subset.

Every K experiment must use the same corresponding held-out evaluation questions.

Do not use final test scores to choose checkpoints, tune parameters, or select the most favorable random seed.

Preserve the ability to run broader official evaluations later.

## 14. Research Statistics

Implement structured logging inside both notebooks.

Record statistics after every training episode.

### Training statistics

- Dataset
- K value
- Random seed
- Episode number
- Selected question ID
- Generated pipeline actions
- Generated pipeline signatures
- Individual candidate rewards
- Mean group reward
- Group reward standard deviation
- Group-relative advantages
- Zero-advantage group indicator
- GRPO policy loss
- Gradient norm
- Optimizer update count
- Unique training questions encountered

### Computational statistics

- Attempted pipeline evaluations
- Successful pipeline evaluations
- Failed pipeline evaluations
- Execution-LLM calls
- Prompt tokens, if available
- Generated tokens, if available
- Time per episode
- Cumulative training time
- Average pipeline-execution latency
- Peak allocated GPU memory
- Peak reserved GPU memory
- Ollama memory information, if available
- Out-of-memory and execution failures

### Evaluation statistics

- Validation exact-match accuracy
- Final held-out accuracy
- HotpotQA token-level F1
- Training mean reward
- Reward variance
- Zero-advantage group percentage
- Accuracy progression across checkpoints
- Accuracy relative to cumulative pipeline evaluations
- Total computation used by each K

If a metric cannot be measured reliably, record it as unavailable rather than inventing a value.

Store logs in CSV or JSONL format.

Maintain a compact experiment summary for each completed run.

## 15. Visualizations

Generate research visualizations directly inside the notebooks using Matplotlib.

Required plots:

1. K versus validation accuracy
2. K versus final held-out accuracy
3. K versus training time
4. K versus pipeline evaluations
5. Average reward versus training episode
6. Zero-advantage group percentage versus K
7. Gradient norm versus training episode
8. Validation accuracy versus checkpoint episode
9. Accuracy versus cumulative pipeline evaluations
10. Total LLM calls versus K
11. Execution latency versus K

For HotpotQA, also produce an exact-match versus F1 comparison.

All plots must display inside the corresponding notebook.

Optionally save the plots to the experiment results directory.

Do not create graphs containing fake experimental results.

## 16. Simple Notebook Execution Interface

Both notebooks must have a clearly labeled configuration and execution section.

Support these experiment controls:

- `K`
- `SEED`
- `PRESET`
- `MODE`
- `NUM_EPISODES`
- `TRAIN_POOL_SIZE`
- `VALIDATION_SIZE`
- `TEST_SIZE`
- `OLLAMA_MODEL`
- `OLLAMA_HOST`
- `CHECKPOINT_INTERVAL`

Supported modes should include:

- `initialize`
- `train`
- `resume`
- `validate`
- `test`
- `compare`

For example, I should be able to:

1. Open the GSM8K notebook.
2. Run the initialization cells.
3. Set K=5 and PRESET="smoke".
4. Run five training episodes.
5. Inspect the training statistics.
6. Switch to K=8 and repeat.
7. Load a saved checkpoint.
8. Run validation or testing.
9. Generate comparison plots.

Provide an explicitly invoked option to run all four K configurations sequentially.

**Do not automatically start training when the notebook is opened or when setup cells are executed.**

Training, testing, and comparison operations must be clearly separated.

The notebook should provide clear error messages when models, checkpoints, datasets, or Ollama services are unavailable.

## 17. Notebook Organization

Both notebooks must contain clearly numbered Markdown headings.

Use this structure:

1. Research Objective and Experiment Overview
2. Environment and Dependency Setup
3. Experiment Configuration
4. Hardware and Ollama Initialization
5. Dataset Loading
6. Dataset Sampling and Episode Schedule
7. DSPy Action Space and Signatures
8. GPT-2 Policy Network Initialization
9. Policy Action Generation
10. DSPy Pipeline Generation
11. DSPy Pipeline Execution
12. Reward Calculation
13. Figure 2 GRPO Loss
14. GRPO Training Loop
15. Checkpoint Management
16. Validation
17. Final Testing
18. Metrics and Statistics
19. Static-K Comparison
20. Visualizations
21. Reproducibility and Experiment Summary

Include concise Markdown explanations before significant code cells.

Keep the notebook readable and straightforward for academic presentation.

Avoid hidden dependencies between notebook cells.

Ensure the notebook can be initialized from a fresh Jupyter kernel.

## 18. Hardware Constraints

My laptop GPU is:

**NVIDIA RTX 3060 Laptop GPU — 6GB VRAM**

I want to retain:

- GPT-2 127M for policy training
- Llama 3.1 8B through Ollama for DSPy execution

Do not automatically substitute smaller models.

Implement practical resource optimizations:

- Sequential pipeline execution
- Avoiding unnecessary GPU-resident model copies
- Configurable Ollama execution endpoint
- Supporting CPU offloading for the execution model
- Controlling unnecessary activation retention
- GPU-memory measurement
- Periodic checkpointing
- Reliable interruption recovery
- Avoiding unnecessary repeated model initialization

Track Ollama runtime and quantization details.

Any hardware-induced methodological deviation must be documented.

Do not change the [Figure_2 (F)-1.png](<../assets/Figure_2 (F)-1.png>) GRPO objective to reduce VRAM consumption.

For testing on your own you can reduce the training size, episode size or testing or validation size on your own. So that you can successfully test them without taking so much time.

## 19. Suggested Repository Structure

Keep the repository minimal.

Suggested layout:

- `DSPy_GRPO.ipynb` — original, unchanged
- `Static_K_GRPO_GSM8K.ipynb` — complete GSM8K experiment
- `Static_K_GRPO_HotpotQA.ipynb` — complete HotpotQA experiment
- `STATIC_K_EXPERIMENT_GUIDE.md` — setup and usage
- `results/gsm8k/k5/`
- `results/gsm8k/k8/`
- `results/gsm8k/k10/`
- `results/gsm8k/k15/`
- `results/hotpotqa/k5/`
- `results/hotpotqa/k8/`
- `results/hotpotqa/k10/`
- `results/hotpotqa/k15/`

Use nested seed and run-specific directories where necessary.

Results directories should contain only experimental artifacts such as configuration snapshots, metrics, checkpoints, and plots.

Do not create a separate Python package or unnecessary framework.

## 20. Verification and Testing

Include lightweight self-checks inside each notebook.

Test the following:

- Dependency availability
- Ollama configuration
- Dataset subset separation
- Reproducible episode sampling
- Sampling with replacement
- Identical episode schedules across K
- Correct group mean reward
- Correct group-relative advantages
- Correct Figure 2 loss
- Gradient propagation
- K=5, 8, 10, 15 support
- Equal-reward groups
- Model reinitialization
- Checkpoint saving and loading
- Interruption-safe resuming
- Correct trained-model selection during evaluation
- Metric logging
- Plot generation

Use small synthetic inputs for tests that do not require Llama inference.

Where possible, validate notebook execution from a clean kernel.

Do not launch full 200-episode training runs automatically.

Only execute lightweight checks and feasible smoke tests during implementation.

## 21. Research Integrity and Documentation

Clearly distinguish:

1. AutoDSPy's published methodology
2. The released notebook's actual implementation
3. Corrections required for software correctness
4. Our reduced-budget experiments
5. Hardware-related adjustments
6. Future Adaptive-K GRPO research

The paper contains potentially inconsistent descriptions of training samples per episode.

Document these ambiguities.

For this project, explicitly follow the one-random-prompt-per-episode procedure supported by Figure 2 and the original notebook.

Do not falsely claim that the reduced experiments reproduce the paper's published benchmark scores.

Keep the original Figure 2 loss unchanged in the primary baselines.

Do not fabricate benchmark results, hardware measurements, or successful test outcomes.

Preserve existing repository work and uncommitted changes.

## 22. Implementation Workflow

Proceed autonomously through the following stages.

**Stage A — Inspection**

Inspect the repository, original notebook, and AutoDSPy paper.

Identify required code changes.

**Stage B — GSM8K Notebook**

Create the complete, self-contained GSM8K static-K GRPO notebook.

**Stage C — HotpotQA Notebook**

Create the equivalent HotpotQA notebook with appropriate dataset handling and textual evaluation metrics.

**Stage D — Experiment Controls**

Implement reproducible episode schedules, configuration presets, training, checkpointing, resuming, validation, and testing.

**Stage E — Research Metrics**

Implement statistics collection, experiment summaries, and visualizations.

**Stage F — Verification**

Run unit-style checks and feasible lightweight smoke tests.

Do not start expensive baseline runs.

**Stage G — Documentation**

Create `STATIC_K_EXPERIMENT_GUIDE.md`, covering:

- Environment setup
- Ollama configuration
- Notebook initialization
- GSM8K training
- HotpotQA training
- Switching K values
- Smoke, pilot, and baseline presets
- Checkpoint saving
- Resuming training
- Validation
- Final testing
- Plot generation
- GPU memory monitoring
- Reproducibility limitations

Do not stop after producing a plan. Proceed to implementation and validation.

## 23. Final Report

When implementation is complete, explain:

1. Which files were created and modified.
2. How each notebook is organized.
3. How to initialize GPT-2 and Ollama.
4. How to start a K=5 smoke test.
5. How to train K=8, K=10, and K=15.
6. How 200 episodes and 200 training samples are handled.
7. How the identical episode-sampling schedule is enforced.
8. How to save and resume experiments.
9. How to validate and test the policy.
10. How to generate comparisons and visualizations.
11. Which verification tests actually passed.
12. Which differences remain relative to the original AutoDSPy implementation.
13. Known issues and hardware limitations.

**Final success criterion:**

Produce two understandable, independent, fully organized Jupyter notebooks that implement AutoDSPy's Figure 2 GRPO methodology and support reproducible Static-K experiments with K=5, 8, 10, and 15.

All model initialization, pipeline generation, execution, training, validation, testing, statistics, and visualization functionality must be available directly within each notebook.

Do not implement Adaptive-K yet.

Do not run the expensive full experiments automatically. For test use smaller runs to verify; keep the full experiment or training run for me.

For the reports you can create Markdown file so I can review what you did.

## Incremental execution instruction

### Treat the above prompt as the master specification for this research project.

Do not attempt to implement everything in a single pass.

Work incrementally in four phases:

1. Audit the original AutoDSPy notebook and research paper.
2. Implement and validate the complete GSM8K notebook.
3. Implement and validate the independent HotpotQA notebook.
4. Finalize experiment management, statistics, visualizations, and reproducibility checks.

For each phase:

- Implement only the relevant requirements.
- Preserve previously validated functionality.
- Run appropriate tests.
- Report changes, tests performed, and unresolved issues.
- Keep all executable research logic inside the corresponding notebooks.

## Autonomous Planning and Documentation

You have permission to create, organize, and maintain any necessary Markdown files, research notes, implementation plans, task trackers, or technical documentation that will help you complete the project correctly across multiple sessions.

Use the master prompt to derive your implementation plan, technical requirements, phase milestones, and acceptance criteria.

You may create a dedicated `docs/` directory for these documents.

At minimum, maintain:

- `MASTER_SPEC.md` — A faithful, complete copy of the master specification. Preserve its requirements without silently changing or omitting them.
- `IMPLEMENTATION_PLAN.md` — A phased implementation plan derived from the master specification.
- `PROGRESS.md` — Completed tasks, remaining tasks, current phase, validation status, and blockers.
- `DECISIONS.md` — Important implementation decisions, research assumptions, methodological deviations, and reasons for each decision.

You may create additional documents when genuinely necessary. Avoid redundant documentation and unnecessary complexity.

## Maintaining Project Context

The master specification must remain the primary source of truth.

At the beginning of every new phase or resumed session:

1. Read the master specification.
2. Read the latest implementation plan and progress tracker.
3. Review relevant decisions and previous validation results.
4. Inspect the current repository state.
5. Identify completed work, pending tasks, and unresolved issues.
6. Continue from the last verified milestone rather than starting over.

Update the progress tracker and relevant documentation whenever meaningful implementation work is completed.

If your implementation plan conflicts with the master specification, prioritize the master specification and document the conflict.

Do not silently change research methodology or experimental requirements.

## Implementation Rules

- Keep all executable research logic inside the corresponding Jupyter notebooks.
- Planning and documentation may be stored in separate Markdown files.
- Preserve the original `DSPy_GRPO.ipynb` unchanged.
- Preserve previously validated functionality.
- Respect any existing `AGENTS.md` instructions.
- Do not overwrite existing documentation or uncommitted work without reviewing it.
- Do not introduce unnecessary Python modules or architectural complexity.
- Use the original AutoDSPy methodology as required by the master specification.

## Phase Execution

For each phase:

1. Review its requirements and dependencies.
2. Create or update the relevant implementation checklist.
3. Implement the required functionality.
4. Run appropriate verification tests.
5. Document changes, test outcomes, unresolved issues, and deviations.
6. Update `PROGRESS.md` and the implementation plan.
7. Clearly mark the phase as completed, partially completed, or blocked based on actual verification.

Do not mark a phase as completed merely because its code has been written.

Work on **one phase at a time**. After completing a phase, provide a concise summary and stop at the phase boundary. Proceed to the next phase when instructed.

Do not launch expensive training experiments automatically.

## Important

You are authorized to make reasonable implementation decisions independently, create necessary documentation, and organize the work without asking for confirmation on routine technical details.

However, preserve all scientific and methodological constraints from the master specification.

Your documentation must make it possible for a new Codex session to understand the current project state and continue the remaining phases without requiring me to paste the entire master prompt again.

**Begin with Phase 1: Audit.** First preserve the master specification in the repository, establish the phased implementation plan and progress tracker, then perform the research audit. Do not implement the GSM8K or HotpotQA notebooks during Phase 1.

Ask permission to run Phase 1 after creating your necessary documents.
