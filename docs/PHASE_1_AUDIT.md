# Phase 1 — Original AutoDSPy research audit

Completed: 2026-10-11 (Asia/Dhaka), on `experiment/static-k`, following explicit user authorization to begin Phase 1.

**Conclusion:** the original notebook implements the requested Figure 2 group-centered, summed policy loss correctly. It is a usable methodological starting point, but its evaluation, answer comparison, initialization, and experiment management need corrections before controlled Static-K experiments. The first-token action approximation is mathematically consistent with its sampler, but restricts what the policy can learn. Preserve that approximation in the primary baseline and document its limitations.

No experiment notebook was created or modified. No real Llama inference, dataset download, pretrained GPT-2 training, or baseline experiment ran. Tests below used isolated original function definitions, small synthetic policies, cached GPT-2 tokenization, and read-only environment probes.

## 1. Scope and evidence conventions

Authority: [MASTER_SPEC.md](MASTER_SPEC.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), and the repository [AGENTS.md](../AGENTS.md). Source implementation: [DSPy_GRPO.ipynb](../DSPy_GRPO.ipynb). Publication: [AutoDSPy.pdf](../AutoDSPy.pdf), including the supplied [Figure 2 image](<../assets/Figure_2 (F)-1.png>).

Notebook cell numbers below are **zero-based JSON cell indices**; in-cell line numbers are one-based. This notebook has 31 cells. PDF page numbers are one-based file pages; printed proceedings pages run from 2881 to 2896. Methodology and experimental passages were extracted from the PDF, and Figure 2, Table 3, and Appendix A.7 were also visually reviewed in rendered pages. The standalone Figure 2 image agrees with the PDF figure.

The graphify skill was reviewed and graph availability was checked. `graphify-out/graph.json` does not exist in this workspace. Therefore there is no existing graph to query; direct source inspection was used. No graph output was invented or rebuilt. No executable repository code changed, so the AGENTS.md post-code-change graph update rule was not triggered.

Existing `run_grpo_baseline.py` and `BASELINE_RESULTS.md` were inspected for contextual risks and left unchanged. Existing claimed experimental results were not verified or reused as new evidence.

## 2. Published methodology, released code, and project protocol

### Verified common core

PDF page 4 / printed page 2884, Figure 2: one randomly selected prompt–ground-truth pair generates a group of candidate pipelines, followed by rewards, group-centered advantages, a summed policy loss, and backpropagation. Inference generates one pipeline.

Notebook cell 22 implements:

- `idx = np.random.randint(len(prompts))` once per episode, followed by the matching prompt and ground truth (line 12 onward).
- A sequential `for _ in range(K)` collection loop for that same pair.
- `group_average = np.mean(rewards_list)` (line 33).
- `advantages = [reward - group_average for reward in rewards_list]` (line 36).
- `loss -= log_prob * advantage` over candidates and their actions (line 44).
- One `loss.backward()` and one Adam `optimizer.step()` per successfully completed episode (lines 47–48).

The loss has **no division by K**, no standard-deviation scaling, clipping, entropy, value network, GAE, KL penalty, or reward shaping. Finite zero-advantage groups have zero loss and zero gradients. Larger K can change gradient magnitudes; recording gradient norms is necessary.

### The episode discrepancy is real

- Figure 2 and its caption explicitly select one pair per iteration.
- Section 4.1, PDF page 5, describes a 200-example training budget.
- Appendix A.1, PDF pages 9–10, varies training pool size and episode count separately.
- Appendix A.2, PDF page 13, describes 200 samples per episode.
- Appendix A.7, PDF page 15, also refers to 200 training samples per episode.
- Released cell 22 processes **one** sampled pair per episode, not the entire dataset.

These descriptions cannot establish a single unambiguous published implementation. Follow the master specification: a configurable 200-example pool, 200 episodes, one pair sampled with replacement per episode, K candidates, at most one optimizer step. Persist one schedule across all K and count observed unique IDs. Do not claim that 200 episodes necessarily expose 200 distinct examples or reproduce a 200-samples-per-episode experiment.

### Which settings come from which source

- Paper Table 3, PDF page 14: learning rate `1e-4`, 200 episodes, default GRPO K=5. Gamma, lambda, clipping epsilon, and entropy coefficient appear in a shared RL hyperparameter table; Figure 2 assigns GAE/clipping/entropy to the separate PPO branch. They must not be imported into this GRPO baseline.
- Paper Appendix A.1, pages 11–12: static K ablations at 5, 10, and 15. **K=8 is a project-specific addition.** The paper's accuracy/latency statements are published claims, not measurements from this repository audit.
- Paper Appendix A.3, page 14: GPT-2 labeled 127M among policy models; Llama 3.1 8B among execution models. The notebook's GPT-2 setup uses checkpoint name `gpt2`; retain this identity and the paper's model label.
- Paper section 4.1, page 5: 200 training samples; 1,300 GSM8K and 1,000 HotpotQA test samples. Our separate datasets, fixed seed 42, 50-question validation sets, preliminary 200-question held-out sets, preset namespaces, and shared schedules are project controls. They are not a claim of exact paper reproduction.
- Notebook cell 13: 10 GSM8K and 10 HotpotQA training rows, combined and shuffled without a seed; no context in the combined training prompts.
- Notebook cell 22 defaults: Adam, learning rate `1e-4`, 500 episodes, K=4. Actual run cells 24 and 25 call two episodes, learning rate `2e-5`, K=4. Running both cells sequentially continues training the same global policy twice.
- Paper Figure 2 includes a broader pipeline/module/teleprompter design. The released GRPO notebook uses one module/signature pair and stop, with CoT/Predict only. Preserve this toy action process, rather than expand it to the full diagram. Table 1 and the appendix describe final experiments without teleprompters, consistent with omitting them here.

## 3. GPT-2 and policy action audit

### Initialization and gradient ownership

Cell 18 initializes `GPT2Tokenizer` and `GPT2LMHeadModel` from `gpt2`, sets the tokenizer padding token to EOS, sets the policy padding ID, and moves the model to CUDA if available. Adam in cell 22 receives that global policy's parameters. Sampling uses `torch.multinomial`; extracting the sampled index with `.item()` is expected for a discrete action. The retained selected log probability remains differentiable.

Isolated tests confirmed gradient propagation through original `generate_pipeline()` to a small supplied policy, and confirmed original `train_grpo()` changes small policy parameters with mixed rewards at all four requested K values. These tests **do not establish pretrained GPT-2 training quality or full-model memory requirements**.

The installed Transformers source sets `from_pretrained()` models to evaluation mode by default. The original notebook never explicitly calls `policy_model.train()`. Evaluation mode disables dropout but does not disable gradients. Therefore do not silently add stochastic dropout when implementing replay/recomputation. Keep the original dropout-disabled policy behavior explicit, or document a deliberate change. Cached GPT-2 configuration confirms 12 layers, width 768, 12 heads, vocabulary 50,257, and a 1,024-token position limit.

### Actual pipeline grammar and signatures

Cell 15 defines `MODULES = ["CoT", "Predict"]`, the 15 signatures below, and stop. Cell 20 selects a module, then a signature, then deterministic stop, with default `max_steps=3`. Each candidate runs **one DSPy program**, not the multistage chain illustrated elsewhere in the paper. Both training and testing strip terminal stop before execution.

Preserve the complete signature pool:

1. `question -> answer`
2. `text -> summary`
3. `question -> reasoning`
4. `question -> hypothesis`
5. `problem -> solution`
6. `problem_description -> explanation`
7. `context -> summary`
8. `context -> briefing`
9. `word_problem -> solution`
10. `math_problem -> answer`
11. `prompt -> response`
12. `query -> response`
13. `text -> response`
14. `prompt -> generated_text`
15. `query -> generated_text`

### First-token scoring: valid sampler, limited expressiveness

For each allowed action, the notebook gathers the GPT-2 logit for its first token, applies softmax over the allowed **action entries**, samples an entry, and records the log of that entry's probability. All 18 action strings tokenize to nonempty sequences. No empty-token/index mismatch occurs for this fixed pool.

Cached GPT-2 tokenization confirmed these collisions among signatures:

- Token 25652: the three `question -> ...` signatures.
- Token 5239: `text -> summary`, `text -> response`.
- Token 45573: `problem -> solution`, `problem_description -> explanation`.
- Token 22866: `context -> summary`, `context -> briefing`.
- Token 16963: `prompt -> response`, `prompt -> generated_text`.
- Token 22766: `query -> response`, `query -> generated_text`.

`word_problem` and `math_problem` start with distinct tokens 4775 and 11018. Modules have distinct first tokens: CoT 7222 and Predict 47. Stop is token 11338.

Thus **15 signature actions have only 9 distinct first-token scores**. Actions in each collision group necessarily have equal probability in the same state; the policy cannot learn to prefer one output-field signature over another in that group. Repeated logits also give larger collision groups more aggregate probability mass. Nevertheless, sampled-entry likelihoods are consistent with the actual constrained categorical sampler; they are not full action-string language-model likelihoods.

The stop choice has probability 1, log probability 0, and no learning signal. There are 30 named module/signature combinations, with more limited independently learnable signature distinctions.

**Primary-baseline decision:** retain this released approximation and its duplicate entries. Replacing it with full-string scores or a new action head would change the policy family, not simply repair an incorrect sampled-index likelihood. Use numerically stable log-softmax for the same distribution in later implementation; do not clamp probabilities in a way that changes the objective.

## 4. Verified correctness and operational issues

### Evaluation can fail or use the wrong policy

Cell 20, `test_model()`, calls `.eval()` on its argument but `generate_pipeline()` reads the global `policy_model`. Passing a separately loaded checkpoint model does not make that model generate the pipeline. A marker-model fixture reproduced this mismatch.

Cells 29–30 refer to undefined `trained_value_head`. Both raise NameError in a clean GRPO environment. Supplying a fake value head does not fix the interface: the HotpotQA call then raises TypeError because it provides `query` twice, while the GSM8K call silently assigns the fourth argument to `query`. GRPO has no value-head requirement.

Correction needed in later notebooks: explicit, consistent model ownership through generation/evaluation; correct GRPO call signatures; restore policy mode and training RNG states around validation. No extra value network should be added.

### Answer matching produces false negatives and false positives

Original cell 20, `compare_answers()` fixtures:

- `[1,234]` versus `1234`: score 0 despite equal numerical values.
- `[5.0]` versus `5`, and `[5]` versus `5.0`: score 0.
- `[1,999]` versus `[1,234]`: score 1 because both bracketed strings are reduced to their first numeric fragment, `1`.
- `Paris` versus `Paris`, and `[Paris]` versus `Paris`: score 0 because unbracketed textual ground truth has no numeric match.
- `[-12.5]` versus `-12.5`: score 1; the existing parser does support this simple negative decimal case.

Cell 26 has a separate evaluation protocol: lowercased equality **or substring containment**, then a judge. It accepts answer `5` inside prediction `50`; a fixture confirmed score 1. The judge score parser tests whether the character `1` occurs anywhere, so `[0.1]` is treated as correct; also reproduced.

Corrections needed: canonical GSM8K numerical comparison using the provided `####` answer; normalized textual HotpotQA exact match and official-style answer F1; one documented answer-extraction protocol across K and evaluation stages. Fixing comparison changes which outputs get exact-match training reward and bypass the judge, so document the impact explicitly. Keep benchmark metrics separate from fallback-inclusive training reward. Do not use substring accuracy or judge scores for strict exact-match checkpoint selection.

### Reward semantics and failure handling

Cell 20 `compute_reward()` gives exact-match reward 1, failed execution (`None`) reward 0, and otherwise asks the same global execution LM to judge correctness. The judge sees prompt, prediction, and target; its parsed float is bounded to [0,1]. A fixture confirmed reward **0.5** is possible. Although the docstring mentions a format bonus, the executable code has no such bonus. Do not add one or binarize the released fractional fallback without recording a methodology change.

The paper's section 3.3 / Figure 2 describes binary reward with LLM semantic fallback; section 4.1 and Appendix A.1 also discuss LLM fallback. Appendix A.7 instead describes embedding-based semantic evaluation at a 0.92 cosine threshold. These are distinct and not reconciled by the released code. Preserve the notebook's LLM judge for the primary baseline; do not add an embedding model merely to match that appendix passage.

Parser tests also found that `[nan]` can produce a nonfinite reward, and a judge transport ConnectionError escapes the parser's exception handler. There is no surrounding episode recovery/checkpoint mechanism in cell 22. Pipeline exceptions are reduced to `None` and printed rather than structurally recorded. Later notebooks need finite-reward/log-probability/gradient checks, explicit failure categories, and service preflight checks, without silently shrinking K or inventing rewards/results. Infrastructure failures must remain distinguishable from incorrect model answers.

### Equal-reward groups and Adam momentum

The original always calls Adam.step after finite zero-advantage loss. A zero gradient does **not** imply zero parameter movement once Adam has momentum from earlier episodes. A one-parameter synthetic demonstration produced a nonzero change after a zero-gradient step.

To preserve released mathematical/optimizer behavior, the primary implementation should retain finite zero-advantage Adam steps and log zero-gradient groups separately from optimizer-step counts. Skipping them would be a methodological change to the optimizer trajectory, not a harmless equivalence. Invalid or nonfinite groups must not update parameters. This decision is documented in DECISIONS.md.

### Initialization, automatic execution, and state management

- Cell 1 deletes the GPT-2 checkpoint and evaluation JSON; it must not be copied into ordinary setup.
- Cells 6 and 9 both start `ollama serve`; the Linux installation cell is unsuitable as a required Windows setup path. Use connection checks and platform-aware setup guidance.
- Cell 17 loads an additional TinyLlama policy before cell 18 replaces it with GPT-2. Avoid copying this unnecessary alternative setup into the requested GPT-2 notebooks.
- Cell 18 automatically reloads a shared checkpoint if it exists. This violates independent fresh K initialization unless explicitly replaced with new/resume/evaluate controls.
- Cell 24 can save the active GPT-2 policy under a TinyLlama checkpoint filename when run in notebook order. Retain a single clear policy identity and independent artifacts.
- Cells 24 and 25 invoke training unconditionally. Do not place equivalent calls in initialization.
- The original saves only policy weights at training completion. No optimizer/RNG/episode/schedule/configuration snapshots, validation, periodic checkpoints, run isolation, or interruption-safe resume exist.
- `random.shuffle`, NumPy question sampling, and PyTorch action sampling have no fixed seed here. A seed plus persisted dataset IDs and episode schedule is required for the new comparison.
- All K candidates' policy forward graphs remain live until one backward pass, including an unnecessary forward for deterministic stop. This can grow activation memory with K. Recompute sampled-action log probabilities one trajectory at a time with unchanged parameters and consistent dropout/modes; accumulate gradients for the same summed objective before one optimizer step. Preserve state text and valid-action ordering.

### Do not reuse the existing baseline runner as methodology

`run_grpo_baseline.py` uses cyclic `episode % len(prompts)` selection, divides advantages by reward standard deviation, and clips gradients to norm 1 (around lines 229–277). These differ from the original notebook and the master specification. Its separate implementation architecture also conflicts with the requested notebook-only workflow. Leave the file and existing report untouched; do not copy their experimental results or objective into the primary baselines.

## 5. Dataset and HotpotQA task audit

GSM8K cell 13 uses `openai/gsm8k`, configuration `main`, and strips ground truth after `####`. Cell 28 selects 10 official test examples. The new notebook must use reproducibly sampled disjoint training/validation subsets and the fixed official-test subset, rather than these toy slices. The [official GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k) confirms the dataset identity and answer format.

HotpotQA cell 13 loads `hotpotqa/hotpot_qa`, **fullwiki**, training rows 0–9, then drops context from training prompts. Cell 28 uses training rows 20–29 as toy held-out examples and preserves raw `context`. Cell 26, with `query=True`, concatenates the Python representation of that context with the question. Despite its retrieval-related docstring, it performs **no Wikipedia search or retrieval**. The policy and DSPy program receive only the constructed prompt string; the sampler/executor have no retrieval tools.

The paper does not identify distractor versus fullwiki or precisely specify context construction in the inspected task/setup passages. A claim that the paper used distractor is therefore unsupported. The [official HotpotQA site](https://hotpotqa.github.io/) distinguishes provided-paragraph distractor evaluation from full-Wikipedia answer finding. Using a fullwiki-named split alone does not make the toy notebook a fullwiki retrieval system.

For Phase 3, prefer the user-requested distractor, provided-context protocol as a **documented project choice**, not verified paper reproduction. Use consistent input formatting across training, validation, and testing, with all provided passages rather than gold supporting-fact selection. Keep answers/supporting-fact annotations out of model inputs, retain their metadata separately, and add no external retrieval. Context availability and policy-input truncation must be documented; GPT-2's 1,024-token limit means blindly applying the original right-truncation to long contexts can remove the pipeline-state suffix.

The [maintainer's dataset card](https://huggingface.co/datasets/hotpotqa/hotpot_qa) provides distractor train/validation and fullwiki train/validation/test configurations. Use publicly labeled development/validation examples for final local scoring. The authorized disjoint training-split alternative remains available if clearly named and also disjoint from policy validation.

For answer-only HotpotQA evaluation, follow the [official evaluator](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py): normalized exact match and token-overlap F1, including the yes/no/noanswer special cases. Supporting-fact and joint metrics require supporting-fact predictions and must not be reported here.

No official dataset was downloaded or locally split in this phase. Dataset loading, revision pinning, sampled-ID separation, and schema integration remain Phase 2/3 checks.

## 6. Environment and hardware observations

These are read-only observations, **not training or inference measurements**:

- NVIDIA driver probe: RTX 3060 Laptop GPU, 6,144 MiB total VRAM, driver 617.14. Snapshot usage was approximately 481–483 MiB; this is not experiment peak memory.
- Default shell Python: `C:\Python314\python.exe`; PyTorch `2.13.0+cpu`, CUDA unavailable. Installed Transformers 5.14.1, DSPy 3.3.0, Datasets 5.0.1, NumPy 2.4.1, Matplotlib 3.10.9. Imports succeeded.
- Existing `venv_gpu\Scripts\python.exe`: Python 3.13.7, PyTorch 2.6.0+cu124, CUDA 12.4, `torch.cuda.is_available() == True`. Metadata reports Transformers 5.14.1, DSPy 3.3.0, Datasets 5.0.1, NumPy 2.4.4, Matplotlib 3.11.1. Cached tokenizer and synthetic Torch checks ran with this interpreter. The virtual environment was not installed or modified by this phase.
- `nbformat` is absent in the inspected interpreters. Notebook JSON and ordinary Python syntax were checked directly; formal nbformat validation is pending notebook implementation.
- Ollama server `http://localhost:11434` answered `/api/version`, `/api/tags`, and `/api/ps`. Version: 0.33.2.
- Requested tag `llama3.1:8b` is installed: 8.0B, GGUF, Q4_K_M; model digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`; artifact size 4,920,753,328 bytes. This disk/artifact size is **not measured runtime VRAM**.
- `/api/ps` returned no loaded models, so current execution-model VRAM and GPU offload proportions were unavailable. No model was loaded to obtain them.

Installed Datasets source no longer supports `trust_remote_code`; avoid copying that obsolete flag. GPT-2 imports and DSPy imports work, but full dataset loading and live DSPy/Ollama execution are not thereby verified.

Installed DSPy LM source enables caching by default, and the original does not explicitly control decoding or cache settings. Repeated candidates may reuse responses. Record the cache/decoding protocol and distinguish pipeline attempts, logical LM invocations, physical requests/retries, and judge requests. Do not infer actual requests or tokens by multiplying K. Choose a documented cache policy in implementation rather than silently changing execution variability.

The [Ollama running-model endpoint](https://docs.ollama.com/api/ps) can expose loaded size/VRAM metadata, and [generation responses](https://docs.ollama.com/api/generate) can expose token counts and durations. Whether the chosen DSPy adapter preserves these fields reliably needs a live integration check. Record unavailable values explicitly.

The 6GB limit warrants sequential candidates, one GPT-2 policy, bounded graphs, configurable execution offloading/endpoint, and periodic saves. No guarantee of simultaneous GPT-2 training plus Llama residency, achievable latency, or peak memory has been established.

## 7. Actual checks and outcomes

Checks used isolated function definitions extracted via Python AST, not Run All. No setup, deletion, install, model-download, or original training invocation cells were executed. Rendered pages and an empty JSON produced by an isolated original evaluation-call fixture are labeled in [audit_artifacts/README.md](audit_artifacts/README.md); they contain no experimental results. A tool policy check rejected deletion without a detailed reason, so those audit files were reversibly relocated into that documented evidence folder instead.

Passed checks:

1. JSON notebook v4 parsed; 31 cells. All 16 ordinary Python code cells parsed with AST; 3 shell/magic cells were inspected but not executed. Saved notebook outputs contained no stored error objects; this is not proof of successful fresh-kernel execution.
2. Cached GPT-2 tokenizer loaded offline; all action strings nonempty; collision groups enumerated above.
3. Original sampler's selected action probabilities matched independently computed restricted log-softmax probabilities; deterministic stop log probability was zero.
4. Original generator propagated nonzero gradients to the small policy's parameters.
5. Synthetic means, centered advantages, unnormalized summed loss, and log-probability gradients matched Figure 2 for K=5,8,10,15.
6. Original training loop, with a two-parameter synthetic policy and stubbed execution/reward, attempted exactly K candidates on one prompt with the matching ground truth, and updated the intended policy for all four K values. Each test ran one synthetic episode; temporary synthetic weight files were removed with their temporary directories.
7. Reward parsing preserved a synthetic fractional fallback score of 0.5; simple negative-decimal comparison passed.
8. System dependency imports, GPU-environment CUDA availability, GPU driver metadata, and read-only Ollama service/model-tag probes succeeded.
9. Original notebook and paper hashes remained unchanged; final documentation/source checks are also recorded in PROGRESS.md.

Defects reproduced (the tests detected expected incorrect original behavior): wrong-policy evaluation ownership; undefined value-head calls; duplicate `query` arguments; comma/decimal/text comparison errors; substring false positives; loose judge-score parsing; nonfinite reward acceptance; and unhandled judge transport errors. The zero-gradient Adam momentum demonstration reproduced nonzero parameter movement after a prior nonzero step, confirming why skipping equal-reward steps is not behaviorally identical.

Not run: real GPT-2 weight initialization/forward/backward, Llama generation, live DSPy programs, official dataset loading or subset-separation checks, validation/test scoring, new checkpoint/resume tests, plotting, or expensive static-K experiments. Those capabilities do not exist in new notebooks yet.

## 8. Phase 2 acceptance checklist derived from the audit

Implement only after instruction to begin Phase 2, entirely within Static_K_GRPO_GSM8K.ipynb:

- Clear 21-section notebook, fresh-kernel setup, explicit controls, and no automatic training/cleanup/checkpoint loading.
- One pretrained GPT-2 policy, fresh Adam per K, explicit model ownership; record revision, versions, mode, device, and execution-model metadata.
- Preserve all 15 signatures, CoT/Predict, constrained first-token categorical behavior, deterministic stop, fractional LLM judge fallback, and Figure 2 summed loss.
- Fix numerical answer normalization and finite-value/error handling; document their reward impact and do not add shaping or textbook GRPO terms.
- Persist disjoint selected dataset IDs and a shared replacement schedule; shorter preset prefixes and independent dataset/K/seed/preset/run namespaces.
- Exactly K attempted candidates per committed episode, one selected example, at most one finite update, centered rewards, full logging, zero-advantage/optimizer distinctions, and bounded activation memory.
- Explicit versioned checkpoints with optimizer, RNG, episode/config/schedule/metrics and robust log-resume consistency. No silent cross-K warm starts or partial-episode duplicate updates.
- Fixed validation with one pipeline per question, validation-only checkpoint choice with a documented tie break, explicit final testing, strict numeric accuracy separate from semantic fallback.
- Actual resource measurements and eleven plots from real logged data; unavailable metrics stay unavailable.
- Notebook-local synthetic checks first, then feasible reduced live checks, with passed/failed/skipped status and original-source preservation.

Phase 1 is complete as an audit. Scientific ambiguities about exact published episode budgets, semantic fallback variant, and HotpotQA input formulation remain documented rather than falsely resolved. These do not prevent implementing the explicitly specified project protocol. Phase 2 has not started.
