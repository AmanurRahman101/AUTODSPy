# Original AutoDSPy paper and checked-in reference notebook

Research date: 19 October 2026. Scope: read-only analysis of the original paper and `DSPy_GRPO.ipynb`; no training or methodology changes. Paper claims below are reported results, not independently reproduced results. PDF page numbers refer to the local 16-page `AutoDSPy.pdf`.

## Source identity and citation correction

The authoritative paper is Nafew Azim et al. (2025), *AutoDSPy: Automating Modular Prompt Design with Reinforcement Learning for Small and Large Language Models*, EMNLP Industry Track, pp. 2881–2896, DOI **10.18653/v1/2025.emnlp-industry.192**. The local PDF title, authors, venue, and page numbers agree with the [official ACL record](https://aclanthology.org/2025.emnlp-industry.192/). The paper identifies [nafew-azim/AUTODSPy](https://github.com/nafew-azim/AUTODSPy) as its code repository and cites Zenodo DOI 10.5281/zenodo.17276875 for code. These paper and code identifiers serve different purposes.

The checked-in [README citation](../README.md#citations) has the right AutoDSPy title and 2025 year but the wrong DOI: `10.18653/v1/2024.emnlp-main.597`. That identifier belongs to Soylu, Potts, and Khattab's [*Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together*](https://aclanthology.org/2024.emnlp-main.597/). This is a citation error, not evidence of a 2024 AutoDSPy publication. No citation file was changed in this analysis.

## What the original work contributes

The original research learns a policy that selects DSPy reasoning modules and input/output signatures from a restricted pool for each question. The policy receives feedback from the final answer produced by executing the selected pipeline. Its aim is to automate pipeline construction and improve answer accuracy and inference latency compared with manually configured DSPy baselines. The policy model and the execution model have distinct roles: the policy chooses the program; the execution model answers the question through that program. The execution model's weights are not updated by the pipeline policy loss. [Source: local PDF §§3.1–3.3, pp.3–5; Figure 2, p.4.]

The broad formulation permits multistage pipelines, ReAct and other modules, and optional teleprompters. The main experimental results exclude teleprompters; Appendix A.1 studies their overhead and explains the choice. The checked-in GRPO reference concretely implements one module plus one signature and a forced stop, using only `CoT` and `Predict` and 15 signatures. This yields 30 nominal module/signature combinations, not the full compositional space suggested by the general diagram. [Sources: PDF Table 1, p.5; Appendix A.1, pp.10–11; notebook cells 15, 20.]

The original framework compares REINFORCE, PPO, and GRPO. Our Adaptive-K research concerns its GRPO path. The paper's GRPO diagram and checked-in training loop define:

\[
\bar r = K^{-1}\sum_{i=1}^{K}r_i,\qquad A_i=r_i-\bar r,
\]

\[
\mathcal L=-\sum_{i=1}^{K}\sum_t A_i\log\pi_\theta(a_{i,t}\mid s_{i,t}).
\]

The loop samples one prompt, samples K pipelines for that prompt, executes and rewards them, computes group-centered advantages, and performs one policy optimizer step. The loss is a **sum** over candidates and actions. There is no division by group size or reward standard deviation, PPO ratio/clipping, entropy term, critic, GAE, or KL term in this GRPO implementation. Generic PPO hyperparameters listed alongside GRPO in Table 3 do not establish that they belong in this loss. [Sources: Figure 2, p.4, visually checked in `docs/audit_artifacts/page_4.png`; notebook cell 22, `train_grpo`.]

Consequences derived from this formula: equal rewards yield zero advantages and zero current policy-loss gradient; K=1 always yields zero advantages; changing K changes both the number of evaluations and the scale/statistics of the summed loss. A zero current gradient does not necessarily mean zero parameter movement if an optimizer has stored momentum. These are mathematical implications, not measured paper results.

## Published experiment setup and results

The main tasks are GSM8K and HotPotQA. The paper reports a 200-example training set, 1,300 GSM8K test examples, and 1,000 HotPotQA test examples. It studies seven policy families: GPT-2 127M, Llama 3.2 1B, Gemma 3 1B, DeepSeek-R1 1.5B, Qwen 2.5 1.5B, Mistral v0.1 7B, and Phi-4 14B. The two execution models are Llama 3.1 8B and Qwen 2.5 14B. Its main GRPO configuration is **200 episodes, learning rate 1e-4, K=5**. [Sources: §4.1 and Table 1, p.5; Appendix A.3–A.4 and Table 3, p.14, visually checked in `docs/audit_artifacts/page_14.png`.]

For GPT-2 with GRPO and Llama 3.1 8B execution, Table 1 reports GSM8K accuracy **82.4%**, mean inference time **8.1 s**, and HotPotQA accuracy **73.9%**, mean inference time **4.2 s**. DSPy CoT with that execution model reports 79.9%, 11.1 s and 72.3%, 8.5 s respectively. Thus the GSM8K gain in this particular comparison is **2.5 percentage points**. GPT-2 GRPO with Qwen 2.5 14B execution reports 83.0% GSM8K and 74.3% HotPotQA; the paper's 77.2% HotPotQA result uses a different policy, DeepSeek-R1 1.5B with Qwen execution. Do not attribute every headline result to one GPT-2 configuration. [Source: local PDF Table 1, p.5.]

Appendix A.2 expands the benchmarks to MGSM, MBPP, LegalBench, and PubMedQA, with 200 training and 200 evaluation examples per benchmark and independently trained policies. MBPP uses pass@1; the others use accuracy. This broader evaluation is the original authors' work, not work demonstrated by the team's GSM8K runs. [Source: local PDF pp.12–13, Table 2.]

Appendix A.7 reports five GSM8K GRPO runs with accuracies 80.8%, 81.5%, 82.1%, 82.7%, 83.4%, summarized as 82.1 ± 1.0%. This is a paper claim, not a replacement for multiple seeds in the current project. [Source: local PDF p.15, Table 4.]

## The original paper already studies fixed K and proposes Adaptive-K

This is the most direct connection to the team's objective. Appendix A.1, **“Effectiveness of GRPO Hyperparameter k”**, studies fixed group sizes **5, 10, 15** with GPT-2 and Llama 3.1 8B. It reports GSM8K declining from 82.4% at K=5 to 60% at K=15, while HotPotQA rises from 73.9% to 77.1%. Figure 4 plots the intermediate setting; exact intermediate values should not be invented from the text. The authors interpret the difference as task-dependent group-size requirements and explicitly identify adaptive group-size selection as future work. [Source: local PDF Figure 4, p.10; Appendix A.1, pp.11–12.]

Therefore, comparing several fixed K values alone is groundwork for the extension. The new objective is to implement and validate a mechanism that chooses how many candidate pipelines to evaluate for a training prompt or episode. The checked-in original reference and the paper's presented GRPO method use fixed K. Adaptive-K is a direct continuation of an explicitly stated open direction in this paper; its broader novelty still requires a separate literature comparison.

## Checked-in reference notebook: actual behavior and caveats

These observations concern the file present on the current branch. They must not automatically be attributed to the exact code used for every published result.

- Cells 13 and 28 use only ten training examples per dataset, ten GSM8K test examples, and HotPotQA `train[20:30]` as a test subset. They are demo settings, not the main paper's reported benchmark evaluation.
- The function default is `num_episodes=500, learning_rate=1e-4, K=4`; actual training calls in cells 24–25 use two episodes, learning rate 2e-5, and K=4. Notebook defaults and paper hyperparameters differ.
- Both a TinyLlama initialization and a GPT-2 initialization appear sequentially. Running all cells overwrites the policy/tokenizer globals with GPT-2; a checkpoint filename containing TinyLlama does not establish which model was trained.
- `generate_pipeline` uses the first tokenizer token of each action label as its score. Signatures beginning with the same token share that score at the selection step. This restricts what preferences the model can express, even though there are 15 nominal signature labels.
- Training randomly selects a prompt with replacement using NumPy. It does not persist a common episode schedule, fixed seed setup, validation checkpoints, optimizer/RNG resume state, or detailed candidate telemetry.
- `compute_reward` first uses regex extraction, then asks the same configured execution LM for a score anywhere in [0,1]. Consequently, fractional semantic rewards are allowed. `evaluate_module` instead uses equality/substring checks and treats any judge response containing the character `1` as correct. Training reward and evaluation correctness are distinct implementations.
- `test_model(trained_model, ...)` sets the supplied model to eval but `generate_pipeline` uses global `policy_model`; the training function does not explicitly restore train mode. Alternate model references and evaluation/training interleaving need care.
- Final evaluation calls reference `trained_value_head`, although no value head is defined for this GRPO notebook. The HotPotQA call also supplies the query argument both positionally and by keyword if that name existed. These calls require correction in an experimental derivative, while preserving this reference file.
- Its initial cleanup cell deletes checkpoint/output files. It saves policy weights at the end of training without the optimizer or RNG state. It is not a resumable experimental harness by itself.

[Source for the observations above: local `DSPy_GRPO.ipynb`, cells 1, 13, 17–18, 20, 22, 24–26, 28–30; zero-based cell indexing.]

## Paper ambiguities to preserve in any reproduction claim

1. Figure 2 and the checked-in loop describe one randomly chosen prompt per episode. Appendix A.2/A.7 also use wording implying 200 samples per episode. The training pool size and number of episode updates must be stated separately in our experiments.
2. The main text and Figure 2 describe binary correctness with an LLM semantic fallback, while the notebook allows fractional judge rewards. Appendix A.7 additionally describes an embedding similarity fallback with threshold 0.92 on representative subsets. These are materially different scoring protocols.
3. Appendix A.6 claims subsecond execution for models up to 8B, while main Table 1 reports several seconds per example. Do not transfer either latency claim to this laptop or treat the two descriptions as an identical measurement setup.
4. Appendix A.1's explanation that larger K introduces harmful gradient noise is the authors' interpretation. Their static ablation does not by itself isolate the cause; summed-loss scale, policy behavior, execution noise, and reward consistency can also matter.
5. The original study's inference-time numbers measure the trained policy plus execution at test time. Adaptive-K primarily targets **training** candidate allocation. A training evaluation saving does not automatically imply faster single-pipeline test inference.

[Sources: local PDF Figure 2, §3.3, §4.1–4.2, Appendices A.1–A.2 and A.6–A.7; checked-in notebook as above.]

## Implication for the current project

The research goal is **Adaptive-K GRPO for AutoDSPy**: allocate a variable number of training pipeline evaluations while preserving the specified centered-advantage loss and then determine whether that allocation improves the accuracy/computation tradeoff. Reward consistency, faithful policy gradients, reproducible splits/schedules, Static-K baselines, and richer logging make that comparison interpretable. They are supporting work; adaptive allocation and its experimental validation are the central contribution still to establish.

The existing fixed-K logs can screen stopping rules and estimate avoided evaluations on observed prefixes. They cannot reveal the trained policies or held-out accuracy that would have resulted from updating on those shorter groups. A claimed Adaptive-K improvement requires actual training under the rule, comparable fixed-K runs, repeated seeds where feasible, and explicit cost accounting for candidate execution and semantic judging. These are methodological conclusions from the objective and implementation, not existing results.
