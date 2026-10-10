# Deviations, assumptions and unresolved evidence

The default is a released-code-faithful **GRPO-inspired** experiment with the paper's target models and budget. It must not be presented as textbook clipped GRPO or as an exact replication of the published scores. See [the audit](REPRODUCTION_AUDIT.md) for page and cell references.

## Verified choices

- Pretrained `gpt2`, not TinyLlama or a newly constructed 127M model. The paper calls it 127M; the actual tied GPT-2 parameter count is recorded at runtime.
- `llama3.1:8b` execution, Predict/CoT only, the released fifteen signatures, first-token categorical scoring with duplicate token IDs, module/signature/forced-stop sequence. A single module executes; the richer architecture diagram is not implemented in the released notebook.
- Adam at 1e-4, 200 episodes, K=5 from A.4. No teleprompter, ReAct, calculator, retrieval, value head, GAE, reference policy or KL penalty.
- The actual released GRPO loss is `-sum(candidate, action) log_probability * (reward - group_mean)`. Figure 2 visually shows this same centered-reward, unclipped formula. Gamma/lambda/clipping/entropy listed in the shared table are not active in this objective. They are not quietly inserted into GRPO.

## Methodological assumptions and repairs

1. **200 samples per episode:** draw 200 independent indices uniformly with replacement from all 7,473 training examples, extending the notebook's random index draw. There is no cap at 200 unique examples. Section 4.1's 200-example training set conflicts with A.7's per-episode wording; exact sampling is unresolved. Set `train_pool_size: 200` only for a separately labeled fixed-pool sensitivity run.
2. **One update per episode:** collect all prompt groups at unchanged weights, average their individual notebook group losses, then take one Adam step. This preserves released one-update behavior and group/action sum scale while adding the paper's stated per-episode workload. The paper does not establish whether its 200 samples meant 200 optimizer steps.
3. **Dropout disabled:** set policy Dropout probabilities to zero to make detached behavior scoring and differentiable replay identical. The original notebook leaves pretrained dropout active unless evaluation has run. This is an explicit methodological assumption, not merely a memory optimization. Regression tests require matching pre-update log probabilities. No training gradients are disabled during replay.
4. **Reward:** retain released continuous LLM partial-credit fallback despite binary wording in the paper. Preserve the released judge instruction wording/field order while normalizing indentation. Fix numeric comparisons with Decimal for commas, signed values and equivalent decimals. No format bonus is added. Invalid judge output scores zero and is logged; server/infrastructure failures abort the run.
5. **Evaluation:** normalized numerical exact match and binary LLM fallback are reported separately. Substring matching and accepting any judge output containing `1` are removed. This can change scores relative to the buggy notebook. A.7's embedding fallback (cosine threshold .92) is not reproduced: its embedding model is unidentified, and no such evaluator occurs in the released code. The implemented semantic metric is explicitly named `LLM_fallback_accuracy`.
6. **Test split:** default final evaluation uses all 1,319 official test examples. The paper says 1,300; `evaluation_limit: 1300` selects and records a seeded subset for a separate comparison. The authors' original indices are unknown. Training never loads test data and evaluation checks question overlap.
7. **Reproducibility:** seed 42, explicit cached GPT-2/dataset revisions, FP32 policy, Llama temperature 0, 4,096 max generated tokens, 8,192 context, execution seed 42 and disabled response cache are assumptions. Historical DSPy/Ollama versions, defaults, quantization and seeds are unpublished. The quantization/digest/server version are read rather than assumed. Provider/server defaults not overridden are recorded from model metadata.

## Hardware-only implementation choices

- Sequential pipeline execution. Detached trajectories on CPU; one action backward at a time with mathematically equivalent accumulation. GPT-2 projects only the final hidden position to vocabulary logits, the scored position in the notebook. Different CUDA matrix shapes produce small FP32 rounding differences: the tested module probabilities differed by 3.65e-6 (absolute); the engineering equivalence assertion uses 1e-5 absolute tolerance, not bit equality.
- Forced stop needs no transformer pass: its probability is one and its gradient/entropy are zero.
- CPU execution (`execution_num_gpu: 0`) or a remote Ollama server reserves local GPU memory for GPT-2. A different placement does not substitute the execution model. Quantization remains an unresolved paper/runtime difference.
- Optional activation checkpointing with dropout disabled. No lower precision or parameter-efficient fine-tuning is enabled by default.
- Save initial weights and Adam/RNG checkpoints every ten episodes plus the final episode, instead of retaining 200 roughly 1.5GB checkpoints. Smoke saves after its single episode. New unique directories preserve previous artifacts. Resume starts a new directory from a completed checkpoint and may redo unsaved episodes.

## Notebook presentation

`DSPy_GRPO.ipynb` embeds the validated reproduction logic as ordinary visible cells with inline configuration and tests. Its training iterator yields after each completed episode so the user can inspect metrics and advance manually or run all remaining episodes. The iterator retains the same sampling, loss, gradient accumulation, Adam update and checkpoint cadence as the modular CLI implementation; no methodology change is intended. Actual executed definition-cell hashes replace module-file hashes for notebook resume provenance. Pauses do not deliberately reseed or alter policy weights. The prior notebook is preserved as `DSPy_GRPO_Original.ipynb`. Full training still requires explicit activation and restart after an independent smoke run.

## Sensitivity modes, not reproduction defaults

Copy the full YAML to a new filename and explicitly select `objective: clipped`, `advantage_mode: standardized`, `entropy_coefficient: 0.01`, `clip_epsilon: 0.2` to investigate normalized clipped optimization. This uses detached behavior probabilities and a tested clipped surrogate. With one on-policy update, pre-update ratios equal one, so clipping does not activate on the first step; no unsupported multiple epochs are added. Gamma/lambda remain inactive. These runs cannot resolve which objective produced the paper's results.

## Readiness criterion

Passing offline tests or the scripted-executor GPU engineering smoke does not pass the live Llama reproduction smoke. A live execution probe, real ten-candidate smoke with checkpoints/telemetry, full training and matched held-out evaluation are distinct milestones. None proves the published 82.4% until the relevant actual evaluation and protocol comparison are complete. Exact replication remains limited by sampling/update/reward/default ambiguities even after a successful run.
