# Validation results - 2026-10-10

These are observed engineering and smoke results, not a reproduction of the paper's benchmark accuracy. The full 200-episode experiment and full held-out evaluation were **not run**.

## Environment and preservation

Python 3.13, torch 2.6.0+cu124, transformers 5.14.1, DSPy 3.3.0, datasets 5.0.1, PyYAML 6.0.3; RTX 3060 Laptop GPU with 6,144 MiB VRAM. Ollama 0.33.2 served actual `llama3.1:8b`, 8.0B, Q4_K_M, digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`. `/api/ps` confirmed size_vram=0, context_length=8192 and loaded model size=6,247,216,577 bytes (this is Ollama's reported model allocation, not measured whole-process peak RSS). Temperature=0, seed=42, max generated tokens=4096, no DSPy response cache.

Retained existing branch `experiment/grpo-gsm8k-experiment`; no branch was created. Git object hashes of all three original notebooks, baseline script, baseline results and README match their original HEAD contents. The initially untracked supplied PDF was retained. Only `.gitignore` was modified among previously tracked files; new reproduction code/config/docs/notebook and graph outputs were added. No existing results or checkpoints were deleted.

## Stage 1: static and notebook validation - passed

- Python compileall passed for implementation and tests.
- Full/smoke YAML validation passed, including model guardrails and budget values.
- Every new notebook code cell parsed and default code cells executed in order. Unit tests ran; live/smoke/full/evaluation flags remained false. The original notebook was never executed.
- Official dataset revisions loaded from cache: train=7,473, test=1,319, no overlapping questions. Fingerprints: train `c6f812ae33c9159d`; test `59ec1b7f9357c7a2`.
- `graphify update .` completed with AST-only extraction: 309 nodes, 534 edges, 21 communities on the final refresh. Graph was absent at initial audit. The graph tool's own environment lacks pypdf and skips PDF semantic extraction; notebooks are unsupported for AST classification. Code graph updates succeeded; this is not a claim of semantic paper/notebook graph coverage. Community labels are automatically inferred, not manually curated.

## Stage 2: unit tests - passed

Final command: `venv_gpu/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py' -v`.

**18 tests passed**, final observed duration 5.244 seconds. Coverage includes first-token duplicate action slots, generation and termination, exact replay probabilities, notebook loss equivalence, population standardization, equal-reward zero gradients, both clipping signs, entropy and behavior detachment, finite/nonzero optimizer updates, accumulated versus full-graph gradients, signed/comma/decimal answers, strict binary judge parsing, infrastructure failures, dataset split protection, DSPy signature fields and adapter failures, checkpoint overwrite protection/RNG round trip, deterministic resume, and four-way evaluator integration.

Additional real pretrained GPT-2 scoring comparison initially failed an overly strict 1e-6 absolute tolerance. Diagnostic CUDA FP32 differences between all-position vocabulary projection and final-position-only projection were max 9.918e-5 in vocabulary logits, 3.651e-6 in module probabilities and 1.937e-6 in signature probabilities. The verified equivalence test now uses 1e-5 absolute probability tolerance. This is mathematical equivalence within floating-point rounding, not bit equality.

## Stage 3: real execution/reward probe - passed

Generated `CoT | query -> generated_text | stop` on official train example 0. Actual CPU Llama returned `[72]`; normalized exact reward=1.0. Observed DSPy execution latency=48.565 seconds (cold/warm loading conditions included, not a benchmark mean).

After restoring released judge instruction field order, a separate live wrong-answer probe (`2+2`, response `[3]`, gold `4`) invoked real Llama fallback and returned `0.0`, with no parse error. Infrastructure/adapter failures are handled differently: infrastructure errors abort; adapter parse errors are recorded and receive no reward.

## Stage 4: real training smoke - passed

Run directory:

`runs/reproduction-smoke/20261010T091315Z-train-607ef5bf`

Config: actual pretrained GPT-2 (124,439,808 tied parameters), official GSM8K train, real CPU Llama 3.1 8B, one episode, two sampled prompts, K=5, Adam 1e-4, centered advantages and released sum-of-log-probabilities objective.

- Selected train indices: 666 and 5783; ten real pipeline evaluations completed.
- Group rewards: `[1,1,1,1,1]` and `[0,1,0,0,1]`; continuous mean training reward=0.7. This is training feedback, not held-out accuracy.
- Equal-reward group count=1; loss=0.9599205256; finite nonzero gradient norm=141.8600464.
- Maximum replay behavior/current log-probability difference=0.0 before the optimizer step.
- Episode duration=373.133 seconds; mean pipeline execution=35.450 seconds; mean policy generation=0.316 seconds. These two training prompts cannot establish representative benchmark latency.
- PyTorch policy peak allocated=2,560,431,616 bytes (~2.56GB decimal / 2.38GiB); peak reserved=2,810,183,680 bytes (~2.81GB / 2.62GiB). These exclude Ollama, display, other processes and CPU RAM.
- `initial_policy.pt` created (497,856,764 bytes); `episode_0001.pt` created (1,493,501,820 bytes). Status reports training_complete, one completed episode; evaluation is a separate artifact.
- Config, model/dataset/runtime provenance, trajectories, loss/reward/memory logs and checkpoint/RNG state saved. No execution or judge parse errors. Console log: `runs/reproduction-validation/final-live-smoke.log`.
- Final live checkpoint reopened successfully with `weights_only=True`; completed episode, nonempty optimizer state, source-file hashes and initial-policy checksum verified against the final source/artifact files.

An earlier passing live smoke is preserved at `runs/reproduction-smoke/20261010T085350Z-train-2b16e309` (370.174 seconds). After final source/initial-checkpoint checksum additions, bounded DSPy history, explicit adapter-error handling and restoration of the released judge field order, the complete ten-candidate live smoke was repeated on the final source as documented above. Both runs produced the same recorded group rewards, loss and gradient norm. Existing smoke artifacts were preserved.

## Additional real-policy engineering smoke - passed

Final run: `runs/engineering-policy-smoke/20261010T090903Z-train-7109dc93`.

Real pretrained GPT-2, official train data and **SCRIPTED execution fixture**, clearly identified in manifest. Verified full/last-position probability comparison, finite nonzero gradients, actual GPT-2 parameter changes, checkpoint round trip, source/initial-checkpoint provenance and GPU memory telemetry. This does not execute Llama, benchmark accuracy, or count as a second live training smoke. Scripted reward/latency values are intentionally excluded from scientific results.

## Four-way held-out smoke evaluation - passed

Run: `runs/reproduction-smoke/20261010T090140Z-evaluation-1c9fc409`.

Used the earlier saved live smoke checkpoint `20261010T085350Z-train-2b16e309/episode_0001.pt`, matching execution model/digest/settings, official held-out indices 1309 and 228, same seed and bracket instructions for every method. Exactly **two examples per method**; no statistical performance conclusions are justified. The evaluation was not repeated for the final-source smoke checkpoint; its command is available in the guide.

- Static Predict: exact=0/2, exact+binary LLM fallback=0/2; mean inference=12.560 seconds.
- Static CoT: exact=1/2, exact+binary LLM fallback=1/2; mean inference=29.828 seconds.
- Saved initial/untrained policy: exact=0/2, exact+binary LLM fallback=0/2; mean inference=36.778 seconds.
- GRPO-inspired smoke-trained policy: exact=1/2, exact+binary LLM fallback=1/2; mean inference=37.223 seconds.

No execution or judge parse failures reported. Pipeline frequencies and generation/execution/judge timing components are recorded per method in `results.json`; raw predictions are in `predictions.jsonl`. Judge latency is excluded from inference means. Methods ran sequentially with the same warm CPU model, not under paper GPU timing conditions. Console log: `runs/reproduction-validation/live-evaluation.log`. Full evaluation remains pending.

## Stage 5 and practical limits

Full configuration is ready to run explicitly; no full experiment was started. Its 200 x 200 x 5 budget is 200,000 pipeline evaluations plus possible judge/adapter calls. Extrapolating the two-prompt live smoke's episode rate gives roughly 86 days on this CPU execution setup. This is a crude planning estimate, not a measured full-run duration: prompt lengths, selected signatures, reward/judge frequency and learned policy behavior can change it greatly. A remote GPU Ollama server running the same Llama 3.1 8B is the practical alternative while GPT-2 stays on the laptop GPU.

The training pipeline, checkpointing and evaluator have been exercised. Scientifically defensible reporting requires disclosing the sampling/update/dropout/reward/default assumptions in the audit and deviations document, then completing the full training and matched held-out evaluation. The paper's results have **not** been reproduced, and missing historical experimental details still prevent an exact numerical replication claim.
