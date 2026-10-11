# Phase 3 — Independent HotpotQA notebook

Status: **completed within the documented lightweight verification scope**, 2026-10-12.
Phase 4 is not started. No full research workload is launched.

## Deliverable and organization

`Static_K_GRPO_HotpotQA.ipynb` contains all 21 numbered sections from the master
specification, with local initialization, execution, training, checkpoint/resume,
evaluation, logging, comparison, plotting and self-check functions. It does not
import or execute GSM8K. The original notebook and verified GSM8K notebook remain
unchanged. Usage instructions are in `STATIC_K_EXPERIMENT_GUIDE.md`.

## Scientific choices

Use official [HotpotQA distractor](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
with all provided context consistently in train/validation/test. The
[official benchmark](https://hotpotqa.github.io/) defines the provided-paragraph
distractor task. The released toy notebook omits context during training but later
uses it in evaluation; the exact publication setup remains unverified. This is an
explicit project choice and input-consistency correction, without external retrieval.
Gold answers/supporting-fact annotations stay outside inputs, while all legitimate
source paragraphs are retained. Supporting-fact scores are not reported.

Defaults: GPT-2, Llama 3.1 8B, Adam 1e-4, seed 42, K=5/8/10/15, train pool 200,
validation 50 additional training examples, final test 200 labeled development
examples. The optional training-split holdout occupies a third disjoint slice after
training and validation and changes protocol identity. Full development evaluation
is opt-in; unlabeled official test examples are never scored.

Preserve CoT/Predict, all fifteen signatures, first-token categorical collisions,
the summed Figure 2 loss, centered advantages, no-grad rollouts and sequential
differentiable replay. One scheduled question per episode, replacement draws,
one Adam step, independent pretrained starts and the shared persisted schedule
remain unchanged. Finite equal-reward groups retain Adam's momentum behavior.

Use [official answer normalization/F1](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py),
including multiset token overlap and the yes/no/noanswer mismatch rule. Main reward
remains normalized exact match one, otherwise the original fractional semantic
judge. F1 never shapes training reward or selects checkpoints. Highest validation
answer EM selects the checkpoint; earliest episode wins ties. Final tests are explicit.

## Resource and reproducibility limits

GPT-2 left truncation at 1024 tokens can discard early context. Question and partial
actions remain at the end; trajectories log actual token counts, budgets and
truncation flags. `MAX_POLICY_TOKENS` is editable, but shorter budgets are explicit
input deviations and must match across K. Llama receives the complete formatted
input within its configured server context; server truncation is unavailable.

The real-GPT-2 long-context synthetic update passed with nonzero gradient norm
86.149981 and changed parameters. PyTorch peak allocated bytes: 4,626,561,024;
reserved bytes: 5,750,390,784 (about 5.36 GiB reserved). These are allocator
measurements from that engineering check, not total device or benchmark memory.
Concurrent applications and Llama residency reduce available capacity; CPU offload
requires host RAM, or use a remote Ollama endpoint. No smaller model is substituted.

Periodic recovery restores the last durable policy/optimizer/RNG/metric state;
unsaved work rolls back with its rows. The attempt journal retains external costs.
Interrupted validation is retried without duplicate effective updates. Call counts
are logical LM.forward attempts, not exact HTTP requests. Missing tokens/time
measurements remain unavailable. Fixed seeds do not guarantee remote determinism.

## Verification

Passed checks:

- Notebook schema and all 21 code-cell ASTs; all 21 numbered headings are present.
- Fresh `autodspy-gpu` kernel executed the final definition cells and **96 synthetic
  assertions (87 distinct labels)**. These cover EM/F1 edge cases, input independence
  from gold metadata, all-context policy/execution consistency, split isolation,
  every K, schedule prefixes, the unnormalized sum loss, gradients, equal-reward
  Adam behavior, selected-policy evaluation, checkpoint compatibility/overwrite
  rejection, nonzero-update interrupted recovery, interrupted intermediate/final
  validation, failed candidates, structured logs and twelve empty-data plot mechanics.
- Official immutable dataset revision `1908d6afbbead072334abe2965f91bd2709910ab`:
  90,447 train and 7,405 labeled development examples; disjoint 200/50/200 selection,
  persisted IDs/context/metadata, identical K schedules and preset prefixes. The
  **planned** 200-episode schedule contains 130 unique training IDs. This is not a
  claim that 200 episodes were executed.
- Real pretrained GPT-2 CUDA long-context update, as described above. Initial
  parameters match the verified GSM8K initialization digest.
- A separate reduced live check completed: K=5, pool 2, one episode, validation/test
  one each, **CPU policy**, policy budget 256, Llama 3.1 8B Q4_K_M with 8 GPU layers,
  context 4096 and output limit 512. Exactly five training candidates succeeded,
  with no failed training pipelines or logged OOM. All five rewards were one;
  advantages/loss/current gradient were zero and one finite Adam step occurred.
  No learning is demonstrated. All sampled policy states were truncated from
  approximately 1,323–1,329 tokens to 256.
- Live checkpoint restoration, standalone validation, validation-selected final
  testing, completed-run comparison and twelve saved/displayed real-data plots.
  Validation EM was zero and F1 0.4, while semantic reward was one: the metrics
  remain separate. Both validation checkpoints tied on EM, so episode 0 won.
  The single held-out engineering example had EM/F1 one. These tiny values are
  integration evidence only and have no benchmark interpretation.
- Five logical training LM calls plus seven evaluation/judge calls = **12 calls**
  across nine pipeline attempts. Returned usage totaled 19,279 prompt and 673
  generated tokens. Training time was 172.241414 seconds; average execution latency
  26.502571 seconds. The external verification driver took 308.108778 seconds;
  complete experiment wall time is still unavailable. CPU policy GPU allocator
  measurements are null; Ollama residency is separately recorded.
- Original notebook, paper and GSM8K hashes remained unchanged. Module/signature,
  policy initialization, action sampling/replay, execution and GRPO implementation
  cells are byte-identical to the verified GSM8K notebook where shared.
- `graphify update .` completed without LLM/API use: 742 nodes, 738 edges, 63
  communities. It skipped unsupported notebooks/PDFs; no graph-based notebook
  verification is claimed. Existing labels may be stale; no paid relabeling ran.

Evidence is in `audit_artifacts/phase3_synthetic_checks.json`,
`phase3_official_data.json`, `phase3_real_gpt2_update.json` and `phase3_live_check.json`.
`phase3_final_verification.json` records source/evidence agreement, actual plot files,
summary reconciliation, documentation checks and preserved hashes.
The final notebook checksum is
`03190623195962152c95f88d49cf9813058673a7fe0ae1fcc8f0c3fa543a7ad7`.
Live artifacts are under
`results/engineering_checks/phase3-live/hotpotqa/k5/seed42/smoke/20261011T183846-7b6a560c/`.
Large checkpoints remain local and ignored by the existing `*.pt` rule.

The engineering driver's cleanup initially passed a config dictionary where the
API helper expected a host string. Its caught cleanup error did not affect training
or evaluation. An explicit corrected unload request succeeded afterwards, and
`/api/ps` verified no loaded model. This is recorded in the live artifact.

## Unrun workloads and remaining issues

No full default smoke/pilot/baseline, live K=8/10/15, full labeled-development test,
CPU-only Llama generation, default 1024-token live joint-model training, or live
induced interruption was run. Synthetic recovery and real CUDA updates do not
prove those external-service/hardware paths. Exact physical HTTP counts and actual
server truncation remain unavailable. Long context and low free host RAM remain
hardware constraints; the shorter CPU-policy live check is an explicit deviation.

GSM8K stayed byte-for-byte unchanged rather than being rewritten or retested broadly.
MASTER_SPEC.md stayed unchanged. Phase 4 consolidation and Adaptive-K are not
implemented. This phase is complete as an independently verified implementation,
not as a completed static-K research experiment. Stop here until Phase 4 is authorized.
