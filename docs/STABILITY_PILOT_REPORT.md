# K=5 controlled stability pilot

Completed 2026-10-13 (Asia/Dhaka). Pilot: `results/gsm8k/k5/seed42/pilot/20261013T032749-85f3db7c`.

Changed `Static_K_GRPO_GSM8K.ipynb` to add rollout action distributions, same-question pre/post-update entropy probes, unique pipeline counts, zero-variance/identical-action flags, scientific-notation loss/gradient logging, and raw semantic-judge traces. Full module and both conditional signature distributions are saved; the per-episode signature entropy is module-probability-weighted conditional entropy, measured in nats. Deterministic stop is excluded from pipeline entropy.

Added notebook-local bounded pilot, comparison, and synthetic instrumentation checks. Saved execution switches remain disabled. Default research learning rate/model are unchanged; the explicit pilot copies the reference configuration.

## Controls

- Fresh pretrained GPT-2 revision/digest matches the original; Adam state empty before training.
- K=5, seed 42, exact persisted selection and 10-question schedule prefix, train pool 50, validation 10, schedule length 200.
- Same original execution model: `llama3.2:3b` (this run's explicit deviation from the default Llama 3.1 8B), same model digest, Ollama 0.33.2, package versions, CUDA policy, temperature 0, seed 42, output budget 512/context 4096, automatic Ollama GPU allocation, reward prompt/parsing and cache settings.
- Only configuration differences: learning rate `1e-4 -> 1e-5`, episode budget `30 -> 10`, label `baseline -> pilot`.
- Original Figure 2 sum and all finite Adam steps preserved. No clipping, entropy bonus, reward normalization or step skipping.
- Exactly 50 training and 20 validation pipeline evaluations. No held-out test, other K, full baseline or second pilot.
- Episode 1 candidates, rewards, loss and gradient matched the original exactly.

## Measured stability

Diversity survived in all nine episodes after episode 1: unique counts `[5,4,4,4,4,3,4,5,3,4]`. The reference had counts `[5,1,1,1,1,1,1,1,1,1]` over the same prefix.

Six pilot episodes (1,3,4,5,8,9) had gradients greater than the explicit `1e-6` cancellation-residual cutoff, compared with only episode 1 in the reference prefix. Zero gradients in episodes 2,6,7,10 correspond to all five rewards being one; these are expected zero advantages despite pipeline diversity.

```text
Episode  Old unique  Pilot unique  Old gradient  Pilot gradient  Module H  Signature H
      1           5             5    2.1737e+02      2.1737e+02    0.6047       2.2043
      2           1             4    0.0000e+00      0.0000e+00    0.5662       2.3358
      3           1             4    0.0000e+00      2.1234e+02    0.6703       2.0751
      4           1             4    0.0000e+00      1.7000e+02    0.5983       2.0260
      5           1             4    0.0000e+00      1.9892e+02    0.4866       2.0902
      6           1             3    0.0000e+00      0.0000e+00    0.6433       2.0630
      7           1             4    0.0000e+00      0.0000e+00    0.6106       2.0982
      8           1             5    0.0000e+00      1.9934e+02    0.6721       2.0254
      9           1             3    0.0000e+00      1.6172e+02    0.6838       1.5776
     10           1             4    0.0000e+00      0.0000e+00    0.6480       1.6508
```

On the first training question, P(Predict) changed `0.292865 -> 0.509039` at the first update, while module entropy changed `0.604697 -> 0.692984` nats. Episode 10 retained module entropy `0.647977` and expected signature entropy `1.650842` before the update; after the zero-gradient Adam step these were `0.679615` and `1.540530`. This illustrates continuing momentum without new reward gradients, while diversity remains present.

This supports improved stability over these ten episodes. One seed and a short pilot do not establish stability over 200 episodes or improved benchmark accuracy. No `1e-6` pilot is warranted by a persistent collapse in this observed window; if a later separately authorized run collapses, recommend a distinct fresh `1e-6` pilot rather than adding methodological changes.

## Rewards and outcomes

Saved 17 semantic-judge training traces with exact prompts, raw returned strings, provider responses and effective DSPy/LiteLLM inference settings; validation traces are saved too. Authentication keys/headers are excluded. Physical HTTP parameters beyond these request settings are not observed. No repeated identical judge prompt in this pilot received conflicting scored rewards (0 observed conflict groups); this does not establish judge determinism or resolve the old inconsistency.

Validation numeric EM: [(0, 0.5), (10, 0.5)]. Ten validation questions are insufficient to conclude a general accuracy gain. Training completed in 155.07 seconds; 0 training execution failures, 0 judge failures, 0 OOM failures.

## Verification and preservation

Existing 111 notebook-local synthetic checks passed; four additional instrumentation checks confirmed entropy/probability correctness, unchanged policy RNG draws and weights, raw judge trace capture without changed requests/secrets, and retention of parse/transport failure traces. Live fields were verified across all ten rows and all 17 training judge traces.

SHA256 checks confirm all 18 files in the original run and existing diagnosis remain unchanged, including checkpoints. The run function also verified preservation of the original saved selection and `DSPy_GRPO.ipynb`. The Figure 2 loss/update cell is byte-for-byte identical to the previously committed notebook. No commit or push was made.

Artifacts: `results/gsm8k/k5/seed42/pilot/20261013T032749-85f3db7c/stability-comparison.json`, `results/gsm8k/k5/seed42/pilot/20261013T032749-85f3db7c/stability-comparison.csv`, `results/gsm8k/k5/seed42/pilot/20261013T032749-85f3db7c/stability-investigation.json`, `training.jsonl`, `attempts.jsonl`, and independent checkpoints in the pilot directory.
