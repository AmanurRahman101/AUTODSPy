# Matching Static-K baseline for the unique-signature scorer

The Adaptive-K notebooks now use `unique-signature-token-label-v1`. Historical Static-K and Adaptive-K runs used the first token of each signature; retain their results as a separate cohort. The new policy assigns each signature a different existing GPT-2 vocabulary token as its scoring label. The signature sent to DSPy is unchanged. Modules and the forced stop retain their prior scoring.

This is an action-label method: it does not compute the language-model likelihood of the full signature. The GPT-2 model supplies initially pretrained logits for the chosen label tokens, so the initial signature distribution can differ substantially from the old one. The Figure 2 centered, summed loss and the prompt schedule are unchanged.

## Changes to make later in each Static-K notebook

Apply the same changes to the GSM8K and HotpotQA Static-K notebooks you intend to compare, including any Colab copies. Preserve the existing notebooks and results; create separate versions for the new protocol.

1. In `build_config`, change `policy_sampling` from `restricted-first-token-multinomial` to `unique-signature-token-label-v1`. Give the static protocol a new version string identifying this scorer. Keep the dataset-specific protocol name explicit.
2. In `initialize_policy`, after loading the tokenizer and model, replace the existing `first_tokens` assignment with the block below. Keep the function return value and the rest of the initialization unchanged.

```python
if cfg.get("policy_sampling") != "unique-signature-token-label-v1":
    raise ValueError("This notebook requires the unique signature scorer")
if len(SIGNATURES) > 26 or len(set(SIGNATURES)) != len(SIGNATURES):
    raise ValueError("Signature list must be unique and contain at most 26 entries")

signature_tokens = {}
for i, signature in enumerate(SIGNATURES):
    label = " " + chr(ord("A") + i)
    ids = tokenizer.encode(label, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("Each signature label must encode as exactly one GPT-2 token")
    signature_tokens[signature] = ids[0]
if len(set(signature_tokens.values())) != len(signature_tokens):
    raise ValueError("Signature scoring labels must have distinct token IDs")

first_tokens = {
    action: tokenizer.encode(action, add_special_tokens=False)[0]
    for action in ACTIONS
}
first_tokens.update(signature_tokens)
cfg["signature_action_token_ids"] = dict(signature_tokens)
```

3. Preserve the exact `SIGNATURES` ordering and tokenizer model/revision used by the matching Adaptive-K run. Labels are positional; reordering changes their meaning. Save the resolved `signature_action_token_ids` in config/checkpoint metadata as above, and reject missing or different mappings on resume and comparison.
4. Keep `action_log_probs`, `sample_actions`, and `replay_log_probs` using the resulting `first_tokens` mapping. Their categorical normalization and replay must be identical to the Adaptive-K notebook. Do not change reward mode, judge/cache identity, dataset splits, schedule, policy initialization, optimizer or loss to make this scorer comparison.
5. Initialize every Static-K run independently. Do not resume an old first-token checkpoint under the new scorer. Run new matching Static-K baselines for the selected K values and seeds after a small verification pilot. Retain old results separately.

## Checks before a research run

- Confirm every signature has a different label token ID, including `problem -> solution` and `problem_description -> explanation`. Confirm the saved mapping is identical between arms.
- On a small synthetic policy, recreate episode 9's reward pattern and verify the two signature-label logits can receive different, nonzero gradients; verify sampled and replayed log probabilities match.
- Confirm the checkpoint loader and comparison reject historical protocol identities and altered mappings. Test an interrupted run and resume under the new mapping.
- Run a small live smoke check using the chosen executor, then compare matching development runs. Record candidate, judge, validation and test costs separately.
- Treat any previously viewed test questions as exploratory when they informed method design; reserve fresh held-out questions for a later confirmation.

The new scorer changes the policy action parameterization and its initial probabilities. Differences between old and new scorer cohorts cannot isolate Adaptive-K. Within the new cohort, a Static-K versus Adaptive-K comparison isolates allocation only if the scorer, mapping, reward protocol and all other controls match.
