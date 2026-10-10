"""Real pretrained GPT-2/official train split, SCRIPTED execution engineering test.

This measures policy memory and tests optimizer/checkpoint integration. It does
NOT execute Llama, provide benchmark accuracy, or pass the live reproduction smoke.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from dataclasses import replace
from pathlib import Path
import json
import torch
from autodspy_reproduction.config import load_config
from autodspy_reproduction.data import load_examples
from autodspy_reproduction.execution import Execution
from autodspy_reproduction.policy import MODULES, action_distribution, load_policy
from autodspy_reproduction.runtime import environment_manifest, load_checkpoint, seed_everything
from autodspy_reproduction.training import train


class ScriptedExecutionFixture:
    def __init__(self, examples):
        self.answers = {row.question: row.answer for row in examples}
        self.counter = 0

    def execute(self, prompt, pipeline):
        self.counter += 1
        answer = self.answers[prompt] if self.counter % 2 else "-999999999"
        return Execution(f"[{answer}]", 0.0)

    def judge(self, prompt):
        return "[0.0]"


def main():
    config = replace(load_config("configs/grpo_smoke_test.yaml"), output_root="runs/engineering-policy-smoke")
    seed_everything(config.seed, config.cpu_threads)
    policy = load_policy(config)
    examples, manifest = load_examples(config, "train")
    state = f"Prompt: {examples[0].question} Pipeline: "
    inputs = policy.tokenizer(state, return_tensors="pt", padding=True, truncation=True,
                              max_length=config.max_policy_tokens).to(policy.device)
    with torch.no_grad():
        original_logits = policy.model(**inputs, use_cache=False).logits[0, -1]
        original_distribution = action_distribution(original_logits, policy.tokenizer, MODULES)
        optimized_distribution = policy.distribution(state, MODULES)
        # GEMM(all positions) versus GEMV(last position) differs slightly in FP32 CUDA.
        probability_difference = float((original_distribution.probs-optimized_distribution.probs).abs().max())
        print(f"Full/last-position probability max absolute difference: {probability_difference}")
        if not torch.allclose(original_distribution.probs, optimized_distribution.probs, atol=1e-5, rtol=0):
            raise AssertionError("Final-position projection changed notebook action probabilities")
    del original_logits, original_distribution, optimized_distribution, inputs
    # Small snapshot proves a real parameter change without retaining another model.
    before = policy.model.transformer.wte.weight[:100].detach().cpu().clone()
    provenance = {"experiment_kind": "engineering_test_scripted_execution_NOT_Llama",
                  "environment": environment_manifest(), "data": manifest,
                  "policy": {"model": "gpt2", "parameters": policy.model.num_parameters(),
                             "token_collisions": policy.token_collisions()},
                  "execution": {"kind": "SCRIPTED_FIXTURE", "benchmark_results": False}}
    run = train(config, policy, examples, ScriptedExecutionFixture(examples), provenance)
    metrics = json.loads((run/"metrics.jsonl").read_text().splitlines()[-1])
    if metrics["gradient_norm"] <= 0 or torch.equal(before, policy.model.transformer.wte.weight[:100].detach().cpu()):
        raise AssertionError("Expected finite nonzero real GPT-2 parameter update")
    checkpoint = run/"episode_0001.pt"
    expected = policy.model.transformer.wte.weight[:100].detach().cpu().clone()
    load_checkpoint(checkpoint, policy)
    if not torch.equal(expected, policy.model.transformer.wte.weight[:100].detach().cpu()):
        raise AssertionError("Checkpoint round trip changed parameters")
    print(f"Engineering policy smoke passed (SCRIPTED executor): {Path(run).resolve()}")


if __name__ == "__main__":
    main()
