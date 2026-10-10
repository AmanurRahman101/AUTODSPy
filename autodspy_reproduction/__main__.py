"""CLI never starts full training implicitly."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

# Required before the first CUDA operation for deterministic matmul on CUDA >=10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .config import load_config
from .data import check_no_leakage, load_examples
from .execution import DSPyExecutor, preflight
from .policy import load_policy
from .runtime import environment_manifest, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "preflight", "execute", "train", "evaluate"))
    parser.add_argument("--config", default="configs/grpo_gsm8k.yaml")
    parser.add_argument("--endpoint", help="Ollama endpoint, including a remote host")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", help="Required for evaluation")
    parser.add_argument("--resume", help="Episode-boundary checkpoint to resume")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    overrides = {}
    if args.endpoint:
        overrides["execution_endpoint"] = args.endpoint
    if args.device:
        overrides["device"] = args.device
    config = replace(config, **overrides).validate()
    if args.command == "validate":
        print(json.dumps(config.to_dict(), indent=2))
        return 0
    if args.command == "evaluate" and not args.checkpoint:
        parser.error("evaluate requires --checkpoint")
    if args.resume and args.command != "train":
        parser.error("--resume is only valid for train")
    seed_everything(config.seed, config.cpu_threads)
    # Fail before policy allocation if the execution stack is absent.
    execution_manifest = preflight(config)
    if args.command == "preflight":
        print(json.dumps({key: execution_manifest[key] for key in ("version", "model", "endpoint", "settings")}, indent=2))
        return 0
    policy = load_policy(config)
    executor = DSPyExecutor(config)
    examples, data_manifest = load_examples(config, "test" if args.command == "evaluate" else "train")
    if args.command == "execute":
        from .reward import compute_reward
        example = examples[0]
        trajectory = policy.generate(example.question)
        execution = executor.execute(example.question, trajectory.pipeline)
        reward = compute_reward(example.question, execution.response, example.answer,
                                executor.judge if config.semantic_fallback else None)
        print(json.dumps({"example": example.to_dict(), "trajectory": trajectory.to_dict(),
                          "execution": execution.to_dict(), "reward": reward.to_dict()}, indent=2))
        if execution.response is None or execution.error or reward.error:
            raise RuntimeError("Execution probe failed; inspect reported diagnostics")
        return 0
    provenance = {"environment": environment_manifest(), "data": data_manifest,
                  "execution": execution_manifest, "policy": {
                      "model": config.policy_model, "revision": config.policy_revision,
                      "parameters": sum(p.numel() for p in policy.model.parameters()),
                      "first_token_collisions": policy.token_collisions(), "dropout": 0.0,
                      "precision": "float32", "max_tokens": config.max_policy_tokens}}
    if args.command == "train":
        from .training import train
        run = train(config, policy, examples, executor, provenance, args.resume)
    else:
        from .evaluation import evaluate
        train_examples, _ = load_examples(config, "train")
        check_no_leakage(train_examples, examples)
        run = evaluate(config, policy, examples, executor, args.checkpoint, provenance)
    print(f"Artifacts: {Path(run).resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Reproduction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
