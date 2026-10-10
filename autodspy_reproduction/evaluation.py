"""Matched four-way held-out evaluation, with separate exact and semantic metrics."""
from collections import Counter
from pathlib import Path
import time
import torch
from .reward import compute_reward
from .runtime import append_json, artifact_hash, cuda_memory, json_write, load_checkpoint, new_run, seed_everything


def evaluate(config, policy, examples, executor, checkpoint, provenance):
    if not examples or any(row.split != "test" for row in examples):
        raise ValueError("Evaluation requires official held-out test examples")
    run = new_run(config.output_root, "evaluation")
    payload = load_checkpoint(checkpoint, policy)
    initial = payload["provenance"]["initial_checkpoint"]
    training_config = payload["config"]
    initial_hash = artifact_hash(initial)
    expected_initial_hash = payload["provenance"].get("initial_checkpoint_sha256")
    if expected_initial_hash and initial_hash != expected_initial_hash:
        raise ValueError("Initial-policy checkpoint identity changed")
    # Scientific comparison settings cannot drift between train/untrained/trained.
    allow = {"output_root", "evaluation_limit", "device", "cpu_threads", "checkpoint_every"}
    for key, value in config.to_dict().items():
        if key not in allow and training_config[key] != value:
            raise ValueError(f"Evaluation config differs from checkpoint: {key}")
    if payload["provenance"].get("execution") != provenance.get("execution"):
        raise ValueError("Execution model identity/settings changed since training")
    json_write(run / "manifest.json", {**provenance, "config": config.to_dict(),
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": artifact_hash(checkpoint),
        "completed_episodes": payload["completed_episodes"], "initial_checkpoint": initial,
        "initial_checkpoint_sha256": initial_hash, "metric": "normalized_numeric_exact_match",
        "semantic_metric": "binary_LLM_fallback", "judge_excluded_from_inference_latency": True})
    del payload  # Avoid retaining a second policy/optimizer copy during evaluation.
    summary = {}
    if policy.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for method in ("static_predict", "static_cot", "untrained_policy", "trained_policy"):
        if method == "untrained_policy":
            load_checkpoint(initial, policy)
        elif method == "trained_policy":
            load_checkpoint(checkpoint, policy)
        seed_everything(config.seed, config.cpu_threads)
        exact, semantic, errors, judge_errors = 0, 0, 0, 0
        generation_total, execution_total, judge_total = 0.0, 0.0, 0.0
        pipelines = Counter()
        for example in examples:
            started = time.perf_counter()
            if method.startswith("static_"):
                pipeline = ("Predict" if method == "static_predict" else "CoT", "question -> answer", "stop")
                generation_seconds = 0.0
            else:
                pipeline = policy.generate(example.question).pipeline
                generation_seconds = time.perf_counter() - started
            execution = executor.execute(example.question, pipeline)
            judged = time.perf_counter()
            result = compute_reward(example.question, execution.response, example.answer,
                                    executor.judge if config.semantic_fallback else None, binary=True)
            judge_seconds = time.perf_counter() - judged
            exact += int(result.exact)
            semantic += int(result.value == 1.0)
            errors += int(execution.error is not None)
            judge_errors += int(result.error is not None)
            generation_total += generation_seconds
            execution_total += execution.seconds
            judge_total += judge_seconds
            pipelines[" | ".join(pipeline)] += 1
            append_json(run / "predictions.jsonl", {"method": method, "example": example.to_dict(),
                "pipeline": pipeline, "execution": execution.to_dict(), "score": result.to_dict(),
                "generation_seconds": generation_seconds, "judge_seconds": judge_seconds})
            print(f"{method} test_index={example.index} exact={result.exact} semantic={result.value}", flush=True)
        n = len(examples)
        summary[method] = {"count": n, "exact_match_accuracy": exact/n,
                           "LLM_fallback_accuracy": semantic/n,
                           "mean_inference_seconds": (generation_total+execution_total)/n,
                           "mean_policy_generation_seconds": generation_total/n,
                           "mean_execution_seconds": execution_total/n, "mean_judge_seconds": judge_total/n,
                           "execution_failures": errors, "judge_parse_failures": judge_errors,
                           "pipeline_frequencies": dict(pipelines), **cuda_memory(policy.device)}
        if hasattr(executor, "runtime"):
            summary[method]["execution_runtime"] = executor.runtime()
        json_write(run / "results.json", summary)
    json_write(run / "status.json", {"status": "evaluation_complete", "examples_per_method": len(examples)})
    return run
