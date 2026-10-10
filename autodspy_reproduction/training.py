"""Bounded activation memory: episode rollouts then sequential action backward."""
from pathlib import Path
import time
import numpy as np
import torch
from .loss import action_loss, group_advantages
from .reward import compute_reward
from .runtime import append_json, artifact_hash, cuda_memory, json_write, load_checkpoint, new_run, save_checkpoint


def update_episode(policy, optimizer, groups, config):
    if not groups:
        raise ValueError("An episode must contain prompt groups")
    optimizer.zero_grad(set_to_none=True)
    total_loss, max_log_difference = 0.0, 0.0
    for trajectories, rewards in groups:
        advantages = group_advantages(rewards, config.advantage_mode)
        for trajectory, advantage in zip(trajectories, advantages):
            for decision in trajectory.decisions:
                if decision.valid_actions == ("stop",):
                    continue
                current, entropy = policy.score(decision)
                difference = abs(float(current.detach()) - decision.old_log_prob)
                max_log_difference = max(max_log_difference, difference)
                # No optimizer step until ALL episode actions have been replayed.
                if difference > 2e-4:
                    raise RuntimeError("Behavior/current probabilities differ before update; replay is inconsistent")
                loss = action_loss(current, decision.old_log_prob, advantage, entropy,
                                   config.objective, config.clip_epsilon, config.entropy_coefficient) / len(groups)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                total_loss += float(loss.detach())
                loss.backward()
    squared_norm = torch.zeros((), device=policy.device)
    for parameter in policy.model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Non-finite policy gradient; optimizer not stepped")
            squared_norm += parameter.grad.detach().float().square().sum()
    grad_norm = float(squared_norm.sqrt())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss, "gradient_norm": grad_norm,
            "max_behavior_current_logprob_difference": max_log_difference}


def train(config, policy, examples, executor, provenance, resume=None):
    config.validate()
    if not examples or any(example.split != "train" for example in examples):
        raise ValueError("Training requires nonempty official train examples only")
    run = new_run(config.output_root)
    json_write(run / "config.json", config.to_dict())
    provenance = dict(provenance, algorithm={"optimizer": "Adam", "betas": [0.9, 0.999],
        "epsilon": 1e-8, "weight_decay": 0.0, "optimizer_steps_per_episode": 1,
        "reduction": "sum_over_candidates_and_actions_mean_over_prompt_groups",
        "advantage_mode": config.advantage_mode, "objective": config.objective,
        "GAE": False, "KL_penalty": False})
    json_write(run / "manifest.json", provenance)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config.learning_rate)
    sampler = np.random.default_rng(config.seed)
    first_episode = 0
    if resume is not None:
        payload = load_checkpoint(resume, policy, optimizer, sampler, config, restore_rng=True)
        previous = payload["provenance"]
        for key in ("data", "policy", "execution"):
            if previous.get(key) != provenance.get(key):
                raise ValueError(f"Resume provenance changed: {key}")
        if previous.get("environment", {}).get("source_sha256") != provenance.get("environment", {}).get("source_sha256"):
            raise ValueError("Resume implementation source changed")
        first_episode = payload["completed_episodes"]
        provenance = dict(provenance, initial_checkpoint=previous["initial_checkpoint"],
                          initial_checkpoint_sha256=previous.get("initial_checkpoint_sha256"),
                          resumed_from=str(Path(resume).resolve()))
    else:
        initial = run / "initial_policy.pt"
        # Store identity in every checkpoint so evaluation uses the correct untrained model.
        provenance = dict(provenance, initial_checkpoint=str(initial.resolve()))
        save_checkpoint(initial, policy, None, config, 0, sampler, provenance)
        provenance["initial_checkpoint_sha256"] = artifact_hash(initial)
    json_write(run / "manifest.json", provenance)
    if first_episode >= config.episodes:
        raise ValueError("Checkpoint already completed configured episode budget")
    if policy.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        for episode in range(first_episode, config.episodes):
            started = time.perf_counter()
            positions = sampler.integers(0, len(examples), size=config.samples_per_episode)
            groups, rewards_flat, execution_times, generation_times = [], [], [], []
            zero_groups = 0
            for sample, position in enumerate(positions):
                example = examples[int(position)]
                trajectories, rewards = [], []
                for candidate in range(config.group_size):
                    generated = time.perf_counter()
                    trajectory = policy.generate(example.question)
                    generation_seconds = time.perf_counter() - generated
                    execution = executor.execute(example.question, trajectory.pipeline)
                    judged = time.perf_counter()
                    reward = compute_reward(example.question, execution.response, example.answer,
                                            executor.judge if config.semantic_fallback else None)
                    judge_seconds = time.perf_counter() - judged
                    append_json(run / "trajectories.jsonl", {
                        "episode": episode + 1, "sample": sample, "candidate": candidate,
                        "example": example.to_dict(), "trajectory": trajectory.to_dict(),
                        "execution": execution.to_dict(), "reward": reward.to_dict(),
                        "generation_seconds": generation_seconds, "judge_seconds": judge_seconds})
                    trajectories.append(trajectory)
                    rewards.append(reward.value)
                    rewards_flat.append(reward.value)
                    execution_times.append(execution.seconds)
                    generation_times.append(generation_seconds)
                zero_groups += int(len(set(rewards)) == 1)
                groups.append((trajectories, rewards))
                print(f"episode={episode+1} sample={sample+1}/{config.samples_per_episode} "
                      f"train_index={example.index} rewards={rewards}", flush=True)
            update = update_episode(policy, optimizer, groups, config)
            metrics = {"episode": episode + 1, "samples": len(groups),
                       "pipeline_evaluations": len(rewards_flat),
                       "train_indices": [examples[int(i)].index for i in positions],
                       "unique_train_indices": len(set(int(i) for i in positions)),
                       "mean_reward": float(np.mean(rewards_flat)), "zero_advantage_groups": zero_groups,
                       "mean_execution_seconds": float(np.mean(execution_times)),
                       "mean_generation_seconds": float(np.mean(generation_times)),
                       "episode_seconds": time.perf_counter() - started,
                       **update, **cuda_memory(policy.device)}
            if hasattr(executor, "runtime"):
                metrics["execution_runtime"] = executor.runtime()
            append_json(run / "metrics.jsonl", metrics)
            print(metrics, flush=True)
            if (episode+1) % config.checkpoint_every == 0 or episode+1 == config.episodes:
                save_checkpoint(run / f"episode_{episode+1:04d}.pt", policy, optimizer,
                                config, episode+1, sampler, provenance)
            del groups
        json_write(run / "status.json", {"status": "training_complete", "evaluation_completed": False,
                                          "completed_episodes": config.episodes})
    except Exception as exc:
        json_write(run / "status.json", {"status": "failed", "error": repr(exc),
                                          "evaluation_completed": False})
        raise
    finally:
        if policy.device.type == "cuda":
            torch.cuda.empty_cache()
    return run
