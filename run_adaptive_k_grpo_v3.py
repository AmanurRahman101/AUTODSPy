import argparse
import json
import re
import time
import random
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import dspy
import datasets

# ============================================================
# Modern HuggingFace Hub Compatibility Adapter
# ============================================================
_real_load_dataset = datasets.load_dataset

def _compat_load_dataset(path, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)
    if path == "gsm8k":
        path = "openai/gsm8k"
    elif path == "hotpot_qa":
        path = "hotpotqa/hotpot_qa"
        if args and args[0] == "fullwiki":
            args = ("distractor",) + args[1:]
        elif kwargs.get("name") == "fullwiki":
            kwargs["name"] = "distractor"
    return _real_load_dataset(path, *args, **kwargs)

datasets.load_dataset = _compat_load_dataset


# ============================================================
# Action Space Definitions & Module Cost Asymmetry
# ============================================================
MODULES = ["CoT", "Predict"]
SIGNATURES = [
    "question -> answer", "text -> summary", "question -> reasoning", "question -> hypothesis",
    "problem -> solution", "problem_description -> explanation", "context -> summary", "context -> briefing",
    "word_problem -> solution", "math_problem -> answer", "prompt -> response", "query -> response",
    "text -> response", "prompt -> generated_text", "query -> generated_text"
]
ACTIONS = MODULES + SIGNATURES + ["stop"]

# Empirical execution cost weights (\omega_m)
# Reflects relative LLM inference / token generation footprint
MODULE_COSTS = {
    "Predict": 1.0,
    "CoT": 2.2,
    "ReAct": 5.5
}


def get_pipeline_cost(pipeline):
    """Computes compound architecture cost based on component modules."""
    if not pipeline:
        return 1.0
    module = pipeline[0]
    return MODULE_COSTS.get(module, 1.0)


# ============================================================
# Answer Extraction & Reward Functions
# ============================================================
def compare_answers(response, ground_truth):
    if response is None:
        return 0.0

    resp_bracket_matches = re.findall(r'\[(.*?)\]', str(response))
    if resp_bracket_matches:
        response_answer = resp_bracket_matches[-1].strip()
    else:
        resp_num_matches = re.findall(r'-?\d*\.?\d+', str(response))
        response_answer = resp_num_matches[-1] if resp_num_matches else None

    gt_bracket_matches = re.findall(r'\[(.*?)\]', str(ground_truth))
    if gt_bracket_matches:
        gt_answer = gt_bracket_matches[-1].strip()
    else:
        gt_num_matches = re.findall(r'-?\d*\.?\d+', str(ground_truth))
        gt_answer = gt_num_matches[-1] if gt_num_matches else None

    if response_answer and gt_answer:
        response_num_match = re.search(r'-?\d*\.?\d+', response_answer)
        gt_num_match = re.search(r'-?\d*\.?\d+', gt_answer)
        if response_num_match and gt_num_match:
            response_num = response_num_match.group(0)
            gt_num = gt_num_match.group(0)
            regex_score = 1.0 if response_num == gt_num else 0.0
        else:
            regex_score = 1.0 if response_answer.strip().lower() == gt_answer.strip().lower() else 0.0
    else:
        regex_score = 0.0

    return regex_score


def compute_reward(prompt, response, ground_truth, lm):
    if response is None:
        return 0.0

    score = compare_answers(response, ground_truth)
    if score > 0.0:
        return score

    eval_prompt = f"""
    Evaluate whether the following response correctly answers the prompt based on the ground truth and give your final score in the square brackets [final score]. only the value in the [] should in your response and nothing else.
    Prompt: {prompt}
    Response: {response}
    Ground Truth: {ground_truth}
    Return a score in range between 1.0 to 0.0 if the response is correct or partially correct (matches or is equivalent to the ground truth), or 0.0 if incorrect.
    final score:[]
    """
    try:
        llm_response = lm(eval_prompt)
        response_bracket_matches = re.findall(r'\[(.*?)\]', llm_response[0])
        if response_bracket_matches:
            response_answer = response_bracket_matches[-1].strip()
        else:
            response_num_matches = re.findall(r'-?\d*\.?\d+', llm_response[0])
            response_answer = response_num_matches[-1] if response_num_matches else None

        if response_answer is not None:
            score = float(response_answer.strip())
            return min(max(score, 0.0), 1.0)
    except Exception:
        pass

    return 0.0


# ============================================================
# Pipeline Generation & Execution
# ============================================================
def generate_pipeline(prompt, policy_model, tokenizer, device, max_steps=3):
    state = {"prompt": prompt, "partial_pipeline": []}
    actions_taken = []
    log_probs = []

    for _ in range(max_steps):
        input_text = f"Prompt: {state['prompt']} Pipeline: {' '.join(state['partial_pipeline'])}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = policy_model(**inputs)
        logits = outputs.logits[:, -1, :]

        if not state["partial_pipeline"]:
            valid_actions = MODULES
        elif len(state["partial_pipeline"]) == 1 and state["partial_pipeline"][0] in MODULES:
            valid_actions = SIGNATURES
        elif len(state["partial_pipeline"]) == 2:
            valid_actions = ["stop"]
        else:
            break

        valid_token_ids = []
        for a in valid_actions:
            tokens = tokenizer.encode(a, add_special_tokens=False)
            if len(tokens) == 0:
                continue
            valid_token_ids.append(tokens[0])

        action_logits = logits[0, valid_token_ids]
        action_probs = torch.softmax(action_logits, dim=-1)
        if torch.isnan(action_probs).any() or action_probs.sum() == 0:
            action_probs = torch.ones(len(valid_actions), device=device) / len(valid_actions)

        try:
            action_choice = torch.multinomial(action_probs, 1).item()
        except RuntimeError:
            action_choice = torch.argmax(action_probs).item()

        action = valid_actions[action_choice]
        log_prob = torch.log(torch.clamp(action_probs[action_choice], min=1e-8))

        actions_taken.append(action)
        log_probs.append(log_prob)
        state["partial_pipeline"].append(action)

        if action == "stop":
            break

    return state["partial_pipeline"], actions_taken, log_probs


def execute_pipeline(prompt, pipeline):
    if len(pipeline) != 2 or pipeline[0] not in MODULES or pipeline[1] not in SIGNATURES:
        return None

    module, signature = pipeline
    if signature.count("->") != 1:
        return None

    match = re.match(r"\s*([a-zA-Z_ ]+)\s*->\s*([a-zA-Z_ ]+)\s*", signature)
    if not match:
        return None

    inputfield = match.group(1).strip().replace(" ", "_").lower()
    outputfield = match.group(2).strip().replace(" ", "_").lower()

    try:
        if module == "CoT":
            program = dspy.ChainOfThought(signature)
        else:
            program = dspy.Predict(signature)

        formatted_input = f"system instruction: Must give your final answer in square brackets without fail! e.g., [final answer] like this: [5] \n prompt: {prompt}"
        response = program(**{inputfield: formatted_input})
        val = response.get(outputfield)
        return str(val) if val is not None else None
    except Exception:
        return None


# ============================================================
# AdaGRPO-DSPy Training Loop (Two-Stage Probe & Expand)
# ============================================================
def train_adagrpo(
    prompts,
    ground_truths,
    policy_model,
    tokenizer,
    device,
    lm,
    num_episodes=30,
    learning_rate=2e-5,
    k_probe=2,
    k_max=6,
    budget=8.8,
    anchor_lambda=0.1,
    save_path="gpt2_trained_policy_model_adagrpo.pt",
    output_metrics="adaptive_metrics.json"
):
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    rewards_history = []
    losses_history = []
    trajectory_logs = []

    total_pipeline_executions = 0
    zero_advantage_count = 0
    rescued_episodes_count = 0
    early_exit_a_count = 0
    early_exit_b_count = 0
    expand_dead_count = 0

    print(f"\n{'='*75}")
    print(f"STARTING AdaGRPO-DSPy ADAPTIVE GROUP SIZING RUN (CSE 498R NSU)")
    print(f"Dataset Size: {len(prompts)} prompts | Episodes: {num_episodes}")
    print(f"Probe Size: {k_probe} | Max Expand: {k_max} | Anchor Lambda: {anchor_lambda}")
    print(f"Policy Network: GPT-2 on {device} | Worker LLM: Ollama (llama3.2:3b)")
    print(f"{'='*75}\n")

    start_wall_clock = time.time()

    for episode in range(num_episodes):
        ep_start_time = time.time()
        idx = episode % len(prompts)
        prompt = prompts[idx]
        ground_truth = ground_truths[idx]

        pipelines = []
        actions_list = []
        log_probs_list = []
        rewards_list = []

        status_flag = ""
        stage_reached = "STAGE_1_PROBE"

        try:
            # ----------------------------------------------------
            # STAGE 1: Informative Probe (k_probe = 2)
            # ----------------------------------------------------
            for _ in range(k_probe):
                pipeline, actions, log_probs = generate_pipeline(prompt, policy_model, tokenizer, device)
                clean_pipeline = pipeline[:-1] if (len(pipeline) > 0 and pipeline[-1] == "stop") else pipeline
                response = execute_pipeline(prompt, clean_pipeline)
                reward = compute_reward(prompt, response, ground_truth, lm)

                pipelines.append(pipeline)
                actions_list.append(actions)
                log_probs_list.append(log_probs)
                rewards_list.append(reward)
                total_pipeline_executions += 1

            probe_mean = float(np.mean(rewards_list))
            probe_std = float(np.std(rewards_list))

            # Branch 1: Mixed Contrast (σ_R > 0) -> Early Exit A
            if probe_std > 0.0:
                status_flag = "EARLY_EXIT_A_ACTIVE"
                early_exit_a_count += 1
                # Take policy step immediately with probe rollouts

            # Branch 2: All-Success (μ_R == 1.0) -> Early Exit B
            elif probe_mean == 1.0:
                status_flag = "EARLY_EXIT_B_SUCCESS"
                early_exit_b_count += 1
                # Mastered query: Lock in progress via Self-Imitation Anchor Loss on CPU

            # Branch 3: All-Failure Deficit (μ_R == 0.0) -> Stage 2: Expand
            else:
                stage_reached = "STAGE_2_EXPAND"
                # Calculate cost-aware expansion ceiling:
                # k_target = clip(floor(B / avg_cost), k_probe, k_max)
                avg_probe_cost = np.mean([get_pipeline_cost(p) for p in pipelines])
                k_target = int(np.clip(np.floor(budget / max(avg_probe_cost, 0.5)), k_probe, k_max))

                expansion_success = False
                for k_step in range(k_probe, k_target):
                    pipeline, actions, log_probs = generate_pipeline(prompt, policy_model, tokenizer, device)
                    clean_pipeline = pipeline[:-1] if (len(pipeline) > 0 and pipeline[-1] == "stop") else pipeline
                    response = execute_pipeline(prompt, clean_pipeline)
                    reward = compute_reward(prompt, response, ground_truth, lm)

                    pipelines.append(pipeline)
                    actions_list.append(actions)
                    log_probs_list.append(log_probs)
                    rewards_list.append(reward)
                    total_pipeline_executions += 1

                    # Early Rescuing Rule: Stop immediately upon finding first success!
                    if reward > 0.0:
                        expansion_success = True
                        rescued_episodes_count += 1
                        status_flag = "EXPAND_RESCUED"
                        break

                if not expansion_success:
                    expand_dead_count += 1
                    status_flag = "EXPAND_FAILED_DEAD"

            # ----------------------------------------------------
            # Advantage Normalization & Policy Gradient Step
            # ----------------------------------------------------
            group_mean = float(np.mean(rewards_list))
            group_std = float(np.std(rewards_list))
            k_total = len(rewards_list)

            # Check if this episode is a dead gradient
            is_zero_advantage = (group_std == 0.0 and status_flag != "EARLY_EXIT_B_SUCCESS")
            if is_zero_advantage:
                zero_advantage_count += 1

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Special Handling for Early Exit B: Self-Imitation Anchor Loss on CPU
            if status_flag == "EARLY_EXIT_B_SUCCESS":
                # L_anchor = -lambda * sum_{i=1}^{k_probe} sum_t log pi_theta(a_{i,t})
                anchor_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for log_probs in log_probs_list:
                    for lp in log_probs:
                        anchor_loss = anchor_loss - lp
                loss = anchor_lambda * anchor_loss
                advantages = [0.0] * k_total

            # Active Learning or Rescued Gradient: Scaled GRPO Advantage
            elif group_std > 0.0:
                # Raw GRPO Advantage: (R_i - mean) / (std + 1e-8)
                raw_advantages = [(r - group_mean) / (group_std + 1e-8) for r in rewards_list]
                # Variance Stabilization Scaling Factor: sqrt(k_probe / k_total)
                var_scale = np.sqrt(float(k_probe) / float(k_total))
                advantages = [adv * var_scale for adv in raw_advantages]

                for k in range(k_total):
                    adv = advantages[k]
                    for log_prob in log_probs_list[k]:
                        loss = loss - log_prob * adv
            else:
                advantages = [0.0] * k_total

            optimizer.zero_grad()
            if loss.requires_grad and loss.grad_fn is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()

            ep_duration = time.time() - ep_start_time
            rewards_history.append(group_mean)
            loss_val = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            losses_history.append(loss_val)

            # Calculate calls saved compared to static K=4
            calls_saved = max(4 - k_total, 0)

            trajectory_logs.append({
                "episode": episode + 1,
                "duration_seconds": round(ep_duration, 2),
                "k_total": k_total,
                "stage": stage_reached,
                "status": status_flag,
                "rewards": [round(r, 2) for r in rewards_list],
                "group_mean": round(group_mean, 3),
                "group_std": round(group_std, 3),
                "loss": round(loss_val, 4),
                "calls_saved": calls_saved
            })

            print(f"Ep {episode+1:02d}/{num_episodes} | Time: {ep_duration:5.2f}s | "
                  f"k: {k_total} | R: {[round(r, 1) for r in rewards_list]} | "
                  f"Mean: {group_mean:.2f} | Loss: {loss_val:6.3f} | [{status_flag}]")

        except Exception as ep_err:
            print(f"Episode {episode+1:02d}/{num_episodes} encountered error: {ep_err}, skipping step.")
            continue

    total_wall_clock = time.time() - start_wall_clock

    torch.save(policy_model.state_dict(), save_path)
    print(f"\nAdaGRPO model weights saved to {save_path}")

    actual_episodes = len(rewards_history)
    zero_adv_pct = (zero_advantage_count / actual_episodes) * 100.0 if actual_episodes > 0 else 0.0
    baseline_executions = actual_episodes * 4
    compute_reduction_pct = ((baseline_executions - total_pipeline_executions) / baseline_executions) * 100.0 if baseline_executions > 0 else 0.0

    metrics = {
        "experiment_name": "AdaGRPO_DSPy_Adaptive_k_v3",
        "num_episodes": actual_episodes,
        "k_probe": k_probe,
        "k_max": k_max,
        "budget": budget,
        "anchor_lambda": anchor_lambda,
        "dataset_size": len(prompts),
        "total_llm_pipeline_executions": total_pipeline_executions,
        "baseline_equivalent_executions": baseline_executions,
        "compute_reduction_percentage": round(compute_reduction_pct, 2),
        "zero_advantage_episodes": zero_advantage_count,
        "zero_advantage_percentage": round(zero_adv_pct, 2),
        "early_exit_a_active_count": early_exit_a_count,
        "early_exit_b_success_count": early_exit_b_count,
        "expand_rescued_count": rescued_episodes_count,
        "expand_failed_dead_count": expand_dead_count,
        "wall_clock_time_seconds": round(total_wall_clock, 2),
        "avg_time_per_episode_seconds": round(total_wall_clock / actual_episodes, 2) if actual_episodes > 0 else 0.0,
        "final_mean_reward": round(float(np.mean(rewards_history[-5:])), 4) if rewards_history else 0.0,
        "rewards_history": [round(r, 4) for r in rewards_history],
        "losses_history": [round(l, 4) for l in losses_history],
        "trajectory_logs": trajectory_logs
    }

    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Adaptive metrics logged to {output_metrics}")

    print("\n" + "="*75)
    print("TABLE 2: AdaGRPO-DSPy EMPIRICAL PERFORMANCE (ADAPTIVE K)")
    print("="*75)
    print(f"Metric 1: Total LLM Pipeline Executions : {total_pipeline_executions} (vs 120 baseline, -{compute_reduction_pct:.1f}%)")
    print(f"Metric 2: Zero-Advantage Episodes       : {zero_advantage_count} / {actual_episodes} ({zero_adv_pct:.1f}% vs 53.3% baseline)")
    print(f"Metric 3: Rescued Gradient Episodes     : {rescued_episodes_count}")
    print(f"Metric 4: Early Exit A (Mixed Signal)   : {early_exit_a_count}")
    print(f"Metric 5: Early Exit B (Anchor Loss)    : {early_exit_b_count}")
    print(f"Metric 6: Total Wall-Clock Time         : {total_wall_clock:.2f} s ({total_wall_clock/60:.2f} min)")
    print(f"Metric 7: Final Mean Reward (Last 5)    : {metrics['final_mean_reward']}")
    print("="*75 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CSE 498R AdaGRPO-DSPy Execution")
    parser.add_argument("--num_episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--k_probe", type=int, default=2, help="Initial probe size (default: 2)")
    parser.add_argument("--k_max", type=int, default=6, help="Maximum expansion ceiling (default: 6)")
    parser.add_argument("--budget", type=float, default=8.8, help="Compute budget B for cost-aware expansion")
    parser.add_argument("--anchor_lambda", type=float, default=0.1, help="Self-imitation anchor loss weight")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama LLM model name")
    parser.add_argument("--save_path", type=str, default="gpt2_trained_policy_model_adagrpo.pt")
    parser.add_argument("--output_metrics", type=str, default="adaptive_metrics.json")
    args = parser.parse_args()

    print(f"Configuring DSPy with Ollama model: {args.model} ...")
    lm = dspy.LM(f"ollama_chat/{args.model}", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)

    print("Loading datasets (GSM8K and HotpotQA)...")
    gsm8k = datasets.load_dataset("gsm8k", "main", split="train[:20]")
    hotpotqa = datasets.load_dataset("hotpot_qa", "fullwiki", split="train[:20]")

    gsm8k_list = list(gsm8k)
    hotpotqa_list = list(hotpotqa)

    data = [(ex["question"], ex["answer"].split("####")[-1].strip()) for ex in gsm8k_list] \
         + [(ex["question"], ex["answer"]) for ex in hotpotqa_list]

    # Exact same seed and prompt slice as baseline for 1-to-1 comparison
    random.seed(42)
    random.shuffle(data)

    data = data[:30]
    prompts, ground_truths = zip(*data)
    prompts = list(prompts)
    ground_truths = list(ground_truths)
    print(f"Prepared 30-prompt evaluation set (Seed 42) for head-to-head comparison.")

    print("Loading GPT-2 policy network on CPU (leaving GPU VRAM for Ollama)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("cpu")
    policy_model = model.to(device)

    train_adagrpo(
        prompts=prompts,
        ground_truths=ground_truths,
        policy_model=policy_model,
        tokenizer=tokenizer,
        device=device,
        lm=lm,
        num_episodes=args.num_episodes,
        learning_rate=2e-5,
        k_probe=args.k_probe,
        k_max=args.k_max,
        budget=args.budget,
        anchor_lambda=args.anchor_lambda,
        save_path=args.save_path,
        output_metrics=args.output_metrics
    )


if __name__ == "__main__":
    main()
