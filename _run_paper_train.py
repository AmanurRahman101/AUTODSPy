import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

NOTEBOOK = Path(__file__).with_name("DSPy_GRPO_statistics.ipynb")
LAST_CELL_INDEX = 30
EVAL_ONLY = "--eval-only" in sys.argv
LEGACY_GPT2_CHECKPOINT = NOTEBOOK.parent / "checkpoints" / "grpo_smoke_log_both_K4_ep30.pt"


def _cli_value(flag):
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{flag} requires a value")
    return sys.argv[index + 1]


def _checkpoint_for(short_name):
    if short_name == "gpt2" and LEGACY_GPT2_CHECKPOINT.is_file():
        return LEGACY_GPT2_CHECKPOINT
    return NOTEBOOK.parent / "checkpoints" / f"grpo_smoke_log_{short_name}_both_K4_ep30.pt"


POLICY_MODEL = _cli_value("--policy-model") or "gpt2"
CHECKPOINT = _checkpoint_for(POLICY_MODEL)
OVERRIDE = {
    "policy_model": POLICY_MODEL,
    "train_dataset": "both",
    "num_episodes": 30,
    "K": 4,
    "run_training": True,
    "run_eval": True,
    "eval_sample_count": 30,
    "random_seed": 42,
    "run_k_sweep": False,
    "run_episode_sweep": False,
}
if EVAL_ONLY:
    OVERRIDE["run_training"] = False
    OVERRIDE["run_eval"] = True
    OVERRIDE["save_path"] = str(CHECKPOINT)
    OVERRIDE["auto_checkpoint_path"] = False
    eval_dataset = _cli_value("--eval-dataset")
    if eval_dataset is not None:
        if eval_dataset not in ("gsm8k", "hotpotqa", "both"):
            raise SystemExit("--eval-dataset must be gsm8k, hotpotqa, or both")
        OVERRIDE["eval_dataset"] = eval_dataset
    eval_start_index = _cli_value("--eval-start-index")
    if eval_start_index is not None:
        OVERRIDE["eval_start_index"] = int(eval_start_index)

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
globs = {"__name__": "__main__"}
os.chdir(NOTEBOOK.parent)

t0 = time.time()
try:
    for i, cell in enumerate(nb["cells"]):
        if i > LAST_CELL_INDEX:
            break
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        print(f"\n>>> Cell {i}", flush=True)
        if src.strip().startswith("!"):
            for line in src.splitlines():
                line = line.strip()
                if not line or not line.startswith("!"):
                    continue
                cmd = line[1:].strip()
                if cmd.startswith("pip "):
                    cmd = f'"{sys.executable}" -m pip {cmd[4:]}'
                r = subprocess.run(cmd, shell=True)
                if r.returncode != 0:
                    raise RuntimeError(f"pip/shell failed cell {i} exit {r.returncode}")
            continue
        exec(compile(src, f"cell_{i}", "exec"), globs)
        if "EXPERIMENT_CONFIG = {" in src and "run_training" in src:
            globs["EXPERIMENT_CONFIG"].update(OVERRIDE)
            short = globs["EXPERIMENT_CONFIG"].get("policy_model", "gpt2")
            mapping = globs.get("MODEL_CONFIG") or {}
            if short not in mapping:
                known = ", ".join(mapping) or "(MODEL_CONFIG missing)"
                raise SystemExit(f"Unknown --policy-model {short!r}. Choose one of: {known}")
            globs["EXPERIMENT_CONFIG"]["policy_model"] = short
            globs["EXPERIMENT_CONFIG"]["policy_model_name"] = mapping[short]
            print("Applied overrides:", OVERRIDE, flush=True)
            print(f"Policy model: {short} -> {mapping[short]}", flush=True)
        if EVAL_ONLY and "trained_model, training_metrics, active_config = run_single_grpo_experiment" in src:
            if not CHECKPOINT.is_file():
                raise FileNotFoundError(CHECKPOINT)
            state = globs["torch"].load(CHECKPOINT, map_location="cpu", weights_only=True)
            policy_model = globs["policy_model"]
            policy_model.load_state_dict(state)
            policy_model.to(globs["policy_device"])
            policy_model.eval()
            print(f"Loaded checkpoint for evaluation only: {CHECKPOINT}", flush=True)
except Exception as e:
    err = str(e).lower()
    if "out of memory" in err or "cuda out of memory" in err:
        print(f"CUDA_OOM: {e}", flush=True)
    else:
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    sys.exit(1)

elapsed = time.time() - t0
metrics = globs.get("training_metrics") or {}
active = globs.get("active_config") or {}
print("EVAL_ONLY_OK" if EVAL_ONLY else "PAPER_TRAIN_OK", flush=True)
print(f"SCRIPT_SECONDS={elapsed:.1f}", flush=True)
print(f"TRAINING_TIME_SECONDS={metrics.get('training_time_seconds')}", flush=True)
print(f"AVG_REWARD={metrics.get('average_reward')}", flush=True)
print(f"FINAL_EPISODE_REWARD={metrics.get('final_episode_reward')}", flush=True)
print(f"DEAD_EPISODE_COUNT={metrics.get('dead_episode_count')}", flush=True)
print(f"DEAD_EPISODE_RATE={metrics.get('dead_episode_rate')}", flush=True)
print(f"REWARD_VAR={metrics.get('reward_variance')}", flush=True)
print(f"CHECKPOINT={active.get('save_path')}", flush=True)
