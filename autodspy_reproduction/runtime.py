"""Run isolation, provenance, episode-boundary checkpoints and deterministic RNG."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import uuid
import numpy as np
import torch


def seed_everything(seed, cpu_threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # On Windows CUDA set CUBLAS_WORKSPACE_CONFIG before starting Python.
    torch.use_deterministic_algorithms(True)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def new_run(output_root, label="train"):
    parent = Path(output_root)
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + label + "-" + uuid.uuid4().hex[:8])
    run.mkdir()  # Never reuse or clear an existing run.
    return run


def shell_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_manifest():
    versions = {}
    for package in ("torch", "transformers", "dspy", "datasets", "numpy", "PyYAML", "litellm", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parent.parent
    sources = sorted((root / "autodspy_reproduction").glob("*.py")) + sorted((root / "configs").glob("grpo*.yaml"))
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions,
            "source_sha256": {str(file.relative_to(root)): artifact_hash(file) for file in sources},
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "git_head": shell_output(["git", "rev-parse", "HEAD"]),
            "git_status": shell_output(["git", "status", "--short"]),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvidia_smi": shell_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                                        "--format=csv,noheader"])}


def cuda_memory(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        return {"policy_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "policy_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    return {"policy_peak_allocated_bytes": None, "policy_peak_reserved_bytes": None}


def artifact_hash(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(filename, policy, optimizer, config, episode, sampler, provenance):
    filename = Path(filename)
    if filename.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {filename}")
    payload = {"format_version": 1, "model": policy.model.state_dict(),
               "optimizer": optimizer.state_dict() if optimizer is not None else None,
               "config": config.to_dict(), "completed_episodes": episode,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "sampler_rng": sampler.bit_generator.state, "provenance": provenance}
    temporary = filename.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(filename)


def load_checkpoint(filename, policy, optimizer=None, sampler=None, expected_config=None, restore_rng=False):
    payload = torch.load(filename, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Not an isolated reproduction checkpoint; legacy weights require a separate conversion")
    if expected_config is not None and payload["config"] != expected_config.to_dict():
        raise ValueError("Resume config differs from checkpoint; resume requires identical settings")
    policy.model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("Initial-policy checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        if sampler is None:
            raise ValueError("A sampler is required when restoring RNG")
        sampler.bit_generator.state = payload["sampler_rng"]
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        if payload["cuda_rng"]:
            if not torch.cuda.is_available():
                raise ValueError("CUDA RNG cannot be resumed without CUDA")
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload
