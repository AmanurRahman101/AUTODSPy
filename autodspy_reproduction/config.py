"""Strict configuration; paper settings and active algorithm settings are separate."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class Config:
    seed: int = 42
    policy_model: str = "gpt2"
    policy_revision: str = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    dataset_revision: str = "740312add88f781978c0658806c59bc2815b9866"
    device: str = "cuda"
    episodes: int = 200
    samples_per_episode: int = 200
    group_size: int = 5
    learning_rate: float = 1e-4
    objective: str = "notebook"
    advantage_mode: str = "centered"
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.0
    paper_gamma: float = 0.99
    paper_lambda: float = 0.95
    paper_entropy_coefficient: float = 0.01
    train_pool_size: int | None = None
    max_policy_tokens: int = 1024
    gradient_checkpointing: bool = False
    cpu_threads: int = 4
    checkpoint_every: int = 10
    execution_model: str = "llama3.1:8b"
    execution_endpoint: str = "http://localhost:11434"
    execution_num_gpu: int = 0
    execution_temperature: float = 0.0
    execution_max_tokens: int = 4096
    execution_context: int = 8192
    execution_timeout: int = 300
    execution_keep_alive: str = "5m"
    semantic_fallback: bool = True
    evaluation_limit: int | None = None
    output_root: str = "runs/reproduction"

    def validate(self):
        if self.policy_model != "gpt2" or self.execution_model != "llama3.1:8b":
            raise ValueError("This reproduction requires GPT-2 and llama3.1:8b; no substitution.")
        for name in ("episodes", "samples_per_episode", "cpu_threads", "checkpoint_every", "max_policy_tokens",
                     "execution_max_tokens", "execution_context", "execution_timeout"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("group_size must be >= 2")
        if not 1 <= self.max_policy_tokens <= 1024:
            raise ValueError("GPT-2 supports at most 1024 policy tokens")
        if self.objective not in ("notebook", "clipped"):
            raise ValueError("objective must be notebook or clipped")
        if self.advantage_mode not in ("centered", "standardized"):
            raise ValueError("advantage_mode must be centered or standardized")
        if self.objective == "notebook" and self.entropy_coefficient != 0:
            raise ValueError("Released notebook objective has no entropy regularization")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.clip_epsilon < 1 or not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError("Invalid clipping or entropy settings")
        if self.execution_num_gpu < 0:
            raise ValueError("Specify CPU (0) or an explicit positive Ollama GPU-layer count")
        if not math.isfinite(self.execution_temperature) or self.execution_temperature < 0:
            raise ValueError("Invalid execution temperature")
        for name in ("train_pool_size", "evaluation_limit"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be null or a positive integer")
        if not self.execution_endpoint.startswith(("http://", "https://")):
            raise ValueError("execution_endpoint must be an HTTP(S) Ollama URL")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(filename):
    data = yaml.safe_load(Path(filename).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return Config(**data).validate()
