"""Official splits; training never loads the held-out split."""
from dataclasses import dataclass, asdict
import hashlib
import json
import random
from .reward import canonical_answer


@dataclass(frozen=True)
class Example:
    index: int
    question: str
    raw_answer: str
    answer: str
    split: str

    def to_dict(self):
        return asdict(self)


def load_examples(config, split):
    from datasets import load_dataset
    if split not in ("train", "test"):
        raise ValueError("Only official train/test splits are allowed")
    dataset = load_dataset("openai/gsm8k", "main", split=split, revision=config.dataset_revision)
    rows = [Example(i, row["question"], row["answer"], canonical_answer(row["answer"]), split)
            for i, row in enumerate(dataset)]
    manifest = {"dataset": "openai/gsm8k", "config": "main", "split": split,
                "revision": config.dataset_revision, "fingerprint": dataset._fingerprint,
                "official_rows": len(rows)}
    if split == "train" and config.train_pool_size is not None:
        if config.train_pool_size > len(rows):
            raise ValueError("Training pool exceeds official split")
        rows = random.Random(config.seed).sample(rows, config.train_pool_size)
    if split == "test" and config.evaluation_limit is not None:
        if config.evaluation_limit > len(rows):
            raise ValueError("Evaluation limit exceeds official split")
        rows = random.Random(config.seed).sample(rows, config.evaluation_limit)
    manifest["selected_indices"] = [row.index for row in rows]
    manifest["content_sha256"] = hashlib.sha256(json.dumps(
        [row.to_dict() for row in rows], sort_keys=True).encode()).hexdigest()
    return rows, manifest


def check_no_leakage(train, test):
    if any(row.split != "train" for row in train) or any(row.split != "test" for row in test):
        raise ValueError("Incorrect train/test split labels")
    overlap = {r.question.strip() for r in train} & {r.question.strip() for r in test}
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping train/test questions")
