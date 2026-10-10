"""Released first-token action distribution, including repeated token IDs."""
from dataclasses import dataclass, asdict
import torch
from torch.distributions import Categorical

MODULES = ("CoT", "Predict")
SIGNATURES = (
    "question -> answer", "text -> summary", "question -> reasoning", "question -> hypothesis",
    "problem -> solution", "problem_description -> explanation", "context -> summary", "context -> briefing",
    "word_problem -> solution", "math_problem -> answer", "prompt -> response", "query -> response",
    "text -> response", "prompt -> generated_text", "query -> generated_text",
)


@dataclass(frozen=True)
class Decision:
    state: str
    valid_actions: tuple[str, ...]
    choice: int
    old_log_prob: float


@dataclass(frozen=True)
class Trajectory:
    pipeline: tuple[str, ...]
    decisions: tuple[Decision, ...]

    def to_dict(self):
        return asdict(self)


def action_distribution(vocab_logits, tokenizer, actions):
    ids = [tokenizer.encode(a, add_special_tokens=False) for a in actions]
    if not ids or any(not tokens for tokens in ids):
        raise ValueError("Every action must tokenize to at least one token")
    # Repeated IDs remain repeated slots. Deduplicating changes the experiment.
    logits = vocab_logits[[tokens[0] for tokens in ids]].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite action logits")
    return Categorical(logits=logits)


class Policy:
    def __init__(self, model, tokenizer, max_tokens=1024, gradient_checkpointing=False):
        self.model, self.tokenizer = model, tokenizer
        self.max_tokens = max_tokens
        self.gradient_checkpointing = gradient_checkpointing
        # Explicit replay-consistency assumption; train/eval preserve gradients.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def distribution(self, state, actions):
        if actions == ("stop",):
            # Deterministic termination has probability 1 and needs no GPT-2 call.
            return Categorical(logits=torch.zeros(1, device=self.device))
        inputs = self.tokenizer(state, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_tokens).to(self.device)
        # Exactly the final-position GPT-2 logits; avoid allocating [T, vocab] logits.
        if hasattr(self.model, "transformer") and hasattr(self.model, "lm_head"):
            hidden = self.model.transformer(**inputs, use_cache=False).last_hidden_state
            logits = self.model.lm_head(hidden[:, -1, :])[0]
        else:
            logits = self.model(**inputs).logits[0, -1]
        return action_distribution(logits, self.tokenizer, actions)

    @torch.no_grad()
    def generate(self, prompt, greedy=False):
        self.model.eval()
        pipeline, decisions = [], []
        for actions in (MODULES, SIGNATURES, ("stop",)):
            state = f"Prompt: {prompt} Pipeline: {' '.join(pipeline)}"
            dist = self.distribution(state, actions)
            choice = int(dist.probs.argmax()) if greedy else int(dist.sample())
            decisions.append(Decision(state, actions, choice, float(dist.log_prob(
                torch.tensor(choice, device=self.device)))))
            pipeline.append(actions[choice])
        return Trajectory(tuple(pipeline), tuple(decisions))

    def score(self, decision):
        self.model.train(self.gradient_checkpointing)
        dist = self.distribution(decision.state, decision.valid_actions)
        return dist.log_prob(torch.tensor(decision.choice, device=self.device)), dist.entropy()

    def token_collisions(self):
        result = {}
        for action in MODULES + SIGNATURES + ("stop",):
            token = self.tokenizer.encode(action, add_special_tokens=False)[0]
            result.setdefault(str(token), []).append(action)
        return {token: actions for token, actions in result.items() if len(actions) > 1}


def load_policy(config):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly select device=cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(config.policy_model, revision=config.policy_revision)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(config.policy_model, revision=config.policy_revision)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(config.device)
    return Policy(model, tokenizer, config.max_policy_tokens, config.gradient_checkpointing)
