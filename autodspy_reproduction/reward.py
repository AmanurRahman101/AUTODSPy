"""Canonical GSM8K numeric exact match and released LLM reward fallback."""
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
import re
import math

NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+")
BRACKET = re.compile(r"\[([^\[\]]*)\]")


def canonical_answer(raw_answer):
    if "####" not in raw_answer:
        raise ValueError("Official GSM8K answer is missing the #### delimiter")
    return raw_answer.rsplit("####", 1)[1].strip()


def normalize_number(text):
    if text is None:
        return None
    text = str(text).replace("\u2212", "-").strip()
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    brackets = BRACKET.findall(text)
    if brackets:
        matches = NUMBER.findall(brackets[-1])
        # Reject ambiguous bracketed answers such as [3 or 4].
        if len(matches) != 1:
            return None
        number = matches[0]
    else:
        matches = NUMBER.findall(text)
        if not matches:
            return None
        number = matches[-1]
    try:
        value = Decimal(number.replace(",", ""))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def exact_match(response, answer):
    left, right = normalize_number(response), normalize_number(answer)
    return left is not None and right is not None and left == right


def parse_judge_score(text, binary=False):
    brackets = BRACKET.findall(str(text))
    candidate = brackets[-1].strip() if brackets else str(text).strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", candidate):
        raise ValueError(f"Unparseable judge score: {text!r}")
    value = float(candidate)
    if not math.isfinite(value) or (binary and value not in (0.0, 1.0)):
        raise ValueError(f"Invalid judge score: {text!r}")
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True)
class Reward:
    value: float
    exact: bool
    fallback_used: bool = False
    judge_text: str | None = None
    error: str | None = None

    def to_dict(self):
        return asdict(self)


def compute_reward(prompt, response, answer, judge=None, binary=False):
    if response is None:
        return Reward(0.0, False)
    if exact_match(response, answer):
        return Reward(1.0, True)
    if judge is None:
        return Reward(0.0, False)
    if binary:
        instruction = ("Evaluate whether the following response correctly answers the prompt based on the ground truth.\n"
                       "Return a score of 1.0 or 0.0 if the response is correct and 0 if incorrect.\n"
                       "Only return the score inside the square brackets [].")
    else:
        instruction = ("Evaluate whether the following response correctly answers the prompt based on the ground truth "
                       "and give your final score in the square brackets[final score]. only the value in the [] "
                       "should in your response and nothing else.")
    query = f"{instruction}\nPrompt: {prompt}\nResponse: {response}\nGround Truth: {answer}"
    if not binary:
        query += ("\nReturn a score in range between 1.0 to 0.0 if the response is correct or partially correct "
                  "(matches or is equivalent to the ground truth), or 0.0 if incorrect.\nfinal score:[]")
    else:
        query += "\nFinal score:[]"
    text = judge(query)  # Infrastructure errors must abort rather than train on zero rewards.
    try:
        return Reward(parse_judge_score(text, binary), False, True, str(text))
    except ValueError as exc:
        return Reward(0.0, False, True, str(text), str(exc))
