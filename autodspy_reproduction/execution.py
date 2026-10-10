"""Sequential DSPy execution through a validated, configurable Ollama server."""
from dataclasses import dataclass, asdict
import json
import time
import urllib.request
from .policy import MODULES, SIGNATURES


def ollama_json(endpoint, route, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + route, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def preflight(config):
    version = ollama_json(config.execution_endpoint, "/api/version")
    models = ollama_json(config.execution_endpoint, "/api/tags")["models"]
    found = next((m for m in models if m.get("name") == config.execution_model
                  or m.get("model") == config.execution_model), None)
    if found is None:
        raise RuntimeError(f"Required {config.execution_model} missing at {config.execution_endpoint}. "
                           "Install it on that server; smaller models are not substitutes.")
    info = ollama_json(config.execution_endpoint, "/api/show", {"model": config.execution_model})
    details = info.get("details", {})
    if details.get("family") != "llama" or not str(details.get("parameter_size", "")).startswith("8"):
        raise RuntimeError("Ollama tag does not identify an 8B Llama model")
    if not details.get("quantization_level"):
        raise RuntimeError("Cannot record execution-model quantization")
    model_info = info.get("model_info", {})
    if "Llama-3.1" not in str(model_info.get("general.basename", "")):
        raise RuntimeError("Cannot verify that this tag contains Llama 3.1")
    # License text and tensor listings are large and do not identify runtime behavior.
    recorded_info = {key: info.get(key) for key in ("details", "model_info", "parameters", "template", "capabilities")}
    return {"version": version, "model": found, "show": recorded_info,
            "endpoint": config.execution_endpoint, "settings": {
                "num_gpu": config.execution_num_gpu, "temperature": config.execution_temperature,
                "num_predict": config.execution_max_tokens, "num_ctx": config.execution_context,
                "seed": config.seed, "keep_alive": config.execution_keep_alive,
                "timeout": config.execution_timeout, "cache": False}}


@dataclass(frozen=True)
class Execution:
    response: str | None
    seconds: float
    error: str | None = None

    def to_dict(self):
        return asdict(self)


def pipeline_pair(pipeline):
    pipeline = tuple(pipeline)
    if pipeline and pipeline[-1] == "stop":
        pipeline = pipeline[:-1]
    if len(pipeline) != 2 or pipeline[0] not in MODULES or pipeline[1] not in SIGNATURES:
        return None
    return pipeline


class DSPyExecutor:
    def __init__(self, config):
        import dspy
        self.config = config
        self.lm = dspy.LM("ollama_chat/" + config.execution_model,
                          api_base=config.execution_endpoint, api_key="",
                          temperature=config.execution_temperature, max_tokens=config.execution_max_tokens,
                          num_ctx=config.execution_context, num_gpu=config.execution_num_gpu,
                          seed=config.seed, keep_alive=config.execution_keep_alive,
                          timeout=config.execution_timeout, cache=False, num_retries=0)

    def runtime(self):
        return ollama_json(self.config.execution_endpoint, "/api/ps")

    def execute(self, prompt, pipeline):
        import dspy
        pair = pipeline_pair(pipeline)
        if pair is None:
            return Execution(None, 0.0, "invalid_pipeline")
        module, signature = pair
        input_field, output_field = (field.strip() for field in signature.split("->"))
        program = dspy.ChainOfThought(signature) if module == "CoT" else dspy.Predict(signature)
        formatted = ("system instruction: Must give your final answer in square brackets without fail! "
                     f"e.g., [final answer] like this: [5] \n prompt: {prompt}")
        started = time.perf_counter()
        # Same executor/settings/instructions are used for learned and static methods.
        from dspy.utils.exceptions import AdapterParseError
        try:
            with dspy.context(lm=self.lm, disable_history=True):
                result = program(**{input_field: formatted})
        except AdapterParseError as exc:
            return Execution(None, time.perf_counter() - started, f"adapter_parse_error: {exc}")
        value = result.get(output_field)
        return Execution(str(value) if value is not None else None,
                         time.perf_counter() - started, None if value is not None else "missing_output_field")

    def judge(self, prompt):
        import dspy
        with dspy.context(disable_history=True):
            return self.lm(prompt)[0]
