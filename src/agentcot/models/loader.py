from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_PROVIDERS = {"hf", "modelscope"}
DEFAULT_MODEL_PATHS = {
    "Llama3-8B": "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct",
    "Qwen2.5-7B": "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct",
}
MODEL_PATHS_ENV = "AGENTCOT_MODEL_PATHS"


def _load_model_paths_from_env() -> dict[str, str]:
    """Load model alias mapping from env var.

    AGENTCOT_MODEL_PATHS expects a JSON object, e.g.
    {"Qwen2.5-7B": "/path/to/local/model"}
    """
    raw = os.getenv(MODEL_PATHS_ENV)
    if not raw:
        return {}

    parsed: Any = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{MODEL_PATHS_ENV} must be a JSON object")

    normalized: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{MODEL_PATHS_ENV} keys and values must be strings")
        normalized[key] = value
    return normalized


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    """Resolve a model alias to a local path if present in mapping."""
    aliases = {**DEFAULT_MODEL_PATHS, **_load_model_paths_from_env()}
    candidate = aliases.get(model_name_or_path, model_name_or_path)
    model_path = Path(candidate)
    return str(model_path) if model_path.exists() else candidate


def load_tokenizer_and_model(
    model_name_or_path: str,
    provider: str = "hf",
    trust_remote_code: bool = True,
):
    """Load model/tokenizer from HF or ModelScope compatible path.

    For ModelScope, local models exported in HuggingFace format are supported.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    resolved = resolve_model_name_or_path(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    return tokenizer, model
