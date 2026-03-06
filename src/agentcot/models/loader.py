from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_PROVIDERS = {"hf", "modelscope"}


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

    model_path = Path(model_name_or_path)
    resolved = str(model_path) if model_path.exists() else model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    return tokenizer, model
