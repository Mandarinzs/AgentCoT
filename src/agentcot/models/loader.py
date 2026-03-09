from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SUPPORTED_PROVIDERS = {"hf", "modelscope"}


def load_tokenizer_and_model(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
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

    tokenizer_source = tokenizer_name_or_path or model_name_or_path
    tokenizer_path = Path(tokenizer_source)
    tokenizer_resolved = str(tokenizer_path) if tokenizer_path.exists() else tokenizer_source

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_resolved, trust_remote_code=trust_remote_code)

    model_dir = Path(resolved)
    has_weights = model_dir.is_dir() and any(
        (model_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")
    )

    if model_dir.is_dir() and not has_weights:
        config = AutoConfig.from_pretrained(resolved, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=trust_remote_code)

    return tokenizer, model
