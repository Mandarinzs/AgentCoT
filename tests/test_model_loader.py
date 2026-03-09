from agentcot.models.loader import resolve_model_name_or_path


def test_resolve_builtin_alias() -> None:
    resolved = resolve_model_name_or_path("Qwen2.5-7B")
    assert resolved.endswith("/Qwen/Qwen2.5-7B-Instruct")


def test_resolve_custom_alias_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENTCOT_MODEL_PATHS", '{"my-model":"/tmp/my-model"}')
    assert resolve_model_name_or_path("my-model") == "/tmp/my-model"
