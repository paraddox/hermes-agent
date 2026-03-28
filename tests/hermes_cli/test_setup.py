import json
import yaml

from hermes_cli.auth import _reset_config_provider, _update_config_for_provider, get_active_provider
from hermes_cli.config import get_config_path, load_config, save_config
from hermes_cli.setup import setup_agent_settings, setup_model_provider


def _maybe_keep_current_tts(question, choices):
    if question != "Select TTS provider:":
        return None
    assert choices[-1].startswith("Keep current (")
    return len(choices) - 1


def _clear_provider_env(monkeypatch):
    for key in (
        "NOUS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)



def test_nous_oauth_setup_keeps_current_model_when_syncing_disk_provider(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 1  # Nous Portal
        if question == "Configure vision:":
            return len(choices) - 1
        if question == "Select default model:":
            assert choices[-1] == "Keep current (anthropic/claude-opus-4.6)"
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])

    def _fake_login_nous(*args, **kwargs):
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {}}))
        _update_config_for_provider("nous", "https://inference.example.com/v1")

    monkeypatch.setattr("hermes_cli.auth._login_nous", _fake_login_nous)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://inference.example.com/v1",
            "api_key": "nous-key",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth.fetch_nous_models",
        lambda *args, **kwargs: ["gemini-3-flash"],
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "nous"
    assert reloaded["model"]["base_url"] == "https://inference.example.com/v1"
    assert reloaded["model"]["default"] == "anthropic/claude-opus-4.6"


def test_custom_setup_clears_active_oauth_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {}}))

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 3
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)

    # _model_flow_custom uses builtins.input (URL, key, model, context_length)
    input_values = iter([
        "https://custom.example/v1",
        "custom-api-key",
        "custom/model",
        "",  # context_length (blank = auto-detect)
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_values))
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.models.probe_api_models",
        lambda api_key, base_url: {"models": ["m"], "probed_url": base_url + "/models"},
    )

    setup_model_provider(config)

    # Core assertion: switching to custom endpoint clears OAuth provider
    assert get_active_provider() is None

    # _model_flow_custom writes config via its own load/save cycle
    reloaded = load_config()
    if isinstance(reloaded.get("model"), dict):
        assert reloaded["model"].get("provider") == "custom"
        assert reloaded["model"].get("default") == "custom/model"


def test_codex_setup_uses_runtime_access_token_for_live_model_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 2  # OpenAI Codex
        if question == "Select default model:":
            return 0
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth._login_openai_codex", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-access-token",
        },
    )

    captured = {}

    def _fake_get_codex_model_ids(access_token=None):
        captured["access_token"] = access_token
        return ["gpt-5.2-codex", "gpt-5.2"]

    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        _fake_get_codex_model_ids,
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert captured["access_token"] == "codex-access-token"
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "openai-codex"
    assert reloaded["model"]["default"] == "gpt-5.2-codex"
    assert reloaded["model"]["base_url"] == "https://chatgpt.com/backend-api/codex"


def test_setup_agent_settings_preserves_raw_user_config_without_enabling_defaults(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config_path = get_config_path()
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 10,
                "compression": {"threshold": 0.8, "enabled": False},
                "custom_section": {"keep": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config()

    prompt_values = iter(["90", "all", "0.8"])

    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *args, **kwargs: next(prompt_values),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda question, choices, default=0: 4 if question == "Session reset mode:" else default,
    )

    setup_agent_settings(config)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["compression"]["threshold"] == 0.8
    assert saved["compression"]["enabled"] is False
    assert saved["custom_section"] == {"keep": True}
    assert "browser" not in saved


def test_update_config_for_provider_preserves_raw_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config_path = get_config_path()
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": "old-model",
                "compression": {"threshold": 0.8},
                "custom_section": {"keep": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _update_config_for_provider("nous", "https://inference.example.com/v1")

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["model"]["provider"] == "nous"
    assert saved["model"]["base_url"] == "https://inference.example.com/v1"
    assert saved["model"]["default"] == "old-model"
    assert saved["compression"]["threshold"] == 0.8
    assert saved["custom_section"] == {"keep": True}
    assert "browser" not in saved


def test_reset_config_provider_preserves_raw_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config_path = get_config_path()
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "provider": "nous",
                    "base_url": "https://inference.example.com/v1",
                    "default": "gemini-3-flash",
                },
                "custom_section": {"keep": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _reset_config_provider()

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["model"]["provider"] == "auto"
    assert saved["model"]["base_url"] == "https://openrouter.ai/api/v1"
    assert saved["model"]["default"] == "gemini-3-flash"
    assert saved["custom_section"] == {"keep": True}
    assert "browser" not in saved
