"""Regression tests for interactive setup provider/model persistence."""

from __future__ import annotations

import copy

from hermes_cli.config import load_config, save_config, save_env_value
from hermes_cli.setup import _print_setup_summary, setup_model_provider
from hermes_cli.mcp_presets import (
    get_bundle_default_checked_servers,
    resolve_provider_mcp_bundle,
)


def _maybe_keep_current_tts(question, choices):
    if question != "Select TTS provider:":
        return None
    assert choices[-1].startswith("Keep current (")
    return len(choices) - 1


def _choice_by_label(choices, label):
    return choices.index(label)


def _read_env(home):
    env_path = home / ".env"
    data = {}
    if not env_path.exists():
        return data
    for line in env_path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k] = v
    return data


def _clear_provider_env(monkeypatch):
    for key in (
        "HERMES_INFERENCE_PROVIDER",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_CN_API_KEY",
        "ANTHROPIC_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _zai_prompt_choice(question, choices, default=0):
    if question == "Select your inference provider:":
        return _choice_by_label(choices, "Z.AI / GLM (Zhipu AI models)")
    if question == "Select default model:":
        return len(choices) - 1
    if question == "Configure vision:":
        return len(choices) - 1
    tts_idx = _maybe_keep_current_tts(question, choices)
    if tts_idx is not None:
        return tts_idx
    raise AssertionError(f"Unexpected prompt_choice call: {question}")


def _zai_prompt(message, *args, **kwargs):
    if "GLM API key" in message:
        return "zai-key"
    return ""


def _patch_common_zai_setup(
    monkeypatch,
    *,
    prompt_yes_no,
    prompt_checklist=None,
    resolve_provider_mcp_bundle=None,
    which=None,
    prompt_choice=_zai_prompt_choice,
    prompt=_zai_prompt,
):
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", prompt)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", prompt_yes_no)
    if prompt_checklist is not None:
        monkeypatch.setattr("hermes_cli.setup.prompt_checklist", prompt_checklist)
    if resolve_provider_mcp_bundle is not None:
        monkeypatch.setattr(
            "hermes_cli.mcp_presets.resolve_provider_mcp_bundle",
            resolve_provider_mcp_bundle,
        )
    if which is not None:
        monkeypatch.setattr("hermes_cli.mcp_presets.shutil.which", which)

    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.auth.get_auth_status",
        lambda provider_id: {"logged_in": False},
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, base_url: [],
    )
    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids",
        lambda provider: [],
    )
    monkeypatch.setattr(
        "hermes_cli.auth.detect_zai_endpoint",
        lambda api_key: {
            "base_url": "https://api.z.ai/api/paas/v4",
            "label": "General",
            "id": "general",
            "model": "glm-5",
        },
    )
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])


def test_setup_keep_current_custom_from_config_does_not_fall_through(tmp_path, monkeypatch):
    """Keep-current custom should not fall through to the generic model menu."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    save_env_value("OPENAI_BASE_URL", "https://example.invalid/v1")
    save_env_value("OPENAI_API_KEY", "custom-key")

    config = load_config()
    config["model"] = {
        "default": "custom/model",
        "provider": "custom",
        "base_url": "https://example.invalid/v1",
    }
    save_config(config)

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            assert choices[-1] == "Keep current (Custom: https://example.invalid/v1)"
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError("Model menu should not appear for keep-current custom")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["default"] == "custom/model"
    assert reloaded["model"]["base_url"] == "https://example.invalid/v1"


def test_setup_custom_endpoint_saves_working_v1_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 3  # Custom endpoint
        if question == "Configure vision:":
            return len(choices) - 1  # Skip
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    # _model_flow_custom uses builtins.input (URL, key, model, context_length)
    input_values = iter([
        "http://localhost:8000",
        "local-key",
        "llm",
        "",  # context_length (blank = auto-detect)
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_values))

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.models.probe_api_models",
        lambda api_key, base_url: {
            "models": ["llm"],
            "probed_url": "http://localhost:8000/v1/models",
            "resolved_base_url": "http://localhost:8000/v1",
            "suggested_base_url": "http://localhost:8000/v1",
            "used_fallback": True,
        },
    )

    setup_model_provider(config)

    env = _read_env(tmp_path)

    # _model_flow_custom saves env vars and config to disk
    assert env.get("OPENAI_BASE_URL") == "http://localhost:8000/v1"
    assert env.get("OPENAI_API_KEY") == "local-key"

    # The model config is saved as a dict by _model_flow_custom
    reloaded = load_config()
    model_cfg = reloaded.get("model", {})
    if isinstance(model_cfg, dict):
        assert model_cfg.get("provider") == "custom"
        assert model_cfg.get("default") == "llm"


def test_setup_keep_current_config_provider_uses_provider_specific_model_menu(tmp_path, monkeypatch):
    """Keep-current should respect config-backed providers, not fall back to OpenRouter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    config["model"] = {
        "default": "claude-opus-4-6",
        "provider": "anthropic",
    }
    save_config(config)

    captured = {"provider_choices": None, "model_choices": None}

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            captured["provider_choices"] = list(choices)
            assert choices[-1] == "Keep current (Anthropic)"
            return len(choices) - 1
        if question == "Configure vision:":
            assert question == "Configure vision:"
            assert choices[-1] == "Skip for now"
            return len(choices) - 1
        if question == "Select default model:":
            captured["model_choices"] = list(choices)
            return len(choices) - 1  # keep current model
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda provider: [])
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    save_config(config)

    assert captured["provider_choices"] is not None
    assert captured["model_choices"] is not None
    assert captured["model_choices"][0] == "claude-opus-4-6"
    assert "anthropic/claude-opus-4.6 (recommended)" not in captured["model_choices"]


def test_setup_keep_current_anthropic_can_configure_openai_vision_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    config["model"] = {
        "default": "claude-opus-4-6",
        "provider": "anthropic",
    }
    save_config(config)

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            assert choices[-1] == "Keep current (Anthropic)"
            return len(choices) - 1
        if question == "Configure vision:":
            return 1
        if question == "Select vision model:":
            assert choices[-1] == "Use default (gpt-4o-mini)"
            return len(choices) - 1
        if question == "Select default model:":
            assert choices[-1] == "Keep current (claude-opus-4-6)"
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda message, *args, **kwargs: "sk-openai" if "OpenAI API key" in message else "",
    )
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda provider: [])
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    env = _read_env(tmp_path)

    assert env.get("OPENAI_API_KEY") == "sk-openai"
    assert env.get("OPENAI_BASE_URL") == "https://api.openai.com/v1"
    assert env.get("AUXILIARY_VISION_MODEL") == "gpt-4o-mini"


def test_setup_zai_bundle_prompts_in_setup_with_bundle_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    captured = {}
    bundle = resolve_provider_mcp_bundle("zai", model="glm-5")
    resolve_calls = []

    def wrapped_resolve_provider_mcp_bundle(provider, model=None):
        resolve_calls.append((provider, model))
        return resolve_provider_mcp_bundle(provider, model=model)

    def fake_prompt_yes_no(question, default=True):
        if question == "Enable z.ai MCP servers during setup?":
            captured["zai_prompt"] = (question, default)
            return True
        return default

    def fake_prompt_checklist(title, items, pre_selected):
        captured["checklist_title"] = title
        captured["checklist_items"] = list(items)
        captured["pre_selected"] = list(pre_selected)
        return list(range(len(items)))

    _patch_common_zai_setup(
        monkeypatch,
        prompt_yes_no=fake_prompt_yes_no,
        prompt_checklist=fake_prompt_checklist,
        resolve_provider_mcp_bundle=wrapped_resolve_provider_mcp_bundle,
        which=lambda cmd: "/usr/bin/npx" if cmd == "npx" else None,
    )
    default_checked, notes = get_bundle_default_checked_servers(bundle)

    setup_model_provider(config)

    assert resolve_calls == [("zai", "glm-5")]
    assert notes == []
    assert captured["checklist_title"] == bundle["prompt"]["checklist_title"]
    assert set(captured["checklist_items"]) == set(bundle["servers"])
    expected_pre_selected = [
        captured["checklist_items"].index(name)
        for name in default_checked
        if name in captured["checklist_items"]
    ]
    assert captured["pre_selected"] == expected_pre_selected
    assert captured["zai_prompt"] == (
        bundle["prompt"]["question"],
        bundle["prompt"]["default"],
    )
    assert {
        captured["checklist_items"][idx]
        for idx in captured["pre_selected"]
    } == {"web-search-prime", "web-reader", "zread", "zai-vision"}

    saved = load_config().get("mcp_servers", {})
    assert set(saved.keys()) == set(bundle["servers"])
    for name, server_data in bundle["servers"].items():
        expected_cfg = server_data["config"]
        assert saved[name]["enabled"] is True
        for key, value in expected_cfg.items():
            if key == "headers":
                assert saved[name][key]["Authorization"].startswith("Bearer ")
                assert "zai-key" in saved[name][key]["Authorization"]
                continue
            if key == "env":
                for env_key, env_value in value.items():
                    if env_value == "${GLM_API_KEY}":
                        assert saved[name][key][env_key] == "zai-key"
                    else:
                        assert saved[name][key][env_key] == env_value
                continue
            assert saved[name][key] == value


def test_setup_zai_bundle_preselects_http_servers_and_shows_note_without_npx(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    captured = {}
    bundle = resolve_provider_mcp_bundle("zai", model="glm-5")

    def fake_prompt_yes_no(question, default=True):
        if question == "Enable z.ai MCP servers during setup?":
            captured["zai_prompt"] = (question, default)
            return True
        return default

    def fake_prompt_checklist(title, items, pre_selected):
        captured["checklist_title"] = title
        captured["checklist_items"] = list(items)
        captured["pre_selected"] = list(pre_selected)
        return list(range(3))

    _patch_common_zai_setup(
        monkeypatch,
        prompt_yes_no=fake_prompt_yes_no,
        prompt_checklist=fake_prompt_checklist,
        which=lambda _cmd: None,
    )
    default_checked, notes = get_bundle_default_checked_servers(bundle)

    setup_model_provider(config)

    assert captured["checklist_title"] == bundle["prompt"]["checklist_title"]
    assert set(captured["checklist_items"]) == set(bundle["servers"])
    expected_pre_selected = [
        captured["checklist_items"].index(name)
        for name in default_checked
        if name in captured["checklist_items"]
    ]
    assert captured["pre_selected"] == expected_pre_selected
    assert captured["zai_prompt"] == (
        bundle["prompt"]["question"],
        bundle["prompt"]["default"],
    )
    pre_selected_names = [
        captured["checklist_items"][idx]
        for idx in captured["pre_selected"]
    ]
    assert pre_selected_names == ["web-search-prime", "web-reader", "zread"]
    assert "zai-vision" not in pre_selected_names
    output = capsys.readouterr().out
    assert notes == ["`zai-vision` was not preselected because `npx` is unavailable."]
    assert notes[0] in output


def test_setup_zai_bundle_merge_preserves_existing_entries_and_reports_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    existing_entry = {
        "url": "https://example.invalid/existing",
        "enabled": False,
        "tools": {"exclude": ["foo"]},
    }
    base = load_config()
    base["mcp_servers"] = {
        "web-reader": copy.deepcopy(existing_entry),
    }
    save_config(base)
    config = load_config()

    def fake_prompt_yes_no(question, default=True):
        if question == "Enable z.ai MCP servers during setup?":
            return True
        return default

    def fake_prompt_checklist(title, items, pre_selected):
        return list(range(len(items)))

    _patch_common_zai_setup(
        monkeypatch,
        prompt_yes_no=fake_prompt_yes_no,
        prompt_checklist=fake_prompt_checklist,
        which=lambda cmd: "/usr/bin/npx" if cmd == "npx" else None,
    )
    bundle = resolve_provider_mcp_bundle("zai", model="glm-5")

    setup_model_provider(config)

    output = capsys.readouterr().out
    reloaded = load_config()

    assert reloaded["mcp_servers"]["web-reader"] == existing_entry
    assert set(reloaded["mcp_servers"].keys()) == set(bundle["servers"]) | {"web-reader"}
    assert "MCP servers added:" in output
    assert "web-search-prime" in output
    assert "zread" in output
    assert "MCP servers already configured (not overwritten): web-reader" in output


def test_setup_zai_bundle_skips_runtime_name_alias_collision(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    existing_entry = {
        "command": "npx",
        "args": ["-y", "old-server"],
    }
    base = load_config()
    base["mcp_servers"] = {
        "zai_vision": copy.deepcopy(existing_entry),
    }
    save_config(base)
    config = load_config()

    def fake_prompt_yes_no(question, default=True):
        if question == "Enable z.ai MCP servers during setup?":
            return True
        return default

    def fake_prompt_checklist(title, items, pre_selected):
        return list(range(len(items)))

    _patch_common_zai_setup(
        monkeypatch,
        prompt_yes_no=fake_prompt_yes_no,
        prompt_checklist=fake_prompt_checklist,
        which=lambda cmd: "/usr/bin/npx" if cmd == "npx" else None,
    )

    setup_model_provider(config)

    output = capsys.readouterr().out
    reloaded = load_config()

    assert reloaded["mcp_servers"]["zai_vision"] == existing_entry
    assert "zai-vision" not in reloaded["mcp_servers"]
    assert (
        "MCP servers skipped because an equivalent server is already configured: "
        "zai-vision (matches existing zai_vision)"
    ) in output


def test_setup_zai_bundle_skips_equivalent_existing_server_by_identity(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    existing_entry = {
        "url": "https://api.z.ai/api/mcp/zread/mcp",
        "headers": {"Authorization": "Bearer ${GLM_API_KEY}"},
        "enabled": False,
        "tools": {"exclude": ["foo"]},
        "timeout": 120,
    }
    base = load_config()
    base["mcp_servers"] = {
        "custom-zread": copy.deepcopy(existing_entry),
    }
    save_config(base)
    monkeypatch.setenv("GLM_API_KEY", "zai-key")
    config = load_config()

    def fake_prompt_yes_no(question, default=True):
        if question == "Enable z.ai MCP servers during setup?":
            return True
        return default

    def fake_prompt_checklist(title, items, pre_selected):
        return list(range(len(items)))

    _patch_common_zai_setup(
        monkeypatch,
        prompt_yes_no=fake_prompt_yes_no,
        prompt_checklist=fake_prompt_checklist,
        which=lambda cmd: "/usr/bin/npx" if cmd == "npx" else None,
    )

    setup_model_provider(config)

    output = capsys.readouterr().out
    reloaded = load_config()

    assert reloaded["mcp_servers"]["custom-zread"]["url"] == existing_entry["url"]
    assert reloaded["mcp_servers"]["custom-zread"]["enabled"] is False
    assert reloaded["mcp_servers"]["custom-zread"]["tools"] == {"exclude": ["foo"]}
    assert reloaded["mcp_servers"]["custom-zread"]["timeout"] == 120
    assert "zread" not in reloaded["mcp_servers"]
    assert (
        "MCP servers already configured under a different name (not duplicated): "
        "zread (matches existing custom-zread)"
    ) in output


def test_setup_zai_bundle_skips_equivalent_unsaved_in_memory_server(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    config["mcp_servers"] = {
        "custom-zread": {
            "url": "https://api.z.ai/api/mcp/zread/mcp",
            "headers": {"Authorization": "Bearer zai-key"},
            "enabled": False,
        }
    }

    def fake_prompt_yes_no(question, default=True):
        if question == "Enable z.ai MCP servers during setup?":
            return True
        return default

    def fake_prompt_checklist(title, items, pre_selected):
        return list(range(len(items)))

    _patch_common_zai_setup(
        monkeypatch,
        prompt_yes_no=fake_prompt_yes_no,
        prompt_checklist=fake_prompt_checklist,
        which=lambda cmd: "/usr/bin/npx" if cmd == "npx" else None,
    )

    setup_model_provider(config)

    output = capsys.readouterr().out
    reloaded = load_config()

    assert reloaded["mcp_servers"]["custom-zread"]["url"] == "https://api.z.ai/api/mcp/zread/mcp"
    assert "zread" not in reloaded["mcp_servers"]
    assert (
        "MCP servers already configured under a different name (not duplicated): "
        "zread (matches existing custom-zread)"
    ) in output


def test_setup_non_zai_provider_with_no_bundle_does_not_enter_mcp_prompt_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    resolve_calls = []

    def wrapped_resolve_provider_mcp_bundle(provider, model=None):
        resolve_calls.append((provider, model))
        return resolve_provider_mcp_bundle(provider, model=model)

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return _choice_by_label(choices, "OpenRouter API key (100+ models, pay-per-use)")
        if question == "Configure vision:":
            return len(choices) - 1
        if question == "Select default model:":
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    def fake_prompt(message, *args, **kwargs):
        if message.startswith("  OpenRouter API key"):
            return "or-key"
        return ""

    def fake_prompt_yes_no(question, default=True):
        raise AssertionError(f"Unexpected prompt_yes_no call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", fake_prompt_yes_no)
    monkeypatch.setattr(
        "hermes_cli.mcp_presets.resolve_provider_mcp_bundle",
        wrapped_resolve_provider_mcp_bundle,
    )
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.auth.get_auth_status",
        lambda provider_id: {"logged_in": False},
    )
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda api_key, base_url: [])
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    assert resolve_calls == [("openrouter", "anthropic/claude-opus-4.6")]


def test_setup_zai_bundle_decline_stops_after_prompt_without_checklist(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    base = load_config()
    base["mcp_servers"] = {
        "existing": {"command": "npx", "args": ["-y", "example"]},
    }
    save_config(base)
    config = load_config()
    original_mcp_servers = copy.deepcopy(config.get("mcp_servers", {}))
    event_log = []
    bundle = resolve_provider_mcp_bundle("zai", model="glm-5")
    resolve_calls = []

    def wrapped_resolve_provider_mcp_bundle(provider, model=None):
        resolve_calls.append((provider, model))
        return resolve_provider_mcp_bundle(provider, model=model)

    def logging_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            event_log.append("provider_prompt")
        elif question == "Select default model:":
            event_log.append("model_prompt")
        elif question == "Configure vision:":
            event_log.append("vision_prompt")
        elif question == "Select TTS provider:":
            event_log.append("tts_prompt")
        return _zai_prompt_choice(question, choices, default)

    def fake_prompt_checklist(*args, **kwargs):
        event_log.append("checklist")
        raise AssertionError("Checklist should not run when MCP setup is declined.")

    zai_prompt_seen = {"called": False, "question": None, "default": None}

    def fake_prompt_yes_no(question, default=True):
        if "Enable z.ai MCP servers during setup?" in question:
            zai_prompt_seen["called"] = True
            zai_prompt_seen["question"] = question
            zai_prompt_seen["default"] = default
            event_log.append("zai_prompt")
            return False
        event_log.append("other_yes_no")
        return default

    _patch_common_zai_setup(
        monkeypatch,
        prompt_yes_no=fake_prompt_yes_no,
        prompt_checklist=fake_prompt_checklist,
        resolve_provider_mcp_bundle=wrapped_resolve_provider_mcp_bundle,
        prompt_choice=logging_prompt_choice,
    )

    setup_model_provider(config)

    reloaded = load_config()
    assert reloaded.get("mcp_servers", {}) == original_mcp_servers
    assert resolve_calls == [("zai", "glm-5")]
    assert zai_prompt_seen["called"] is True
    assert zai_prompt_seen["question"] == bundle["prompt"]["question"]
    assert zai_prompt_seen["default"] == bundle["prompt"]["default"]
    assert "provider_prompt" in event_log
    assert "model_prompt" in event_log
    assert "zai_prompt" in event_log
    assert "checklist" not in event_log
    assert event_log.index("model_prompt") < event_log.index("zai_prompt")


def test_setup_zai_bundle_failure_warns_and_continues(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    prompts_seen = []

    def fake_prompt_yes_no(question, default=True):
        prompts_seen.append(question)
        return default

    _patch_common_zai_setup(
        monkeypatch,
        prompt_yes_no=fake_prompt_yes_no,
        resolve_provider_mcp_bundle=lambda provider, model=None: (_ for _ in ()).throw(
            ValueError("broken preset data")
        ),
    )

    setup_model_provider(config)

    output = capsys.readouterr().out
    reloaded = load_config()

    assert "Failed to resolve provider-linked MCP presets" in output
    assert "Enable z.ai MCP servers during setup?" not in prompts_seen
    assert "mcp_servers" not in reloaded or reloaded["mcp_servers"] == {}
    model_cfg = reloaded.get("model", {})
    assert isinstance(model_cfg, dict)
    assert model_cfg.get("provider") == "zai"
    assert model_cfg.get("default") == "glm-5"


def test_setup_copilot_uses_gh_auth_and_saves_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return _choice_by_label(choices, "GitHub Copilot (uses GITHUB_TOKEN or gh auth token)")
        if question == "Select default model:":
            assert "gpt-4.1" in choices
            assert "gpt-5.4" in choices
            return choices.index("gpt-5.4")
        if question == "Select reasoning effort:":
            assert "low" in choices
            assert "high" in choices
            return choices.index("high")
        if question == "Configure vision:":
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    def fake_prompt(message, *args, **kwargs):
        raise AssertionError(f"Unexpected prompt call: {message}")

    def fake_get_auth_status(provider_id):
        if provider_id == "copilot":
            return {"logged_in": True}
        return {"logged_in": False}

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth.get_auth_status", fake_get_auth_status)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "gh-cli-token",
            "base_url": "https://api.githubcopilot.com",
            "source": "gh auth token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_github_model_catalog",
        lambda api_key: [
            {
                "id": "gpt-4.1",
                "capabilities": {"type": "chat", "supports": {}},
                "supported_endpoints": ["/chat/completions"],
            },
            {
                "id": "gpt-5.4",
                "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}},
                "supported_endpoints": ["/responses"],
            },
        ],
    )
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    save_config(config)

    env = _read_env(tmp_path)
    reloaded = load_config()

    assert env.get("GITHUB_TOKEN") is None
    assert reloaded["model"]["provider"] == "copilot"
    assert reloaded["model"]["base_url"] == "https://api.githubcopilot.com"
    assert reloaded["model"]["default"] == "gpt-5.4"
    assert reloaded["model"]["api_mode"] == "codex_responses"
    assert reloaded["agent"]["reasoning_effort"] == "high"


def test_setup_copilot_acp_uses_model_picker_and_saves_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return _choice_by_label(choices, "GitHub Copilot ACP (spawns `copilot --acp --stdio`)")
        if question == "Select default model:":
            assert "gpt-4.1" in choices
            assert "gpt-5.4" in choices
            return choices.index("gpt-5.4")
        if question == "Configure vision:":
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    def fake_prompt(message, *args, **kwargs):
        raise AssertionError(f"Unexpected prompt call: {message}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth.get_auth_status", lambda provider_id: {"logged_in": provider_id == "copilot-acp"})
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": "copilot",
            "api_key": "gh-cli-token",
            "base_url": "https://api.githubcopilot.com",
            "source": "gh auth token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_github_model_catalog",
        lambda api_key: [
            {
                "id": "gpt-4.1",
                "capabilities": {"type": "chat", "supports": {}},
                "supported_endpoints": ["/chat/completions"],
            },
            {
                "id": "gpt-5.4",
                "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}},
                "supported_endpoints": ["/responses"],
            },
        ],
    )
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert reloaded["model"]["provider"] == "copilot-acp"
    assert reloaded["model"]["base_url"] == "acp://copilot"
    assert reloaded["model"]["default"] == "gpt-5.4"
    assert reloaded["model"]["api_mode"] == "chat_completions"


def test_setup_switch_custom_to_codex_clears_custom_endpoint_and_updates_config(tmp_path, monkeypatch):
    """Switching from custom to Codex should clear custom endpoint overrides."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    save_env_value("OPENAI_BASE_URL", "https://example.invalid/v1")
    save_env_value("OPENAI_API_KEY", "sk-custom")
    save_env_value("OPENROUTER_API_KEY", "sk-or")

    config = load_config()
    config["model"] = {
        "default": "custom/model",
        "provider": "custom",
        "base_url": "https://example.invalid/v1",
    }
    save_config(config)

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
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth._login_openai_codex", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-...oken",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        lambda **kwargs: ["openai/gpt-5.3-codex", "openai/gpt-5-codex-mini"],
    )

    setup_model_provider(config)
    save_config(config)

    env = _read_env(tmp_path)
    reloaded = load_config()

    assert env.get("OPENAI_BASE_URL") == ""
    assert env.get("OPENAI_API_KEY") == ""
    assert reloaded["model"]["provider"] == "openai-codex"
    assert reloaded["model"]["default"] == "openai/gpt-5.3-codex"
    assert reloaded["model"]["base_url"] == "https://chatgpt.com/backend-api/codex"


def test_setup_summary_marks_codex_auth_as_vision_available(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    (tmp_path / "auth.json").write_text(
        '{"active_provider":"openai-codex","providers":{"openai-codex":{"tokens":{"access_token": "***", "refresh_token": "***"}}}}'
    )

    monkeypatch.setattr("shutil.which", lambda _name: None)

    _print_setup_summary(load_config(), tmp_path)
    output = capsys.readouterr().out

    assert "Vision (image analysis)" in output
    assert "missing run 'hermes setup' to configure" not in output
    assert "Mixture of Agents" in output
    assert "missing OPENROUTER_API_KEY" in output


def test_setup_summary_marks_anthropic_auth_as_vision_available(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-key")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: ["anthropic"])

    _print_setup_summary(load_config(), tmp_path)
    output = capsys.readouterr().out

    assert "Vision (image analysis)" in output
    assert "missing run 'hermes setup' to configure" not in output
