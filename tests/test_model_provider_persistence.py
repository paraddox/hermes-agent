"""Tests that provider selection via `hermes model` always persists correctly.

Regression tests for the bug where _save_model_choice could save config.model
as a plain string, causing subsequent provider writes (which check
isinstance(model, dict)) to silently fail — leaving the provider unset and
falling back to auto-detection.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
import yaml


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal string-format config."""
    home = tmp_path / "hermes"
    home.mkdir()
    config_yaml = home / "config.yaml"
    # Start with model as a plain string — the format that triggered the bug
    config_yaml.write_text("model: some-old-model\n")
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear env vars that could interfere
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return home


class TestSaveModelChoiceAlwaysDict:
    def test_string_model_becomes_dict(self, config_home):
        """When config.model is a plain string, _save_model_choice must
        convert it to a dict so provider can be set afterwards."""
        from hermes_cli.auth import _save_model_choice

        _save_model_choice("kimi-k2.5")

        import yaml
        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict), (
            f"Expected model to be a dict after save, got {type(model)}: {model}"
        )
        assert model["default"] == "kimi-k2.5"

    def test_dict_model_stays_dict(self, config_home):
        """When config.model is already a dict, _save_model_choice preserves it."""
        (config_home / "config.yaml").write_text(
            "model:\n  default: old-model\n  provider: openrouter\n"
        )
        from hermes_cli.auth import _save_model_choice

        _save_model_choice("new-model")

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model["default"] == "new-model"
        assert model["provider"] == "openrouter"  # preserved

    def test_save_model_choice_preserves_raw_user_config(self, config_home):
        """Saving a model choice must not rewrite unrelated defaults into config.yaml."""
        (config_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": "old-model",
                    "compression": {"threshold": 0.8, "enabled": False},
                    "custom_section": {"keep": True},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        from hermes_cli.auth import _save_model_choice

        _save_model_choice("new-model")

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        assert config["model"]["default"] == "new-model"
        assert config["compression"] == {"threshold": 0.8, "enabled": False}
        assert config["custom_section"] == {"keep": True}
        assert "terminal" not in config
        assert "browser" not in config

    def test_save_model_choice_preserves_raw_env_placeholders(self, config_home, monkeypatch):
        """Saving a model choice must not expand unrelated env placeholders."""
        monkeypatch.setenv("GLM_API_KEY", "secret-key-123")
        (config_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": "old-model",
                    "mcp_servers": {
                        "zread": {
                            "url": "https://example.com",
                            "headers": {"Authorization": "Bearer ${GLM_API_KEY}"},
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        from hermes_cli.auth import _save_model_choice

        _save_model_choice("new-model")

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        assert config["model"]["default"] == "new-model"
        assert (
            config["mcp_servers"]["zread"]["headers"]["Authorization"]
            == "Bearer ${GLM_API_KEY}"
        )

    def test_model_flow_openrouter_preserves_raw_user_config(self, config_home, monkeypatch):
        """OpenRouter model selection must not rewrite unrelated defaults."""
        (config_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": "old-model",
                    "compression": {"enabled": False},
                    "custom_section": {"keep": True},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_openrouter

        with patch("hermes_cli.models.model_ids", return_value=["new-model"]), patch(
            "hermes_cli.auth._prompt_model_selection", return_value="new-model"
        ), patch("hermes_cli.auth.deactivate_provider"):
            _model_flow_openrouter(load_config(), current_model="old-model")

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        assert config["model"]["default"] == "new-model"
        assert config["model"]["provider"] == "openrouter"
        assert config["compression"] == {"enabled": False}
        assert config["custom_section"] == {"keep": True}
        assert "terminal" not in config

    def test_save_custom_provider_preserves_raw_user_config(self, config_home):
        """Saving a named custom provider must not rewrite unrelated defaults."""
        (config_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "custom_section": {"keep": True},
                    "compression": {"enabled": False},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        from hermes_cli.main import _save_custom_provider

        _save_custom_provider("https://example.com/v1", model="my-model", context_length=1234)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        assert config["custom_section"] == {"keep": True}
        assert config["compression"] == {"enabled": False}
        assert config["custom_providers"][0]["base_url"] == "https://example.com/v1"
        assert config["custom_providers"][0]["model"] == "my-model"
        assert "terminal" not in config

    def test_model_flow_nous_preserves_raw_env_placeholders(self, config_home, monkeypatch):
        """Logged-in Nous model selection must not expand unrelated env placeholders."""
        monkeypatch.setenv("GLM_API_KEY", "secret-key-123")
        (config_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {"default": "old-model"},
                    "mcp_servers": {
                        "zread": {
                            "url": "https://example.com",
                            "headers": {"Authorization": "Bearer ${GLM_API_KEY}"},
                        }
                    },
                    "custom_section": {"keep": True},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_nous

        config = load_config()

        with patch("hermes_cli.auth.get_provider_auth_state", return_value={"access_token": "tok"}), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="nous-model",
        ), patch(
            "hermes_cli.auth._save_model_choice",
            lambda *_a, **_k: None,
        ), patch(
            "hermes_cli.auth._update_config_for_provider",
            lambda *_a, **_k: None,
        ), patch(
            "hermes_cli.auth.resolve_nous_runtime_credentials",
            return_value={"base_url": "https://inference.example.com/v1"},
        ), patch(
            "hermes_cli.nous_subscription.apply_nous_provider_defaults",
            return_value=set(),
        ), patch(
            "hermes_cli.nous_subscription.get_nous_subscription_explainer_lines",
            return_value=[],
        ), patch(
            "hermes_cli.config.get_env_value",
            return_value="",
        ), patch(
            "hermes_cli.config.save_env_value",
            lambda *_a, **_k: None,
        ):
            _model_flow_nous(
                config,
                current_model="old-model",
                args=SimpleNamespace(
                    portal_url=None,
                    inference_url=None,
                    client_id=None,
                    scope=None,
                    no_browser=False,
                    timeout=15.0,
                    ca_bundle=None,
                    insecure=False,
                ),
            )

        saved = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        assert saved["model"]["default"] == "nous-model"
        assert saved["model"]["provider"] == "nous"
        assert saved["custom_section"] == {"keep": True}
        assert (
            saved["mcp_servers"]["zread"]["headers"]["Authorization"]
            == "Bearer ${GLM_API_KEY}"
        )
        assert "browser" not in saved

    def test_memory_off_preserves_raw_env_placeholders(self, config_home, monkeypatch):
        """`hermes memory off` must not expand unrelated env placeholders."""
        monkeypatch.setenv("GLM_API_KEY", "secret-key-123")
        (config_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "memory": {"provider": "honcho"},
                    "mcp_servers": {
                        "zread": {
                            "url": "https://example.com",
                            "headers": {"Authorization": "Bearer ${GLM_API_KEY}"},
                        }
                    },
                    "custom_section": {"keep": True},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        from hermes_cli import main as hermes_main

        original_argv = sys.argv[:]
        try:
            sys.argv = ["hermes", "memory", "off"]
            hermes_main.main()
        finally:
            sys.argv = original_argv

        saved = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        if "memory" in saved:
            assert saved["memory"].get("provider", "") == ""
        assert saved["custom_section"] == {"keep": True}
        assert (
            saved["mcp_servers"]["zread"]["headers"]["Authorization"]
            == "Bearer ${GLM_API_KEY}"
        )
        assert "browser" not in saved


class TestProviderPersistsAfterModelSave:
    def test_api_key_provider_saved_when_model_was_string(self, config_home, monkeypatch):
        """_model_flow_api_key_provider must persist the provider even when
        config.model started as a plain string."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("kimi-coding")
        if not pconfig:
            pytest.skip("kimi-coding not in PROVIDER_REGISTRY")

        # Simulate: user has a Kimi API key, model was a string
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        # Mock the model selection prompt to return "kimi-k2.5"
        # Also mock input() for the base URL prompt and builtins.input
        with patch("hermes_cli.auth._prompt_model_selection", return_value="kimi-k2.5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "kimi-coding", "old-model")

        import yaml
        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict), f"model should be dict, got {type(model)}"
        assert model.get("provider") == "kimi-coding", (
            f"provider should be 'kimi-coding', got {model.get('provider')}"
        )
        assert model.get("default") == "kimi-k2.5"

    def test_copilot_provider_saved_when_selected(self, config_home):
        """_model_flow_copilot should persist provider/base_url/model together."""
        from hermes_cli.main import _model_flow_copilot
        from hermes_cli.config import load_config

        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "copilot",
                "api_key": "gh-cli-token",
                "base_url": "https://api.githubcopilot.com",
                "source": "gh auth token",
            },
        ), patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=[
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
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="gpt-5.4",
        ), patch(
            "hermes_cli.main._prompt_reasoning_effort_selection",
            return_value="high",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ):
            _model_flow_copilot(load_config(), "old-model")

        import yaml

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict), f"model should be dict, got {type(model)}"
        assert model.get("provider") == "copilot"
        assert model.get("base_url") == "https://api.githubcopilot.com"
        assert model.get("default") == "gpt-5.4"
        assert model.get("api_mode") == "codex_responses"
        assert config["agent"]["reasoning_effort"] == "high"

    def test_copilot_acp_provider_saved_when_selected(self, config_home):
        """_model_flow_copilot_acp should persist provider/base_url/model together."""
        from hermes_cli.main import _model_flow_copilot_acp
        from hermes_cli.config import load_config

        with patch(
            "hermes_cli.auth.get_external_process_provider_status",
            return_value={
                "resolved_command": "/usr/local/bin/copilot",
                "command": "copilot",
                "base_url": "acp://copilot",
            },
        ), patch(
            "hermes_cli.auth.resolve_external_process_provider_credentials",
            return_value={
                "provider": "copilot-acp",
                "api_key": "copilot-acp",
                "base_url": "acp://copilot",
                "command": "/usr/local/bin/copilot",
                "args": ["--acp", "--stdio"],
                "source": "process",
            },
        ), patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "copilot",
                "api_key": "gh-cli-token",
                "base_url": "https://api.githubcopilot.com",
                "source": "gh auth token",
            },
        ), patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=[
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
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="gpt-5.4",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ):
            _model_flow_copilot_acp(load_config(), "old-model")

        import yaml

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict), f"model should be dict, got {type(model)}"
        assert model.get("provider") == "copilot-acp"
        assert model.get("base_url") == "acp://copilot"
        assert model.get("default") == "gpt-5.4"
        assert model.get("api_mode") == "chat_completions"

    def test_opencode_go_models_are_selectable_and_persist_normalized(self, config_home, monkeypatch):
        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")

        with patch("hermes_cli.models.fetch_api_models", return_value=["opencode-go/kimi-k2.5", "opencode-go/minimax-m2.7"]), \
             patch("hermes_cli.auth._prompt_model_selection", return_value="kimi-k2.5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "opencode-go", "opencode-go/kimi-k2.5")

        import yaml
        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model.get("provider") == "opencode-go"
        assert model.get("default") == "kimi-k2.5"
        assert model.get("api_mode") == "chat_completions"

    def test_opencode_go_same_provider_switch_recomputes_api_mode(self, config_home, monkeypatch):
        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")
        (config_home / "config.yaml").write_text(
            "model:\n"
            "  default: kimi-k2.5\n"
            "  provider: opencode-go\n"
            "  base_url: https://opencode.ai/zen/go/v1\n"
            "  api_mode: chat_completions\n"
        )

        with patch("hermes_cli.models.fetch_api_models", return_value=["opencode-go/kimi-k2.5", "opencode-go/minimax-m2.5"]), \
             patch("hermes_cli.auth._prompt_model_selection", return_value="minimax-m2.5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "opencode-go", "kimi-k2.5")

        import yaml
        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model.get("provider") == "opencode-go"
        assert model.get("default") == "minimax-m2.5"
        assert model.get("api_mode") == "anthropic_messages"
