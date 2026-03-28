import copy
import textwrap
from pathlib import Path

import hermes_cli.mcp_presets as mcp_presets
import pytest


def _set_test_preset_data(monkeypatch, tmp_path: Path, yaml_text: str) -> Path:
    preset_path = tmp_path / "mcp_provider_presets.yaml"
    preset_path.write_text(textwrap.dedent(yaml_text).strip() + "\n", encoding="utf-8")
    monkeypatch.setattr(mcp_presets, "_PRESET_DATA_PATH", preset_path)
    return preset_path


def test_shipped_zai_bundle_resolves_from_real_yaml():
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")

    assert bundle["prompt"] == {
        "question": "Enable z.ai MCP servers during setup?",
        "default": True,
        "checklist_title": "Select z.ai MCP servers to enable:",
    }
    assert set(bundle["servers"]) == {
        "web-search-prime",
        "web-reader",
        "zread",
        "zai-vision",
    }
    assert (
        bundle["servers"]["web-search-prime"]["config"]["headers"]["Authorization"]
        == "Bearer ${GLM_API_KEY}"
    )
    assert bundle["servers"]["zai-vision"]["config"]["command"] == "npx"
    assert bundle["servers"]["zai-vision"]["config"]["env"] == {
        "Z_AI_API_KEY": "${GLM_API_KEY}",
        "Z_AI_MODE": "ZAI",
    }


def test_shipped_preset_catalog_validates_as_a_whole():
    presets = mcp_presets._load_presets()

    assert isinstance(presets, list)
    assert presets


def test_provider_only_resolution_uses_matching_provider_entry(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              alpha-core:
                default_checked: true
                config:
                  url: "https://alpha.example/core"
        """,
    )

    bundle = mcp_presets.resolve_provider_mcp_bundle("alpha", model="alpha-fast")

    assert bundle["prompt"] == {
        "question": "Enable Alpha MCP?",
        "default": False,
        "checklist_title": "Alpha servers",
    }
    assert set(bundle["servers"]) == {"alpha-core"}
    assert bundle["servers"]["alpha-core"]["config"]["url"] == "https://alpha.example/core"


def test_provider_model_resolution_overrides_provider_prompt_and_server(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              shared:
                default_checked: true
                config:
                  url: "https://alpha.example/shared"
              alpha-core:
                config:
                  url: "https://alpha.example/core"
          - match:
              provider: alpha
              model: alpha-fast
            prompt:
              question: "Enable Alpha Fast MCP?"
              default: true
            servers:
              shared:
                default_checked_when:
                  command_available: "uvx"
                note_when_unchecked: "`shared` was not preselected because `uvx` is unavailable."
                config:
                  command: "uvx"
                  args: ["alpha-fast-shared"]
              alpha-fast-only:
                config:
                  url: "https://alpha.example/fast"
        """,
    )

    bundle = mcp_presets.resolve_provider_mcp_bundle("alpha", model="alpha-fast")

    assert bundle["prompt"] == {
        "question": "Enable Alpha Fast MCP?",
        "default": True,
        "checklist_title": "Alpha servers",
    }
    assert set(bundle["servers"]) == {"shared", "alpha-core", "alpha-fast-only"}
    assert bundle["servers"]["shared"]["config"] == {
        "command": "uvx",
        "args": ["alpha-fast-shared"],
    }
    assert bundle["servers"]["alpha-core"]["config"]["url"] == "https://alpha.example/core"
    assert bundle["servers"]["alpha-fast-only"]["config"]["url"] == "https://alpha.example/fast"


def test_provider_model_resolution_deep_merges_server_entry(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              shared:
                default_checked_when:
                  command_available: "npx"
                note_when_unchecked: "provider note"
                config:
                  command: "npx"
                  args: ["provider-shared"]
                  env:
                    ALPHA_MODE: "provider"
                    ALPHA_REGION: "global"
          - match:
              provider: alpha
              model: alpha-fast
            prompt:
              question: "Enable Alpha Fast MCP?"
              default: true
            servers:
              shared:
                config:
                  env:
                    ALPHA_MODE: "fast"
        """,
    )

    bundle = mcp_presets.resolve_provider_mcp_bundle("alpha", model="alpha-fast")

    assert bundle["servers"]["shared"] == {
        "default_checked_when": {"command_available": "npx"},
        "note_when_unchecked": "provider note",
        "config": {
            "command": "npx",
            "args": ["provider-shared"],
            "env": {
                "ALPHA_MODE": "fast",
                "ALPHA_REGION": "global",
            },
        },
    }


def test_provider_model_resolution_requires_exact_model_match(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              alpha-core:
                config:
                  url: "https://alpha.example/core"
          - match:
              provider: alpha
              model: alpha-fast
            prompt:
              question: "Enable Alpha Fast MCP?"
              default: true
            servers:
              alpha-fast-only:
                config:
                  url: "https://alpha.example/fast"
        """,
    )

    bundle = mcp_presets.resolve_provider_mcp_bundle("alpha", model="alpha-slow")

    assert bundle["prompt"] == {
        "question": "Enable Alpha MCP?",
        "default": False,
        "checklist_title": "Alpha servers",
    }
    assert set(bundle["servers"]) == {"alpha-core"}


def test_duplicate_match_entries_raise_validation_error(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              alpha-core:
                config:
                  url: "https://alpha.example/core"
          - match:
              provider: alpha
            prompt:
              question: "Enable duplicate Alpha MCP?"
              default: true
              checklist_title: "Alpha duplicate servers"
            servers:
              alpha-extra:
                config:
                  url: "https://alpha.example/extra"
        """,
    )

    with pytest.raises(ValueError, match="Duplicate MCP preset match"):
        mcp_presets.resolve_provider_mcp_bundle("alpha")


def test_runtime_name_collision_in_resolved_bundle_raises_validation_error(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              alpha-core:
                config:
                  url: "https://alpha.example/core"
          - match:
              provider: alpha
              model: alpha-fast
            prompt:
              question: "Enable Alpha Fast MCP?"
              default: true
            servers:
              alpha_core:
                config:
                  url: "https://alpha.example/core-fast"
        """,
    )

    with pytest.raises(ValueError, match="same runtime name"):
        mcp_presets.resolve_provider_mcp_bundle("alpha", model="alpha-fast")


def test_invalid_yaml_raises_validation_error(monkeypatch, tmp_path):
    _set_test_preset_data(monkeypatch, tmp_path, "presets: [")

    with pytest.raises(ValueError, match="Invalid MCP preset YAML"):
        mcp_presets.resolve_provider_mcp_bundle("alpha")


def test_invalid_server_entry_raises_validation_error(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              alpha-core: "https://alpha.example/core"
        """,
    )

    with pytest.raises(ValueError, match="must be a mapping"):
        mcp_presets.resolve_provider_mcp_bundle("alpha")


def test_server_config_with_both_transports_raises_validation_error(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              alpha-core:
                config:
                  url: "https://alpha.example/core"
                  command: "uvx"
        """,
    )

    with pytest.raises(ValueError, match="cannot define both 'url' and 'command'"):
        mcp_presets.resolve_provider_mcp_bundle("alpha")


def test_unsupported_default_checked_when_key_raises_validation_error(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              alpha-core:
                default_checked_when:
                  env_var_present: "ALPHA_KEY"
                config:
                  url: "https://alpha.example/core"
        """,
    )

    with pytest.raises(ValueError, match="Unsupported default_checked_when keys"):
        mcp_presets.resolve_provider_mcp_bundle("alpha")


def test_unrelated_invalid_preset_breaks_catalog_validation(monkeypatch, tmp_path):
    _set_test_preset_data(
        monkeypatch,
        tmp_path,
        """
        presets:
          - match:
              provider: alpha
            prompt:
              question: "Enable Alpha MCP?"
              default: false
              checklist_title: "Alpha servers"
            servers:
              alpha-core: "https://alpha.example/core"
          - match:
              provider: zai
            prompt:
              question: "Enable z.ai MCP servers during setup?"
              default: true
              checklist_title: "Select z.ai MCP servers to enable:"
            servers:
              web-reader:
                default_checked: true
                config:
                  url: "https://api.z.ai/api/mcp/web_reader/mcp"
        """,
    )

    with pytest.raises(ValueError, match="must be a mapping"):
        mcp_presets.resolve_provider_mcp_bundle("zai", model="glm-5")


def test_get_bundle_default_checked_servers_marks_command_gated_server_when_available(monkeypatch):
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")
    monkeypatch.setattr(
        mcp_presets.shutil,
        "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd == "npx" else None,
    )

    checked, notes = mcp_presets.get_bundle_default_checked_servers(bundle)

    assert checked == ["web-search-prime", "web-reader", "zread", "zai-vision"]
    assert notes == []


def test_get_bundle_default_checked_servers_emits_note_when_command_is_missing(monkeypatch):
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")
    monkeypatch.setattr(mcp_presets.shutil, "which", lambda _cmd: None)

    checked, notes = mcp_presets.get_bundle_default_checked_servers(bundle)

    assert checked == ["web-search-prime", "web-reader", "zread"]
    assert notes == ["`zai-vision` was not preselected because `npx` is unavailable."]


def test_merge_selected_provider_mcp_servers_preserves_existing_entries():
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")
    existing_entry = {
        "url": "https://example.invalid/mcp",
        "enabled": False,
        "tools": {"exclude": ["foo"]},
    }
    config = {"mcp_servers": {"web-search-prime": copy.deepcopy(existing_entry)}}

    summary = mcp_presets.merge_selected_provider_mcp_servers(
        config,
        bundle,
        ["web-search-prime", "zread"],
    )

    assert config["mcp_servers"]["web-search-prime"] == existing_entry
    assert config["mcp_servers"]["zread"]["enabled"] is True
    assert summary == {
        "added": ["zread"],
        "skipped_existing": ["web-search-prime"],
        "skipped_runtime_collisions": [],
        "skipped_equivalent_existing": [],
    }


def test_merge_selected_provider_mcp_servers_tolerates_malformed_existing_mcp_servers():
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")
    config = {"mcp_servers": ["not", "a", "mapping"]}

    summary = mcp_presets.merge_selected_provider_mcp_servers(config, bundle, ["zread"])

    assert config["mcp_servers"] == {
        "zread": {
            "url": "https://api.z.ai/api/mcp/zread/mcp",
            "headers": {"Authorization": "Bearer ${GLM_API_KEY}"},
            "enabled": True,
        }
    }
    assert summary == {
        "added": ["zread"],
        "skipped_existing": [],
        "skipped_runtime_collisions": [],
        "skipped_equivalent_existing": [],
    }


def test_merge_selected_provider_mcp_servers_skips_existing_runtime_name_collision():
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")
    config = {
        "mcp_servers": {
            "zai_vision": {
                "command": "npx",
                "args": ["-y", "old-server"],
            }
        }
    }

    summary = mcp_presets.merge_selected_provider_mcp_servers(
        config,
        bundle,
        ["zai-vision"],
    )

    assert config["mcp_servers"] == {
        "zai_vision": {
            "command": "npx",
            "args": ["-y", "old-server"],
        }
    }
    assert summary == {
        "added": [],
        "skipped_existing": [],
        "skipped_runtime_collisions": [
            {
                "requested": "zai-vision",
                "existing": "zai_vision",
            }
        ],
        "skipped_equivalent_existing": [],
    }


def test_merge_selected_provider_mcp_servers_skips_existing_equivalent_server():
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")
    config = {
        "mcp_servers": {
            "custom-zread": {
                "url": "https://api.z.ai/api/mcp/zread/mcp",
                "headers": {"Authorization": "Bearer ${GLM_API_KEY}"},
                "enabled": False,
                "tools": {"exclude": ["foo"]},
                "timeout": 120,
            }
        }
    }

    summary = mcp_presets.merge_selected_provider_mcp_servers(
        config,
        bundle,
        ["zread"],
    )

    assert config["mcp_servers"] == {
        "custom-zread": {
            "url": "https://api.z.ai/api/mcp/zread/mcp",
            "headers": {"Authorization": "Bearer ${GLM_API_KEY}"},
            "enabled": False,
            "tools": {"exclude": ["foo"]},
            "timeout": 120,
        }
    }
    assert summary == {
        "added": [],
        "skipped_existing": [],
        "skipped_runtime_collisions": [],
        "skipped_equivalent_existing": [
            {
                "requested": "zread",
                "existing": "custom-zread",
            }
        ],
    }


def test_merge_selected_provider_mcp_servers_matches_raw_existing_config_for_equivalence():
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")
    config = {
        "mcp_servers": {
            "custom-zread": {
                "url": "https://api.z.ai/api/mcp/zread/mcp",
                "headers": {"Authorization": "Bearer zai-key"},
                "enabled": False,
            }
        }
    }
    raw_existing_mcp_servers = {
        "custom-zread": {
            "url": "https://api.z.ai/api/mcp/zread/mcp",
            "headers": {"Authorization": "Bearer ${GLM_API_KEY}"},
            "enabled": False,
        }
    }

    summary = mcp_presets.merge_selected_provider_mcp_servers(
        config,
        bundle,
        ["zread"],
        existing_mcp_servers_raw=raw_existing_mcp_servers,
    )

    assert config["mcp_servers"] == {
        "custom-zread": {
            "url": "https://api.z.ai/api/mcp/zread/mcp",
            "headers": {"Authorization": "Bearer zai-key"},
            "enabled": False,
        }
    }
    assert summary == {
        "added": [],
        "skipped_existing": [],
        "skipped_runtime_collisions": [],
        "skipped_equivalent_existing": [
            {
                "requested": "zread",
                "existing": "custom-zread",
            }
        ],
    }


def test_merge_selected_provider_mcp_servers_matches_expanded_in_memory_config_for_equivalence(
    monkeypatch,
):
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")
    config = {
        "mcp_servers": {
            "custom-zread": {
                "url": "https://api.z.ai/api/mcp/zread/mcp",
                "headers": {"Authorization": "Bearer zai-key"},
                "enabled": False,
            }
        }
    }
    monkeypatch.setenv("GLM_API_KEY", "zai-key")

    summary = mcp_presets.merge_selected_provider_mcp_servers(
        config,
        bundle,
        ["zread"],
    )

    assert config["mcp_servers"] == {
        "custom-zread": {
            "url": "https://api.z.ai/api/mcp/zread/mcp",
            "headers": {"Authorization": "Bearer zai-key"},
            "enabled": False,
        }
    }
    assert summary == {
        "added": [],
        "skipped_existing": [],
        "skipped_runtime_collisions": [],
        "skipped_equivalent_existing": [
            {
                "requested": "zread",
                "existing": "custom-zread",
            }
        ],
    }


def test_merge_selected_provider_mcp_servers_is_noop_for_empty_and_unknown_selections():
    bundle = mcp_presets.resolve_provider_mcp_bundle("zai")

    empty_config = {}
    empty_before = copy.deepcopy(empty_config)
    empty_summary = mcp_presets.merge_selected_provider_mcp_servers(empty_config, bundle, [])

    assert empty_config == empty_before
    assert empty_summary == {
        "added": [],
        "skipped_existing": [],
        "skipped_runtime_collisions": [],
        "skipped_equivalent_existing": [],
    }

    unknown_config = {}
    unknown_before = copy.deepcopy(unknown_config)
    unknown_summary = mcp_presets.merge_selected_provider_mcp_servers(
        unknown_config,
        bundle,
        ["does-not-exist"],
    )

    assert unknown_config == unknown_before
    assert unknown_summary == {
        "added": [],
        "skipped_existing": [],
        "skipped_runtime_collisions": [],
        "skipped_equivalent_existing": [],
    }
