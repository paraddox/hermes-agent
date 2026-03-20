"""Tests for Honcho CLI helpers."""

import builtins
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from honcho_integration.cli import (
    _bind_memory_session_context,
    _load_cli_memory_backend,
    _resolve_api_key,
    cmd_identity,
    cmd_migrate,
    cmd_status,
)
from tools import honcho_tools


class TestResolveApiKey:
    def test_prefers_host_scoped_key(self):
        cfg = {
            "apiKey": "root-key",
            "hosts": {
                "hermes": {
                    "apiKey": "host-key",
                }
            },
        }
        assert _resolve_api_key(cfg) == "host-key"

    def test_falls_back_to_root_key(self):
        cfg = {
            "apiKey": "root-key",
            "hosts": {"hermes": {}},
        }
        assert _resolve_api_key(cfg) == "root-key"

    def test_falls_back_to_env_key(self, monkeypatch):
        monkeypatch.setenv("HONCHO_API_KEY", "env-key")
        assert _resolve_api_key({}) == "env-key"
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)


class TestHonchoCliStatus:
    @pytest.mark.parametrize(
        ("config_payload", "config_error"),
        [
            (None, None),
            ("{bad json", json.JSONDecodeError("Expecting property name enclosed in double quotes", "{bad json", 1)),
        ],
    )
    def test_cmd_status_uses_env_fallback_when_config_is_missing_or_unreadable(
        self, monkeypatch, capsys, tmp_path, config_payload, config_error
    ):
        config_path = tmp_path / "config.json"
        if config_payload is not None:
            config_path.write_text(config_payload, encoding="utf-8")

        if config_error is None:
            read_config = lambda: {}
        else:
            def read_config():
                raise config_error

        monkeypatch.setenv("HONCHO_API_KEY", "env-honcho-key")
        monkeypatch.setattr("honcho_integration.cli.GLOBAL_CONFIG_PATH", config_path)
        monkeypatch.setattr("honcho_integration.client.GLOBAL_CONFIG_PATH", config_path)
        monkeypatch.setattr("honcho_integration.cli._read_config", read_config)

        cmd_status(Namespace())

        out = capsys.readouterr().out
        assert "No Honcho config found" not in out
        assert "Config error:" not in out
        assert "Enabled:        True" in out
        assert "API key:        ...ncho-key" in out
        assert "Connection... OK" in out

    def test_cmd_status_reports_external_backend_without_honcho_sdk(
        self, monkeypatch, capsys
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="hermes",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )
        fake_bundle = SimpleNamespace(
            manager=object(),
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"profile", "search"}),
            ),
        )

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "honcho":
                raise ImportError("honcho missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )
        monkeypatch.setattr(builtins, "__import__", fake_import)

        cmd_status(Namespace())

        out = capsys.readouterr().out
        assert "External backend:" in out
        assert "External Test (external-test)" in out
        assert "Capabilities:   profile, search" in out
        assert "Active" in out
        assert "No API key configured" not in out
        assert "honcho-ai is not installed" not in out

    def test_cmd_status_prefers_active_external_backend_config_values(
        self, monkeypatch, capsys
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="honcho-workspace",
            host="hermes",
            ai_peer="hermes",
            peer_name="honcho-user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "honcho-session",
        )
        fake_external_cfg = SimpleNamespace(
            enabled=True,
            workspace_id="external-workspace",
            ai_peer="assistant",
            peer_name="alice",
            recall_mode="tools",
            memory_mode="honcho",
            peer_memory_modes={},
            write_frequency=5,
            resolve_session_name=lambda *args, **kwargs: "external-session",
        )
        fake_bundle = SimpleNamespace(
            manager=object(),
            config=fake_external_cfg,
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"profile", "search"}),
            ),
        )

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )

        cmd_status(Namespace())

        out = capsys.readouterr().out
        assert "API key:        n/a (external backend)" in out
        assert "Workspace:      external-workspace" in out
        assert "AI peer:        assistant" in out
        assert "User peer:      alice" in out
        assert "Session key:    external-session" in out
        assert "Recall mode:    tools" in out
        assert "Memory mode:    honcho" in out
        assert "Write freq:     5" in out
        assert "honcho-workspace" not in out
        assert "honcho-session" not in out

    def test_cmd_status_reports_external_backend_active_elsewhere_when_probe_succeeds(
        self, monkeypatch, capsys
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        fake_probe = {
            "backend_id": "external-test",
            "display_name": "External Test",
            "capabilities": frozenset({"profile", "search", "answer"}),
            "status": "active_elsewhere",
            "detail": "vector store lock is held at /tmp/external-store/LOCK",
            "config": fake_hcfg,
        }

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: (_ for _ in ()).throw(RuntimeError("Can't lock read-write collection")),
        )
        monkeypatch.setattr(
            "honcho_integration.cli._probe_external_backend_status",
            lambda *args, **kwargs: fake_probe,
        )

        cmd_status(Namespace())

        out = capsys.readouterr().out
        assert "External backend: External Test (external-test)" in out
        assert "Capabilities:   answer, profile, search" in out
        assert "Active (in use by another Hermes process)" in out
        assert "vector store lock is held" in out
        assert "External backend failed" not in out


class TestHonchoCliIdentity:
    def test_cmd_identity_reports_external_backend_load_failures_without_honcho_label(
        self, monkeypatch, capsys
    ):
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        cmd_identity(Namespace(file=None, show=True))

        out = capsys.readouterr().out
        assert "External memory backend identity command failed: boom" in out
        assert "Honcho connection failed" not in out

    def test_cmd_identity_show_handles_external_backend_runtime_errors_gracefully(
        self, monkeypatch, capsys
    ):
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        class ExplodingIdentityManager:
            def get_or_create(self, session_key):
                return object()

            def get_peer_card(self, session_key):
                raise RuntimeError("backend exploded")

            def get_ai_representation(self, session_key):
                return {"representation": "assistant persona", "card": ""}

        fake_bundle = SimpleNamespace(
            manager=ExplodingIdentityManager(),
            config=fake_hcfg,
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"profile", "ai_identity"}),
            ),
        )

        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )

        cmd_identity(Namespace(file=None, show=True))

        out = capsys.readouterr().out
        assert "External Test identity command failed: backend exploded" in out
        assert "Traceback" not in out

    def test_cmd_identity_uses_external_backend_without_honcho_api_key(
        self, monkeypatch, capsys
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        class ExternalIdentityManager:
            def get_or_create(self, session_key):
                return object()

            def get_peer_card(self, session_key):
                return ["prefers vim"]

            def get_ai_representation(self, session_key):
                return {"representation": "assistant persona", "card": ""}

        fake_bundle = SimpleNamespace(
            manager=ExternalIdentityManager(),
            config=fake_hcfg,
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"profile", "ai_identity"}),
            ),
        )

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )

        cmd_identity(Namespace(file=None, show=True))

        out = capsys.readouterr().out
        assert "User peer (user)" in out
        assert "prefers vim" in out
        assert "AI peer (assistant)" in out
        assert "assistant persona" in out
        assert "No API key configured" not in out
        assert "Honcho connection failed" not in out

    def test_cmd_identity_show_degrades_cleanly_when_profile_capability_is_missing(
        self, monkeypatch, capsys
    ):
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        class AiIdentityOnlyManager:
            def get_or_create(self, session_key):
                return object()

            def get_peer_card(self, session_key):
                raise AssertionError("profile path should not be called without capability")

            def get_ai_representation(self, session_key):
                return {"representation": "assistant persona", "card": ""}

        fake_bundle = SimpleNamespace(
            manager=AiIdentityOnlyManager(),
            config=fake_hcfg,
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"ai_identity"}),
            ),
        )

        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )

        cmd_identity(Namespace(file=None, show=True))

        out = capsys.readouterr().out
        assert "User peer (user)" in out
        assert "Profile view is not supported by the active memory backend." in out
        assert "AI peer (assistant)" in out
        assert "assistant persona" in out
        assert "identity command failed" not in out

    def test_cmd_migrate_uses_external_backend_without_honcho_api_key(
        self, monkeypatch, capsys, tmp_path
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        class ExternalMigrateManager:
            def __init__(self):
                self.calls = []

            def get_or_create(self, session_key):
                return object()

            def migrate_memory_files(self, session_key, memory_dir):
                self.calls.append((session_key, memory_dir))
                return True

        manager = ExternalMigrateManager()
        fake_bundle = SimpleNamespace(
            manager=manager,
            config=fake_hcfg,
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"migrate"}),
            ),
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "USER.md").write_text("prefers vim\n", encoding="utf-8")

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )
        monkeypatch.setattr("honcho_integration.cli._prompt", lambda *args, **kwargs: "y")
        monkeypatch.setattr("honcho_integration.cli.os.getcwd", lambda: str(workspace))
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

        cmd_migrate(Namespace())

        out = capsys.readouterr().out
        assert manager.calls == [("session-1", str(workspace))]
        assert "External memory backend is already active." in out
        assert "Uploaded user memory files from:" in out
        assert "Run 'hermes honcho setup' first." not in out

    def test_cmd_migrate_detects_active_external_backend_from_resolved_config(
        self, monkeypatch, capsys, tmp_path
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        resolved_cfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )
        external_cfg = SimpleNamespace(
            enabled=True,
            workspace_id="workspace-1",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        fake_bundle = SimpleNamespace(
            manager=object(),
            config=external_cfg,
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"migrate"}),
            ),
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: resolved_cfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )
        monkeypatch.setattr("honcho_integration.cli.os.getcwd", lambda: str(workspace))
        monkeypatch.setattr("honcho_integration.cli._prompt", lambda *args, **kwargs: "n")

        cmd_migrate(Namespace())

        out = capsys.readouterr().out
        assert "External memory backend is already active." in out
        assert "Honcho API key already configured" not in out

    def test_cmd_migrate_active_external_backend_skips_honcho_setup_next_steps(
        self, monkeypatch, capsys, tmp_path
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )
        fake_bundle = SimpleNamespace(
            manager=object(),
            config=fake_hcfg,
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"migrate", "ai_identity"}),
            ),
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )
        monkeypatch.setattr("honcho_integration.cli.os.getcwd", lambda: str(workspace))

        cmd_migrate(Namespace())

        out = capsys.readouterr().out
        assert "External memory backend is already active." in out
        assert "1. hermes honcho setup" not in out
        assert "1. hermes honcho status" in out


class TestLoadCliMemoryBackend:
    def test_reports_inactive_external_backend_without_honcho_only_config_attrs(
        self, monkeypatch
    ):
        resolved_cfg = SimpleNamespace(
            enabled=True,
            api_key="",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )
        external_cfg = SimpleNamespace(
            enabled=True,
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )
        fake_bundle = SimpleNamespace(
            manager=None,
            config=external_cfg,
            manifest=SimpleNamespace(backend_id="external-test"),
        )

        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: resolved_cfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )

        with pytest.raises(
            RuntimeError,
            match="External memory backend is configured but inactive.",
        ):
            _load_cli_memory_backend(require_manager=True)


class TestBindMemorySessionContext:
    def setup_method(self):
        self.orig_manager = honcho_tools._session_manager
        self.orig_key = honcho_tools._session_key
        self.orig_capabilities = honcho_tools._backend_capabilities

    def teardown_method(self):
        honcho_tools._session_manager = self.orig_manager
        honcho_tools._session_key = self.orig_key
        honcho_tools._backend_capabilities = self.orig_capabilities

    def test_preserves_backend_capabilities_when_rebinding_session_key(self):
        manager = object()
        honcho_tools.set_session_context(manager, "old-session", capabilities={"profile"})

        agent = SimpleNamespace(
            _honcho=manager,
            _memory_backend_capabilities=lambda: {"profile"},
        )

        _bind_memory_session_context(agent, "new-session")

        assert honcho_tools._session_manager is manager
        assert honcho_tools._session_key == "new-session"
        assert honcho_tools._backend_capabilities == {"profile"}


class TestHonchoCliMigrateCapabilities:
    def test_cmd_migrate_hides_conversation_tools_when_no_backend_is_active(
        self, monkeypatch, capsys, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: {})
        monkeypatch.setattr(
            "honcho_integration.cli._load_cli_memory_backend",
            lambda require_manager=True: (_ for _ in ()).throw(
                RuntimeError("No API key configured. Run 'hermes honcho setup' first.")
            ),
        )
        monkeypatch.setattr("honcho_integration.cli.os.getcwd", lambda: str(workspace))
        monkeypatch.setattr("honcho_integration.cli._prompt", lambda *args, **kwargs: "n")

        cmd_migrate(Namespace())

        out = capsys.readouterr().out
        assert "No conversation memory tools are exposed by the active backend." in out
        assert "honcho_context" not in out
        assert "honcho_search" not in out
        assert "honcho_profile" not in out
        assert "honcho_conclude" not in out

    def test_cmd_migrate_external_backend_load_failure_skips_honcho_onboarding(
        self, monkeypatch, capsys, tmp_path
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "honcho_integration.cli._load_cli_memory_backend",
            lambda require_manager=True: (_ for _ in ()).throw(
                RuntimeError("failed to import memory backend module")
            ),
        )
        monkeypatch.setattr("honcho_integration.cli.os.getcwd", lambda: str(workspace))
        monkeypatch.setattr("honcho_integration.cli._prompt", lambda *args, **kwargs: "n")

        cmd_migrate(Namespace())

        out = capsys.readouterr().out
        assert "failed to import memory backend module" in out
        assert "Run 'hermes honcho setup' now?" not in out
        assert "Honcho is a cloud memory service" not in out
        assert "1. hermes honcho setup" not in out

    def test_cmd_migrate_hides_unsupported_external_tool_and_identity_guidance(
        self, monkeypatch, capsys, tmp_path
    ):
        fake_config = {
            "hosts": {
                "hermes": {
                    "experimental": {
                        "memory_backend_factory": "pkg.module:create_backend",
                    }
                }
            }
        }
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="",
            workspace_id="workspace-1",
            host="hermes",
            ai_peer="assistant",
            peer_name="user",
            recall_mode="hybrid",
            memory_mode="hybrid",
            peer_memory_modes={},
            write_frequency="async",
            memory_backend_factory="pkg.module:create_backend",
            resolve_session_name=lambda *args, **kwargs: "session-1",
        )

        class MigrateOnlyManager:
            def get_or_create(self, session_key):
                return object()

            def migrate_memory_files(self, session_key, memory_dir):
                return True

        fake_bundle = SimpleNamespace(
            manager=MigrateOnlyManager(),
            config=fake_hcfg,
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
                capabilities=frozenset({"migrate"}),
            ),
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "AGENTS.md").write_text("agent identity\n", encoding="utf-8")

        monkeypatch.setattr("honcho_integration.cli._read_config", lambda: fake_config)
        monkeypatch.setattr(
            "honcho_integration.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "memory_backends.factory.load_memory_backend",
            lambda: fake_bundle,
        )
        monkeypatch.setattr("honcho_integration.cli.os.getcwd", lambda: str(workspace))
        monkeypatch.setattr("honcho_integration.cli._prompt", lambda *args, **kwargs: "n")

        cmd_migrate(Namespace())

        out = capsys.readouterr().out
        assert "Active memory backend does not support AI identity seeding." in out
        assert "No conversation memory tools are exposed by the active backend." in out
        assert "honcho_context" not in out
        assert "honcho_search" not in out
        assert "honcho_profile" not in out
        assert "honcho_conclude" not in out
        assert "hermes honcho identity --show" not in out
