"""Tests for Hermes memory backend factory resolution."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from memory_backends.base import MemoryBackendLoadError, validate_memory_backend_session
from memory_backends.factory import load_memory_backend


class TestMemoryBackendFactory:
    def test_uses_builtin_honcho_when_no_external_factory_is_configured(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"apiKey": "honcho-key", "enabled": True}))
        manager = MagicMock()

        with (
            patch("memory_backends.honcho.get_honcho_client", return_value=MagicMock()),
            patch("memory_backends.honcho.HonchoSessionManager", return_value=manager),
        ):
            bundle = load_memory_backend(config_path=config_path)

        assert bundle.manager is manager
        assert bundle.config.api_key == "honcho-key"
        assert bundle.manifest.backend_id == "honcho"

    def test_uses_external_factory_when_configured(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend"
                            }
                        }
                    }
                }
            )
        )

        bundle = load_memory_backend(config_path=config_path)

        assert bundle.manager is not None
        assert bundle.manifest.backend_id == "fake-backend"
        assert bundle.config.resolve_session_name(session_id="session-1") == "session-1"

    def test_preserves_requested_host_when_falling_back_to_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)

        bundle = load_memory_backend(
            host="telegram",
            config_path=tmp_path / "nonexistent.json",
        )

        assert bundle.config.host == "telegram"
        assert bundle.config.workspace_id == "telegram"
        assert bundle.config.ai_peer == "telegram"
        assert bundle.manifest.backend_id == "honcho"

    def test_rejects_external_backend_with_invalid_protocol_version(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend_invalid_manifest"
                            }
                        }
                    }
                }
            )
        )

        with pytest.raises(MemoryBackendLoadError, match="protocol_version"):
            load_memory_backend(config_path=config_path)

    def test_rejects_external_backend_missing_required_manager_methods(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend_missing_method"
                            }
                        }
                    }
                }
            )
        )

        with pytest.raises(MemoryBackendLoadError, match="manager is missing required methods"):
            load_memory_backend(config_path=config_path)

    def test_rejects_external_backend_with_unknown_capabilities(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend_unknown_capability"
                            }
                        }
                    }
                }
            )
        )

        with pytest.raises(MemoryBackendLoadError, match="unknown capabilities"):
            load_memory_backend(config_path=config_path)

    def test_rejects_external_backend_ai_identity_capability_without_methods(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend_ai_identity_missing_methods"
                            }
                        }
                    }
                }
            )
        )

        with pytest.raises(MemoryBackendLoadError, match="declares capability 'ai_identity'"):
            load_memory_backend(config_path=config_path)

    def test_rejects_external_backend_with_invalid_import_path(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "does.not.exist:create_backend"
                            }
                        }
                    }
                }
            )
        )

        with pytest.raises(MemoryBackendLoadError, match="failed to import memory backend module"):
            load_memory_backend(config_path=config_path)

    def test_rejects_enabled_external_backend_without_active_manager(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend_enabled_without_manager"
                            }
                        }
                    }
                }
            )
        )

        with pytest.raises(MemoryBackendLoadError, match="did not produce an active manager"):
            load_memory_backend(config_path=config_path)

    def test_rejects_disabled_external_backend_that_returns_a_manager(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend_disabled_with_manager"
                            }
                        }
                    }
                }
            )
        )

        with pytest.raises(
            MemoryBackendLoadError,
            match="disabled but returned an active manager",
        ):
            load_memory_backend(config_path=config_path)

    def test_rejects_invalid_backend_session_shape(self):
        with pytest.raises(MemoryBackendLoadError, match="missing required attribute: messages"):
            validate_memory_backend_session({"session_key": "abc"})

    def test_rejects_backend_session_without_add_message(self):
        class SessionWithoutAddMessage:
            messages = []

        with pytest.raises(MemoryBackendLoadError, match="missing required method: add_message"):
            validate_memory_backend_session(SessionWithoutAddMessage())

    def test_disabled_external_backend_does_not_invoke_external_loader(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "enabled": False,
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend"
                            }
                        }
                    },
                }
            )
        )

        with patch("memory_backends.factory.load_external_backend") as external_loader:
            bundle = load_memory_backend(config_path=config_path)

        external_loader.assert_not_called()
        assert bundle.manager is None
        assert bundle.config.enabled is False
        assert bundle.manifest.backend_id == "external"

    def test_rejects_context_recall_mode_without_prefetch_capability(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "experimental": {
                                "memory_backend_factory": "tests.fakes.fake_memory_backend:create_backend_context_mode_without_prefetch"
                            }
                        }
                    }
                }
            )
        )

        with pytest.raises(
            MemoryBackendLoadError,
            match="recall_mode 'context' requires 'prefetch' capability",
        ):
            load_memory_backend(config_path=config_path)
