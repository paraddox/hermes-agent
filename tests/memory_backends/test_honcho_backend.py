"""Direct tests for the built-in Honcho backend wrapper."""

from __future__ import annotations

from typing import get_args, get_type_hints
from unittest.mock import MagicMock, patch

from memory_backends.base import MemoryBackendManager
from honcho_integration.client import HonchoClientConfig
from memory_backends.honcho import load_honcho_backend


def test_load_honcho_backend_builds_manager_when_config_is_active():
    config = HonchoClientConfig(
        api_key="honcho-key",
        enabled=True,
        context_tokens=512,
    )
    manager = MagicMock()

    with (
        patch("memory_backends.honcho.get_honcho_client", return_value=MagicMock()),
        patch("memory_backends.honcho.HonchoSessionManager", return_value=manager),
    ):
        bundle = load_honcho_backend(config=config)

    assert bundle.manager is manager
    assert bundle.config is config
    assert bundle.manifest.backend_id == "honcho"
    assert "profile" in bundle.manifest.capabilities
    assert "search" in bundle.manifest.capabilities
    assert "ai_identity" in bundle.manifest.capabilities


def test_load_honcho_backend_returns_inactive_manager_without_api_key():
    config = HonchoClientConfig(
        api_key=None,
        enabled=True,
    )

    bundle = load_honcho_backend(config=config)

    assert bundle.manager is None
    assert bundle.config is config
    assert bundle.manifest.backend_id == "honcho"


def test_memory_backend_protocol_accepts_list_or_string_peer_cards():
    hints = get_type_hints(MemoryBackendManager.get_peer_card)
    return_args = set(get_args(hints["return"]))

    assert str in return_args
    assert list[str] in return_args
