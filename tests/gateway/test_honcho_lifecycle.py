"""Tests for gateway-owned Honcho lifecycle helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._honcho_manifests = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.adapters = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    return runner


def _make_event(text="/reset"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="user-1",
            user_name="alice",
        ),
    )


class TestGatewayHonchoLifecycle:
    def test_gateway_reuses_honcho_manager_for_session_key(self):
        runner = _make_runner()
        hcfg = SimpleNamespace(
            enabled=True,
            api_key="honcho-key",
            ai_peer="hermes",
            peer_name="alice",
            context_tokens=123,
            peer_memory_mode=lambda peer: "hybrid",
        )
        manager = MagicMock()
        manifest = SimpleNamespace(backend_id="honcho")

        with (
            patch(
                "gateway.run.load_memory_backend",
                return_value=SimpleNamespace(manager=manager, config=hcfg, manifest=manifest),
            ) as mock_loader,
        ):
            first_mgr, first_cfg, first_manifest = runner._get_or_create_gateway_honcho("session-key")
            second_mgr, second_cfg, second_manifest = runner._get_or_create_gateway_honcho("session-key")

        assert first_mgr is manager
        assert second_mgr is manager
        assert first_cfg is hcfg
        assert second_cfg is hcfg
        assert first_manifest is manifest
        assert second_manifest is manifest
        mock_loader.assert_called_once()

    def test_gateway_skips_honcho_manager_when_disabled(self):
        runner = _make_runner()
        hcfg = SimpleNamespace(
            enabled=False,
            api_key="honcho-key",
            ai_peer="hermes",
            peer_name="alice",
        )
        manifest = SimpleNamespace(backend_id="honcho")

        with (
            patch(
                "gateway.run.load_memory_backend",
                return_value=SimpleNamespace(manager=None, config=hcfg, manifest=manifest),
            ) as mock_loader,
        ):
            manager, cfg, returned_manifest = runner._get_or_create_gateway_honcho("session-key")

        assert manager is None
        assert cfg is hcfg
        assert returned_manifest is manifest
        mock_loader.assert_called_once()

    def test_gateway_caches_manifest_for_session_key(self):
        runner = _make_runner()
        hcfg = SimpleNamespace(
            enabled=True,
            api_key="honcho-key",
            ai_peer="hermes",
            peer_name="alice",
            context_tokens=123,
            peer_memory_mode=lambda peer: "hybrid",
        )
        manager = MagicMock()
        manifest = SimpleNamespace(
            backend_id="external-test",
            display_name="External Test",
            capabilities=frozenset({"profile", "search"}),
        )

        with patch(
            "gateway.run.load_memory_backend",
            return_value=SimpleNamespace(manager=manager, config=hcfg, manifest=manifest),
        ):
            returned = runner._get_or_create_gateway_honcho("session-key")

        assert returned == (manager, hcfg, manifest)
        assert runner._honcho_manifests["session-key"] is manifest

    def test_gateway_shutdown_keeps_shared_manager_alive_until_last_session(self):
        runner = _make_runner()
        shared_manager = MagicMock()
        hcfg = SimpleNamespace(enabled=True, api_key="honcho-key")
        manifest = SimpleNamespace(backend_id="external-test")
        runner._honcho_managers = {
            "session-a": shared_manager,
            "session-b": shared_manager,
        }
        runner._honcho_configs = {
            "session-a": hcfg,
            "session-b": hcfg,
        }
        runner._honcho_manifests = {
            "session-a": manifest,
            "session-b": manifest,
        }

        runner._shutdown_gateway_honcho("session-a")

        shared_manager.shutdown.assert_not_called()
        assert "session-b" in runner._honcho_managers

        runner._shutdown_gateway_honcho("session-b")

        shared_manager.shutdown.assert_called_once_with()

    def test_gateway_flushes_shared_manager_when_a_session_ends(self):
        runner = _make_runner()
        shared_manager = MagicMock()
        hcfg = SimpleNamespace(enabled=True, api_key="honcho-key")
        manifest = SimpleNamespace(backend_id="external-test")
        runner._honcho_managers = {
            "session-a": shared_manager,
            "session-b": shared_manager,
        }
        runner._honcho_configs = {
            "session-a": hcfg,
            "session-b": hcfg,
        }
        runner._honcho_manifests = {
            "session-a": manifest,
            "session-b": manifest,
        }

        runner._shutdown_gateway_honcho("session-a")

        shared_manager.flush_all.assert_called_once_with()
        shared_manager.shutdown.assert_not_called()
        assert "session-b" in runner._honcho_managers

    @pytest.mark.asyncio
    async def test_reset_shuts_down_gateway_honcho_manager(self):
        runner = _make_runner()
        event = _make_event()
        runner._shutdown_gateway_honcho = MagicMock()
        runner._async_flush_memories = AsyncMock()
        runner.session_store = MagicMock()
        runner.session_store._generate_session_key.return_value = "gateway-key"
        runner.session_store._entries = {
            "gateway-key": SimpleNamespace(session_id="old-session"),
        }
        runner.session_store.reset_session.return_value = SimpleNamespace(session_id="new-session")

        result = await runner._handle_reset_command(event)

        runner._shutdown_gateway_honcho.assert_called_once_with("gateway-key")
        runner._async_flush_memories.assert_called_once_with("old-session", "gateway-key")
        assert "Session reset" in result

    def test_flush_memories_reuses_gateway_session_key_and_skips_honcho_sync(self):
        runner = _make_runner()
        runner.session_store = MagicMock()
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        cached_manager = MagicMock()
        cached_config = SimpleNamespace(enabled=True, api_key="honcho-key")
        cached_manifest = SimpleNamespace(backend_id="external-test")
        runner._honcho_managers["gateway-key"] = cached_manager
        runner._honcho_configs["gateway-key"] = cached_config
        runner._honcho_manifests["gateway-key"] = cached_manifest
        tmp_agent = MagicMock()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
            patch("gateway.run._resolve_gateway_model", return_value="model-name"),
            patch("run_agent.AIAgent", return_value=tmp_agent) as mock_agent_cls,
        ):
            runner._flush_memories_for_session("old-session", "gateway-key")

        mock_agent_cls.assert_called_once()
        _, kwargs = mock_agent_cls.call_args
        assert kwargs["session_id"] == "old-session"
        assert kwargs["honcho_session_key"] == "gateway-key"
        assert kwargs["honcho_manager"] is cached_manager
        assert kwargs["honcho_config"] is cached_config
        assert kwargs["honcho_manifest"] is cached_manifest
        tmp_agent.run_conversation.assert_called_once()
        _, run_kwargs = tmp_agent.run_conversation.call_args
        assert run_kwargs["sync_honcho"] is False
