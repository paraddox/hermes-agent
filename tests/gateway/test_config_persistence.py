"""Tests for gateway config mutation commands using the shared config writer."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/personality pirate", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


@pytest.mark.asyncio
async def test_personality_command_uses_shared_config_writer(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "agent:\n"
        "  personalities:\n"
        "    pirate: Arrr\n"
        "mcp_servers:\n"
        "  zread:\n"
        "    headers:\n"
        "      Authorization: Bearer ${GLM_API_KEY}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    runner = _make_runner()
    result = await runner._handle_personality_command(_make_event("/personality pirate"))

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    saved_text = config_path.read_text(encoding="utf-8")
    assert saved["agent"]["system_prompt"] == "Arrr"
    assert "Authorization: Bearer ${GLM_API_KEY}" in saved_text
    assert "# ── Security" in saved_text
    assert "takes effect on next message" in result


@pytest.mark.asyncio
async def test_sethome_command_uses_shared_config_writer(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "mcp_servers:\n"
        "  zread:\n"
        "    headers:\n"
        "      Authorization: Bearer ${GLM_API_KEY}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    runner = _make_runner()
    result = await runner._handle_set_home_command(_make_event("/sethome"))

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    saved_text = config_path.read_text(encoding="utf-8")
    assert saved["TELEGRAM_HOME_CHANNEL"] == "67890"
    assert "Authorization: Bearer ${GLM_API_KEY}" in saved_text
    assert "# ── Security" in saved_text
    assert "Home channel set" in result
