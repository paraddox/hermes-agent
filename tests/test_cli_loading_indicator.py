"""Regression tests for loading feedback on slow slash commands."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


class TestCLILoadingIndicator:
    def _make_cli(self):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj._app = None
        cli_obj._last_invalidate = 0.0
        cli_obj._command_running = False
        cli_obj._command_status = ""
        return cli_obj

    def test_skills_command_sets_busy_state_and_prints_status(self, capsys):
        cli_obj = self._make_cli()
        seen = {}

        def fake_handle(cmd: str):
            seen["cmd"] = cmd
            seen["running"] = cli_obj._command_running
            seen["status"] = cli_obj._command_status
            print("skills done")

        with patch.object(cli_obj, "_handle_skills_command", side_effect=fake_handle), \
             patch.object(cli_obj, "_invalidate") as invalidate_mock:
            assert cli_obj.process_command("/skills search kubernetes")

        output = capsys.readouterr().out
        assert "⏳ Searching skills..." in output
        assert "skills done" in output
        assert seen == {
            "cmd": "/skills search kubernetes",
            "running": True,
            "status": "Searching skills...",
        }
        assert cli_obj._command_running is False
        assert cli_obj._command_status == ""
        assert invalidate_mock.call_count == 2

    def test_reload_mcp_sets_busy_state_and_prints_status(self, capsys):
        cli_obj = self._make_cli()
        seen = {}

        def fake_reload():
            seen["running"] = cli_obj._command_running
            seen["status"] = cli_obj._command_status
            print("reload done")

        with patch.object(cli_obj, "_reload_mcp", side_effect=fake_reload), \
             patch.object(cli_obj, "_invalidate") as invalidate_mock:
            assert cli_obj.process_command("/reload-mcp")

        output = capsys.readouterr().out
        assert "⏳ Reloading MCP servers..." in output
        assert "reload done" in output
        assert seen == {
            "running": True,
            "status": "Reloading MCP servers...",
        }
        assert cli_obj._command_running is False
        assert cli_obj._command_status == ""
        assert invalidate_mock.call_count == 2

    def test_reload_mcp_preserves_context_mode_memory_tool_hiding(self):
        cli_obj = self._make_cli()
        cli_obj.conversation_history = []

        class StubAgent:
            def __init__(self):
                self.enabled_toolsets = None
                self.tools = []
                self.valid_tool_names = set()
                self._honcho = object()
                self._honcho_config = SimpleNamespace(recall_mode="context")
                self._persist_session = MagicMock()

            def _apply_memory_backend_tool_filters(self):
                self.tools = [
                    tool for tool in self.tools
                    if tool["function"]["name"] == "web_search"
                ]
                self.valid_tool_names = {
                    tool["function"]["name"] for tool in self.tools
                }

        cli_obj.agent = StubAgent()

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "honcho_profile",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "honcho_search",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        with (
            patch("tools.mcp_tool.shutdown_mcp_servers"),
            patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
            patch("tools.mcp_tool._servers", {}),
            patch("model_tools.get_tool_definitions", return_value=tool_defs),
        ):
            cli_obj._reload_mcp()

        assert cli_obj.agent.valid_tool_names == {"web_search"}

    def test_session_indicator_uses_external_backend_label_without_honcho_link(self):
        cli_obj = self._make_cli()
        cli_obj.session_id = "session-1"

        fake_bundle = SimpleNamespace(
            manager=object(),
            config=SimpleNamespace(
                enabled=True,
                workspace_id="external-workspace",
                resolve_session_name=lambda **kwargs: kwargs.get("session_id") or "session-1",
            ),
            manifest=SimpleNamespace(
                backend_id="external-test",
                display_name="External Test",
            ),
        )

        with (
            patch("memory_backends.factory.load_memory_backend", return_value=fake_bundle),
            patch("agent.display.write_tty") as write_tty_mock,
        ):
            cli_obj._show_memory_backend_session_indicator()

        write_tty_mock.assert_called_once()
        rendered = write_tty_mock.call_args.args[0]
        assert "External Test session:" in rendered
        assert "Honcho session:" not in rendered
        assert "https://app.honcho.dev" not in rendered
