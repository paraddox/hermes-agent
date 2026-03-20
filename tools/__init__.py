#!/usr/bin/env python3
"""Convenience exports for the tools package.

This package is used in two different ways:

- `model_tools.py` imports specific submodules to register tools.
- Some callers use convenience imports like `from tools import terminal_tool`.

The second path should not make the whole package unimportable when an
unrelated optional dependency is missing. Keep package init fail-open and let
callers import the specific submodule they need.
"""

from __future__ import annotations

import importlib
import logging


logger = logging.getLogger(__name__)

__all__: list[str] = []


def _optional_export(module_name: str, *names: str) -> None:
    """Expose convenience symbols without hard-failing package import."""

    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except Exception as exc:
        logger.debug(
            "Skipping tools.%s convenience exports during package init: %s",
            module_name,
            exc,
        )
        return

    for name in names:
        globals()[name] = getattr(module, name)
    __all__.extend(names)


_optional_export(
    "web_tools",
    "web_search_tool",
    "web_extract_tool",
    "web_crawl_tool",
    "check_firecrawl_api_key",
)

_optional_export(
    "terminal_tool",
    "terminal_tool",
    "check_terminal_requirements",
    "cleanup_vm",
    "cleanup_all_environments",
    "get_active_environments_info",
    "register_task_env_overrides",
    "clear_task_env_overrides",
    "TERMINAL_TOOL_DESCRIPTION",
)

_optional_export(
    "vision_tools",
    "vision_analyze_tool",
    "check_vision_requirements",
)

_optional_export(
    "mixture_of_agents_tool",
    "mixture_of_agents_tool",
    "check_moa_requirements",
)

_optional_export(
    "image_generation_tool",
    "image_generate_tool",
    "check_image_generation_requirements",
)

_optional_export(
    "skills_tool",
    "skills_list",
    "skill_view",
    "check_skills_requirements",
    "SKILLS_TOOL_DESCRIPTION",
)

_optional_export(
    "skill_manager_tool",
    "skill_manage",
    "check_skill_manage_requirements",
    "SKILL_MANAGE_SCHEMA",
)

_optional_export(
    "browser_tool",
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_back",
    "browser_press",
    "browser_close",
    "browser_get_images",
    "browser_vision",
    "cleanup_browser",
    "cleanup_all_browsers",
    "get_active_browser_sessions",
    "check_browser_requirements",
    "BROWSER_TOOL_SCHEMAS",
)

_optional_export(
    "cronjob_tools",
    "cronjob",
    "schedule_cronjob",
    "list_cronjobs",
    "remove_cronjob",
    "check_cronjob_requirements",
    "get_cronjob_tool_definitions",
    "CRONJOB_SCHEMA",
)

_optional_export(
    "rl_training_tool",
    "rl_list_environments",
    "rl_select_environment",
    "rl_get_current_config",
    "rl_edit_config",
    "rl_start_training",
    "rl_check_status",
    "rl_stop_training",
    "rl_get_results",
    "rl_list_runs",
    "rl_test_inference",
    "check_rl_api_keys",
    "get_missing_keys",
)

_optional_export(
    "file_tools",
    "read_file_tool",
    "write_file_tool",
    "patch_tool",
    "search_tool",
    "get_file_tools",
    "clear_file_ops_cache",
)

_optional_export(
    "tts_tool",
    "text_to_speech_tool",
    "check_tts_requirements",
)

_optional_export(
    "todo_tool",
    "todo_tool",
    "check_todo_requirements",
    "TODO_SCHEMA",
    "TodoStore",
)

_optional_export(
    "clarify_tool",
    "clarify_tool",
    "check_clarify_requirements",
    "CLARIFY_SCHEMA",
)

_optional_export(
    "code_execution_tool",
    "execute_code",
    "check_sandbox_requirements",
    "EXECUTE_CODE_SCHEMA",
)

_optional_export(
    "delegate_tool",
    "delegate_task",
    "check_delegate_requirements",
    "DELEGATE_TASK_SCHEMA",
)


def check_file_requirements():
    """File tools only require the terminal backend to be available."""

    from .terminal_tool import check_terminal_requirements

    return check_terminal_requirements()


__all__.append("check_file_requirements")
