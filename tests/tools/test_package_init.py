"""Regression tests for optional-dependency-safe tools package imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_agent_import_tolerates_missing_optional_tool_dependencies(tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    script = f"""
import builtins
import pathlib
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})

_real_import = builtins.__import__
_blocked = {{"firecrawl", "fal_client"}}

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root in _blocked:
        raise ModuleNotFoundError(f"No module named '{{root}}'")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _guarded_import

import run_agent
from tools import honcho_tools

print(run_agent.__name__)
print(honcho_tools.__name__)
"""

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "run_agent" in result.stdout
    assert "tools.honcho_tools" in result.stdout
