"""Regression checks for backend-abstraction docs staying in sync."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_honcho_integration_spec_html_matches_backend_abstraction_story():
    html = (PROJECT_ROOT / "docs" / "honcho-integration-spec.html").read_text(encoding="utf-8")

    assert "There is no plugin boundary." not in html
    assert "small backend seam" in html
    assert "experimental external backend factory can replace it out of tree" in html
    assert "openclaw honcho setup" not in html
    assert "openclaw honcho identity" not in html
    assert "openclaw honcho status" not in html
    assert '"  honcho status                    — show config + connection"' not in html
    assert '"  honcho mode [hybrid|honcho]       — show or set memory mode"' not in html
    assert '"  honcho sessions                  — list session mappings"' not in html
    assert '"  honcho map &lt;name&gt;                — map directory to session"' not in html
    assert '"  honcho identity [file] [--show]  — seed or show AI identity"' not in html
    assert '"  honcho setup                     — full interactive wizard"' not in html
    assert '"  hermes honcho status                    — show full config + connection"' in html
    assert '"  hermes honcho mode [hybrid|honcho]       — show or set memory mode"' in html
    assert '"  hermes honcho sessions                  — list directory→session mappings"' in html
    assert '"  hermes honcho map &lt;name&gt;                — map cwd to a session name"' in html
    assert '"  hermes honcho identity [&lt;file&gt;] [--show] — seed or show AI peer identity"' in html
    assert '"  hermes honcho setup                     — full interactive wizard"' in html
    assert "honcho mode [hybrid|honcho|local]" not in html
    assert "honcho mode [hybrid|honcho]" in html
    assert "userMemoryMode" not in html
    assert "agentMemoryMode" not in html
    assert "Local files only — skip Honcho sync for this peer" not in html


def test_memory_backend_abstraction_doc_covers_manifest_and_stability_contract():
    markdown = (PROJECT_ROOT / "docs" / "memory-backend-abstraction.md").read_text(
        encoding="utf-8"
    )

    assert "protocol_version" in markdown
    assert "backend_id" in markdown
    assert "display_name" in markdown
    assert "capabilities" in markdown
    assert "config_source" in markdown
    assert "Stability expectations" in markdown
    assert "Honcho remains the only built-in backend" in markdown
    assert "Hermes does not silently fall back to Honcho" in markdown
    assert "CLI loads the same backend factory directly" in markdown
    assert "because they all flow through `AIAgent`" not in markdown


def test_honcho_integration_spec_matches_current_memory_mode_contract():
    markdown = (PROJECT_ROOT / "docs" / "honcho-integration-spec.md").read_text(
        encoding="utf-8"
    )

    assert "`local` | Local files only" not in markdown
    assert "honcho mode [hybrid|honcho|local]" not in markdown
    assert "honcho mode [hybrid|honcho]" in markdown
    assert "userMemoryMode" not in markdown
    assert "agentMemoryMode" not in markdown
    assert '"memoryMode": {' in markdown
    assert "  honcho status                    — show config + connection" not in markdown
    assert "  honcho mode [hybrid|honcho]       — show or set memory mode" not in markdown
    assert "  honcho sessions                  — list session mappings" not in markdown
    assert "  honcho map <name>                — map directory to session" not in markdown
    assert "  honcho identity [file] [--show]  — seed or show AI identity" not in markdown
    assert "  honcho setup                     — full interactive wizard" not in markdown
    assert "  hermes honcho status                    — show full config + connection" in markdown
    assert "  hermes honcho mode [hybrid|honcho]       — show or set memory mode" in markdown
    assert "  hermes honcho sessions                  — list directory→session mappings" in markdown
    assert "  hermes honcho map <name>                — map cwd to a session name" in markdown
    assert "  hermes honcho identity [<file>] [--show] — seed or show AI peer identity" in markdown
    assert "  hermes honcho setup                     — full interactive wizard" in markdown
