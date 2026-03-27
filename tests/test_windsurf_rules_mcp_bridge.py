"""Tests for Windsurf rules and MCP bridge."""

from __future__ import annotations

import json
from pathlib import Path

from context.windsurf_rules import load_windsurf_rules
from runtime.windsurf_mcp_bridge import load_windsurf_mcp_presets


def test_windsurf_rules_workspace(tmp_path: Path) -> None:
    rules = tmp_path / ".windsurf" / "rules"
    rules.mkdir(parents=True)
    (rules / "py.md").write_text(
        "---\ntrigger: always_on\n---\nUse Python 3.11",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text("agents content", encoding="utf-8")
    text = load_windsurf_rules(str(tmp_path))
    assert "Use Python 3.11" in text
    assert "agents content" in text


def test_windsurf_mcp_bridge_file(tmp_path: Path) -> None:
    p = tmp_path / "mcp_config.json"
    p.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "r": {"serverUrl": "https://m.example/mcp", "headers": {"A": "b"}},
                }
            }
        ),
        encoding="utf-8",
    )
    presets = load_windsurf_mcp_presets(str(p))
    assert any(k.endswith("r") for k in presets)
