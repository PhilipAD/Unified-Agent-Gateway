"""Tests for MCP config parsing and Gemini MCP bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import MCPServerPreset
from runtime.gemini_mcp_bridge import load_gemini_cli_mcp_presets
from tools.mcp_config_loader import parse_mcp_server_configs


def test_parse_mcp_server_configs_http() -> None:
    raw = {
        "remote": {"url": "https://example.com/mcp", "headers": {"X": "1"}},
        "stdio": {"command": "npx", "args": ["x"]},
    }
    out = parse_mcp_server_configs(raw, namespace_prefix="t.")
    assert "t.remote" in out
    assert isinstance(out["t.remote"], MCPServerPreset)
    assert out["t.remote"].url == "https://example.com/mcp"
    assert "t.stdio" not in out


def test_gemini_mcp_bridge_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    (fake_home / ".gemini").mkdir()
    (fake_home / ".gemini" / "settings.json").write_text(
        json.dumps({"mcpServers": {}}),
        encoding="utf-8",
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    gem = ws / ".gemini"
    gem.mkdir()
    (gem / "settings.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "u": {"httpUrl": "https://u.example/mcp", "type": "http"},
                }
            }
        ),
        encoding="utf-8",
    )
    presets = load_gemini_cli_mcp_presets(str(ws), system_config_dir=None)
    assert any("gemini." in k for k in presets)
    key = next(k for k in presets if k.endswith("u"))
    assert "example" in presets[key].url
