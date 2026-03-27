"""Load Gemini CLI ``settings.json`` MCP servers into :class:`MCPServerPreset`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import MCPServerPreset
from tools.mcp_config_loader import parse_mcp_server_configs

logger = logging.getLogger(__name__)


def _read_settings(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Gemini settings unreadable %s: %s", path, exc)
        return {}


def _gemini_settings_paths(workspace_dir: str, system_config_dir: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if system_config_dir:
        p = Path(system_config_dir).expanduser() / "settings.json"
        paths.append(p)
    paths.append(Path.home() / ".gemini" / "settings.json")
    paths.append(Path(workspace_dir).resolve() / ".gemini" / "settings.json")
    return paths


def load_gemini_cli_mcp_presets(
    workspace_dir: str = ".",
    *,
    system_config_dir: Optional[str] = None,
) -> Dict[str, MCPServerPreset]:
    """Merge MCP entries from system, user, workspace (later overrides earlier)."""
    merged_servers: Dict[str, Any] = {}
    for p in _gemini_settings_paths(workspace_dir, system_config_dir):
        data = _read_settings(p)
        servers = data.get("mcpServers") or data.get("mcp_servers")
        if isinstance(servers, dict):
            merged_servers.update(servers)
    return parse_mcp_server_configs(merged_servers, namespace_prefix="gemini.")
