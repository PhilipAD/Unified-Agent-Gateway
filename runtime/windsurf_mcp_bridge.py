"""Load Windsurf ``mcp_config.json`` into :class:`MCPServerPreset`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from config.settings import MCPServerPreset
from tools.mcp_config_loader import parse_mcp_server_configs

logger = logging.getLogger(__name__)


def load_windsurf_mcp_presets(config_path: Optional[str] = None) -> Dict[str, MCPServerPreset]:
    path = Path(
        config_path
        or str(Path.home() / ".codeium" / "windsurf" / "mcp_config.json")
    ).expanduser()
    if not path.is_file():
        logger.debug("Windsurf mcp_config not found: %s", path)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read Windsurf MCP config %s: %s", path, exc)
        return {}
    servers = data.get("mcpServers") or data.get("mcp_servers")
    if not isinstance(servers, dict):
        return {}
    return parse_mcp_server_configs(servers, namespace_prefix="windsurf.")
