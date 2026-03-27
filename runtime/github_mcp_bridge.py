"""Build a GitHub / Copilot remote MCP :class:`MCPServerPreset`."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from config.settings import MCPServerPreset

logger = logging.getLogger(__name__)

DEFAULT_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp"


def _github_token() -> Optional[str]:
    return (
        os.environ.get("COPILOT_GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )


def load_github_mcp_presets(
    *,
    url: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    token: Optional[str] = None,
    namespace: str = "github_mcp",
) -> Dict[str, MCPServerPreset]:
    """Remote streamable HTTP MCP preset (Bearer token).

    *toolsets* may be passed as query string later by caller if the server supports it;
    stored in metadata for documentation.
    """
    auth = token or _github_token()
    if not auth:
        logger.warning("GitHub MCP bridge: no token in COPILOT_GITHUB_TOKEN/GH_TOKEN/GITHUB_TOKEN")
    base = (url or DEFAULT_GITHUB_MCP_URL).rstrip("/")
    if toolsets:
        qs = ",".join(toolsets)
        full_url = f"{base}?toolsets={qs}"
    else:
        full_url = base
    headers: Dict[str, str] = {}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    preset = MCPServerPreset(
        url=full_url,
        transport="streamable_http",
        headers=headers,
        timeout_seconds=60.0,
        metadata={"toolsets": toolsets or []},
    )
    return {namespace: preset}
