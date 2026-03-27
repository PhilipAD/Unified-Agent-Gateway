"""Parse ``mcpServers``-style dicts from agent config files into UAG presets.

Maps remote HTTP / SSE transports to :class:`config.settings.MCPServerPreset`.
Stdio/command-based servers are skipped with a warning (not bridgeable via
:class:`tools.mcp_http_client.InlineMCPClient`).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

from config.settings import MCPServerPreset

logger = logging.getLogger(__name__)

_ENV_INTERP = re.compile(r"\$\{env:([^}]+)\}")


def _expand_env_in_str(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        return os.environ.get(key, m.group(0))

    return _ENV_INTERP.sub(repl, value)


def _expand_env_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env_in_str(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_obj(v) for v in obj]
    return obj


def parse_mcp_server_configs(
    raw_servers: Optional[Dict[str, Any]],
    *,
    namespace_prefix: str = "",
    env_interpolation: bool = True,
) -> Dict[str, MCPServerPreset]:
    """Convert a tool's ``mcpServers`` mapping into ``MCPServerPreset`` entries.

    Skips stdio (``command`` / ``args`` without remote URL) and logs a warning.
    """
    if not raw_servers:
        return {}

    out: Dict[str, MCPServerPreset] = {}
    for name, entry in raw_servers.items():
        if not isinstance(entry, dict):
            logger.warning("MCP server '%s': entry is not an object — skipped", name)
            continue

        cfg = _expand_env_obj(dict(entry)) if env_interpolation else dict(entry)

        has_command = "command" in cfg and cfg.get("command")
        has_args = "args" in cfg
        url = (
            cfg.get("url")
            or cfg.get("serverUrl")
            or cfg.get("httpUrl")
            or cfg.get("http_url")
        )
        if isinstance(url, str):
            url = url.strip()

        if (has_command or has_args) and not url:
            logger.warning(
                "MCP server '%s': stdio/command transport not bridgeable to HTTP — skipped",
                name,
            )
            continue

        if not url:
            if cfg.get("type") in ("stdio", "command") or has_command:
                logger.warning(
                    "MCP server '%s': no remote URL — skipped (stdio not supported here)",
                    name,
                )
            else:
                logger.warning("MCP server '%s': no url/serverUrl/httpUrl — skipped", name)
            continue

        transport = cfg.get("transport") or cfg.get("type") or "streamable_http"
        if transport in ("http", "streamable_http", "streamable-http"):
            transport_lit: Any = "streamable_http"
        elif transport in ("sse", "legacy_sse"):
            transport_lit = "sse"
        else:
            # Gemini / Windsurf often use type field
            t = str(transport).lower()
            if "sse" in t:
                transport_lit = "sse"
            else:
                transport_lit = "streamable_http"

        headers: Dict[str, str] = {}
        raw_headers = cfg.get("headers") or cfg.get("http_headers") or {}
        if isinstance(raw_headers, dict):
            headers = {str(k): str(v) for k, v in raw_headers.items()}

        timeout = float(
            cfg.get("timeout")
            or cfg.get("timeout_seconds")
            or cfg.get("timeoutSeconds")
            or 30.0
        )

        metadata: Dict[str, Any] = {}
        for key in (
            "includeTools",
            "excludeTools",
            "include_tools",
            "exclude_tools",
            "disabledTools",
            "disabled_tools",
            "enabled_tools",
            "enabledTools",
        ):
            if key in cfg:
                metadata[key] = cfg[key]

        if cfg.get("oauth") or cfg.get("requiresOAuth"):
            logger.warning(
                "MCP server '%s': OAuth may require interactive flow — preset created; "
                "ensure tokens are in headers if needed",
                name,
            )

        ns = f"{namespace_prefix}{name}" if namespace_prefix else name
        preset = MCPServerPreset(
            url=str(url),
            transport=transport_lit,
            headers=headers,
            timeout_seconds=timeout,
            metadata=metadata,
        )
        out[ns] = preset

    return out


def merge_mcp_presets(
    base: Dict[str, MCPServerPreset],
    extra: Dict[str, MCPServerPreset],
) -> Dict[str, MCPServerPreset]:
    """Return a shallow copy of *base* with *extra* keys merged (extra wins)."""
    merged = dict(base)
    merged.update(extra)
    return merged
