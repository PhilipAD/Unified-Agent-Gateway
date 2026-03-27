"""Run ``codex mcp-server`` as a stdio MCP client (optional dependency: ``mcp``)."""

from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Sequence

from tools.mcp_loader import load_mcp_tools_from_server
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_CODEX_MCP_STACK: AsyncExitStack | None = None


class _StdioMCPClientAdapter:
    """Adapts MCP ``ClientSession`` over stdio to :class:`tools.mcp_loader.MCPClient`."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def list_tools(self) -> Sequence[Any]:
        from tools.mcp_http_client import _MCPToolAdapter  # noqa: SLF001

        result = await self._session.list_tools()
        return [_MCPToolAdapter(t) for t in result.tools]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        from tools.mcp_http_client import _extract_text_from_content

        result = await self._session.call_tool(name, arguments)
        if getattr(result, "isError", False):
            raise RuntimeError(
                f"MCP tool '{name}' error: {_extract_text_from_content(result.content)}"
            )
        return _extract_text_from_content(result.content)


async def load_codex_mcp_tools(
    registry: ToolRegistry,
    *,
    namespace: str = "codex",
    command: str = "codex",
    extra_args: Optional[List[str]] = None,
) -> int:
    """Start ``codex mcp-server`` and register its tools under *namespace*."""
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as exc:
        logger.error("Codex MCP bridge requires the 'mcp' package: %s", exc)
        return 0

    args = ["mcp-server"]
    if extra_args:
        args.extend(extra_args)
    params = StdioServerParameters(command=command, args=args, env=dict(os.environ))
    stack = AsyncExitStack()
    await stack.__aenter__()
    try:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        adapter = _StdioMCPClientAdapter(session)
        count = await load_mcp_tools_from_server(registry, adapter, namespace)
        logger.info("Registered %d tools from Codex MCP (%s)", count, namespace)
        global _CODEX_MCP_STACK
        _CODEX_MCP_STACK = stack
        return count
    except Exception as exc:
        await stack.aclose()
        logger.error("Codex MCP bridge failed: %s", exc)
        return 0
