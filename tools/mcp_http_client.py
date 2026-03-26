"""Inline per-request MCP client.

Wraps the official ``mcp`` SDK's ``ClientSession`` over either SSE or
Streamable HTTP transport.  Designed as an async context manager so the
session stays alive for the full duration of a gateway request (tool calls
happen inside the same session), then cleans up automatically.

Usage::

    async with InlineMCPClient(url="http://mcp-host/sse", transport="sse") as client:
        tools = await client.list_tools()
        result = await client.call_tool("my_tool", {"arg": "value"})
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Literal, Optional, Sequence

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class _MCPToolAdapter:
    """Thin adapter that makes an MCP Tool look like our MCPToolDescriptor Protocol."""

    def __init__(self, mcp_tool: Any) -> None:
        self.name: str = mcp_tool.name
        self.description: str = mcp_tool.description or ""
        # MCP uses inputSchema; normalize to our expected schema key
        raw_schema = getattr(mcp_tool, "inputSchema", None)
        if raw_schema is None:
            raw_schema = getattr(mcp_tool, "input_schema", None)
        if raw_schema is None:
            self.schema: Dict[str, Any] = {"type": "object"}
        elif hasattr(raw_schema, "model_dump"):
            self.schema = raw_schema.model_dump(exclude_none=True)
        elif isinstance(raw_schema, dict):
            self.schema = raw_schema
        else:
            self.schema = {"type": "object"}


def _extract_text_from_content(content: Any) -> str:
    """Convert MCP CallToolResult content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict):
                parts.append(block.get("text", json.dumps(block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if hasattr(content, "text"):
        return content.text
    return str(content)


class InlineMCPClient:
    """Async context manager for a per-request MCP session over HTTP."""

    def __init__(
        self,
        url: str,
        transport: Literal["sse", "streamable_http"] = "streamable_http",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.transport = transport
        self.headers = headers or {}
        self.timeout = timeout
        self._stack: AsyncExitStack = AsyncExitStack()
        self._session: Optional[ClientSession] = None

    async def __aenter__(self) -> "InlineMCPClient":
        await self._stack.__aenter__()
        try:
            if self.transport == "sse":
                read, write = await self._stack.enter_async_context(
                    sse_client(
                        url=self.url,
                        headers=self.headers,
                        timeout=self.timeout,
                    )
                )
            else:
                # streamable_http yields (read, write, get_session_id)
                read, write, _ = await self._stack.enter_async_context(
                    streamablehttp_client(
                        url=self.url,
                        headers=self.headers,
                        timeout=self.timeout,
                    )
                )

            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
            logger.info("MCP session initialized: %s (%s)", self.url, self.transport)
        except Exception as exc:
            await self._stack.aclose()
            raise RuntimeError(f"Failed to connect to MCP server at {self.url!r}: {exc}") from exc
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._stack.aclose()
        logger.info("MCP session closed: %s", self.url)

    async def list_tools(self) -> Sequence[_MCPToolAdapter]:
        if self._session is None:
            raise RuntimeError("MCP client not connected — use as async context manager")
        result = await self._session.list_tools()
        return [_MCPToolAdapter(t) for t in result.tools]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError("MCP client not connected — use as async context manager")
        result = await self._session.call_tool(name, arguments)
        if getattr(result, "isError", False):
            raise RuntimeError(
                f"MCP tool '{name}' returned error: {_extract_text_from_content(result.content)}"
            )
        return _extract_text_from_content(result.content)
