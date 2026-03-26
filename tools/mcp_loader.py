"""Load tools from MCP servers and register them into ToolRegistry.

This module is intentionally protocol-agnostic: it accepts any ``client``
object that exposes ``list_tools()`` and ``call_tool(name, args)`` async
methods.  Concrete MCP client implementations (subprocess, TCP, HTTP) are
supplied externally.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Protocol, Sequence

from tools.registry import ToolFn, ToolRegistry, ToolSource

logger = logging.getLogger(__name__)


class MCPToolDescriptor(Protocol):
    name: str
    description: str
    schema: Dict[str, Any]  # JSON Schema of tool input


class MCPClient(Protocol):
    async def list_tools(self) -> Sequence[MCPToolDescriptor]: ...
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any: ...


def _make_handler(client: MCPClient, tool_name: str) -> ToolFn:
    """Factory that captures *client* and *tool_name* by value, avoiding
    late-binding closure bugs when registering tools in a loop."""

    async def handler(**kwargs: Any) -> Any:
        return await client.call_tool(tool_name, kwargs)

    return handler


async def load_mcp_tools_from_server(
    registry: ToolRegistry,
    client: MCPClient,
    namespace: str,
) -> int:
    """Discover tools on *client* and register them under *namespace*.

    Returns the number of tools registered.
    """
    mcp_tools = await client.list_tools()
    count = 0
    for t in mcp_tools:
        qualified_name = f"{namespace}.{t.name}"
        registry.register(
            name=qualified_name,
            description=t.description,
            json_schema=t.schema,
            source=ToolSource.MCP,
            handler=_make_handler(client, t.name),
            metadata={"namespace": namespace, "original_name": t.name},
        )
        count += 1
        logger.info("Registered MCP tool %s from namespace '%s'", qualified_name, namespace)
    return count
