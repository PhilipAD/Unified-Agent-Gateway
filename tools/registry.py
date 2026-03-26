from __future__ import annotations

from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from core.types import ToolDefinition

ToolFn = Callable[..., Awaitable[Any]]


class ToolSource(str, Enum):
    PYTHON = "python"
    MCP = "mcp"
    HTTP = "http"
    CONTEXT_FORGE = "context_forge"


class RegisteredTool:
    def __init__(
        self,
        name: str,
        description: str,
        json_schema: Dict[str, Any],
        source: ToolSource,
        handler: ToolFn,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.json_schema = json_schema
        self.source = source
        self.handler = handler
        self.metadata = metadata or {}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        json_schema: Dict[str, Any],
        source: ToolSource,
        handler: ToolFn,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            json_schema=json_schema,
            source=source,
            handler=handler,
            metadata=metadata,
        )

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_for_provider(self) -> List[ToolDefinition]:
        """Return provider-agnostic tool definitions (name, description, schema only)."""
        return [
            ToolDefinition(
                name=t.name,
                description=t.description,
                json_schema=t.json_schema,
            )
            for t in self._tools.values()
        ]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_registered(self) -> List[RegisteredTool]:
        """Return full registered tool objects, including source metadata."""
        return list(self._tools.values())

    def copy(self) -> "ToolRegistry":
        """Shallow copy registry entries for per-request isolation."""
        new_registry = ToolRegistry()
        for tool in self._tools.values():
            new_registry.register(
                name=tool.name,
                description=tool.description,
                json_schema=tool.json_schema,
                source=tool.source,
                handler=tool.handler,
                metadata=dict(tool.metadata),
            )
        return new_registry
