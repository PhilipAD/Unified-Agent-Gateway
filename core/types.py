from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class NormalizedMessage:
    role: Role
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    thinking_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        if self.thinking_content is not None:
            d["thinking_content"] = self.thinking_content
        return d


@dataclass
class ToolDefinition:
    name: str
    description: str
    json_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "json_schema": self.json_schema,
        }


@dataclass
class ToolResult:
    tool_call_id: str
    output: Any


@dataclass
class StreamEvent:
    """Normalized streaming event emitted by providers."""

    type: str  # "chunk", "tool_call", "usage", "metadata", "error", "done"
    delta: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    usage: Optional[Dict[str, Any]] = None
    response: Optional[NormalizedResponse] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.delta is not None:
            d["delta"] = self.delta
        if self.tool_call is not None:
            d["tool_call"] = {
                "id": self.tool_call.id,
                "name": self.tool_call.name,
                "arguments": self.tool_call.arguments,
            }
        if self.usage is not None:
            d["usage"] = self.usage
        if self.metadata is not None:
            d["metadata"] = self.metadata
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class NormalizedResponse:
    messages: List[NormalizedMessage] = field(default_factory=list)
    conversation: List[NormalizedMessage] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    provider: Optional[str] = None
    model: Optional[str] = None
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "usage": self.usage,
        }
        if self.provider:
            d["provider"] = self.provider
        if self.model:
            d["model"] = self.model
        if self.conversation:
            d["conversation"] = [m.to_dict() for m in self.conversation]
        return d


class GatewayError(Exception):
    """Base exception for gateway errors."""

    def __init__(self, message: str, provider: Optional[str] = None, status_code: int = 500):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
