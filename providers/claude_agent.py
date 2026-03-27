"""Claude Agent SDK provider (Claude Code harness)."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from core.types import (
    GatewayError,
    NormalizedMessage,
    NormalizedResponse,
    Role,
    StreamEvent,
    ToolDefinition,
)
from providers.base import BaseProvider

logger = logging.getLogger(__name__)

_SDK_MODULE: Any = None


def _get_sdk():
    global _SDK_MODULE
    if _SDK_MODULE is None:
        try:
            import claude_agent_sdk as m  # type: ignore

            _SDK_MODULE = m
        except ImportError:
            _SDK_MODULE = False
    return _SDK_MODULE if _SDK_MODULE is not False else None


class ClaudeAgentProvider(BaseProvider):
    name = "claude_agent"

    def _build_prompt(self, messages: List[NormalizedMessage]) -> str:
        parts: List[str] = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            c = m.content
            text = c if isinstance(c, str) else str(c)
            parts.append(f"{role.upper()}:\n{text}")
        return "\n\n".join(parts)

    def _options(self, kwargs: Dict[str, Any]) -> Any:
        sdk = _get_sdk()
        assert sdk is not None
        ClaudeAgentOptions = getattr(sdk, "ClaudeAgentOptions", None)
        if ClaudeAgentOptions is None:
            raise GatewayError(
                "claude_agent_sdk.ClaudeAgentOptions missing",
                provider=self.name,
                status_code=503,
            )
        extra = {**self.extra, **kwargs}
        opts_kw: Dict[str, Any] = {"model": self.model}
        for key in (
            "allowed_tools",
            "permission_mode",
            "setting_sources",
            "system_prompt",
            "mcp_servers",
            "max_turns",
            "cwd",
            "env",
            "agents",
            "hooks",
            "plugins",
        ):
            if extra.get(key) is not None:
                opts_kw[key] = extra[key]
        return ClaudeAgentOptions(**opts_kw)

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        sdk = _get_sdk()
        if sdk is None:
            raise GatewayError(
                "claude-agent-sdk is not installed. "
                "Install with: pip install 'unified-agents-sdk[claude-agent]'",
                provider=self.name,
                status_code=503,
            )
        query_fn = getattr(sdk, "query", None)
        if not callable(query_fn):
            raise GatewayError(
                "claude_agent_sdk.query not found",
                provider=self.name,
                status_code=503,
            )

        if tools:
            logger.warning(
                "UAG tools were passed but are not auto-bridged; "
                "configure mcp_servers in profile.extra."
            )

        prompt = self._build_prompt(messages)
        options = self._options(kwargs)
        text_parts: List[str] = []
        usage: Dict[str, Any] = {}
        try:
            async for msg in query_fn(prompt=prompt, options=options):
                cls = type(msg).__name__
                if cls == "AssistantMessage" or "Assistant" in cls:
                    content = getattr(msg, "content", None)
                    if content is not None:
                        text_parts.append(str(content))
                if cls == "ResultMessage" or "Result" in cls:
                    usage["stop_reason"] = getattr(msg, "stop_reason", None)
                    usage["session_id"] = getattr(msg, "session_id", None)
                    u = getattr(msg, "usage", None)
                    if u is not None:
                        usage["usage_detail"] = str(u)
        except Exception as exc:
            name = type(exc).__name__
            status = 500
            if "NotFound" in name:
                status = 503
            elif "Connection" in name:
                status = 502
            raise GatewayError(str(exc), provider=self.name, status_code=status) from exc

        out = "\n".join(text_parts).strip()
        return NormalizedResponse(
            messages=[NormalizedMessage(role=Role.ASSISTANT, content=out)],
            usage=usage,
            provider=self.name,
            model=self.model,
        )

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        sdk = _get_sdk()
        if sdk is None:
            yield StreamEvent(type="error", error="claude-agent-sdk not installed")
            return
        query_fn = getattr(sdk, "query", None)
        if not callable(query_fn):
            yield StreamEvent(type="error", error="claude_agent_sdk.query not found")
            return
        if tools:
            logger.warning(
                "UAG tools were passed but are not auto-bridged; "
                "configure mcp_servers in profile.extra."
            )
        prompt = self._build_prompt(messages)
        options = self._options(kwargs)
        text_parts: List[str] = []
        usage: Dict[str, Any] = {}
        try:
            async for msg in query_fn(prompt=prompt, options=options):
                cls = type(msg).__name__
                if cls == "AssistantMessage" or "Assistant" in cls:
                    content = getattr(msg, "content", None)
                    if content is not None:
                        chunk = str(content)
                        text_parts.append(chunk)
                        yield StreamEvent(type="chunk", delta=chunk)
                if cls == "ResultMessage" or "Result" in cls:
                    usage["stop_reason"] = getattr(msg, "stop_reason", None)
                    usage["session_id"] = getattr(msg, "session_id", None)
                yield StreamEvent(type="metadata", metadata={"message_type": cls})
        except Exception as exc:
            yield StreamEvent(type="error", error=str(exc))
            return
        out = "\n".join(text_parts).strip()
        done = NormalizedResponse(
            messages=[NormalizedMessage(role=Role.ASSISTANT, content=out)],
            usage=usage,
            provider=self.name,
            model=self.model,
        )
        yield StreamEvent(type="done", response=done)
