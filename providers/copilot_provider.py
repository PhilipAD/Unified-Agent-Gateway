"""GitHub Copilot SDK provider (technical preview; optional dependency)."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, List, Optional

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

_COPILOT = None


def _load_copilot():
    global _COPILOT
    if _COPILOT is not None:
        return _COPILOT
    try:
        import github_copilot_sdk as gcs  # type: ignore

        _COPILOT = gcs
        return gcs
    except ImportError:
        try:
            import copilot_sdk as gcs  # type: ignore

            _COPILOT = gcs
            return gcs
        except ImportError:
            return None


class CopilotProvider(BaseProvider):
    name = "copilot"

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        gcs = _load_copilot()
        if gcs is None:
            raise GatewayError(
                "github-copilot-sdk is not installed. "
                "Install with: pip install 'unified-agents-sdk[copilot]'",
                provider=self.name,
                status_code=503,
            )
        if tools:
            logger.warning("Copilot SDK tool bridging is not wired in this preview provider.")
        prompt = self._flatten_messages(messages)
        # Public API varies by SDK version; call a generic entrypoint if present.
        runner = getattr(gcs, "run", None) or getattr(gcs, "complete", None)
        if callable(runner):
            result = runner(prompt, model=self.model)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[misc]
            text = getattr(result, "text", None) or getattr(result, "output", None) or str(result)
        else:
            raise GatewayError(
                "github_copilot_sdk has no recognized run/complete API in this environment",
                provider=self.name,
                status_code=501,
            )
        return NormalizedResponse(
            messages=[NormalizedMessage(role=Role.ASSISTANT, content=str(text))],
            usage={},
            provider=self.name,
            model=self.model,
        )

    def _flatten_messages(self, messages: List[NormalizedMessage]) -> str:
        parts: List[str] = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            c = m.content
            parts.append(f"{role}: {c}" if isinstance(c, str) else f"{role}: {c!r}")
        return "\n\n".join(parts)

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        resp = await self.run(messages, tools, **kwargs)
        if resp.messages:
            c = resp.messages[-1].content
            t = c if isinstance(c, str) else str(c)
            yield StreamEvent(type="chunk", delta=t)
        yield StreamEvent(type="done", response=resp)
