from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from context.registry import ContextRegistry
from core.types import (
    NormalizedMessage,
    NormalizedResponse,
    Role,
    ToolDefinition,
    ToolResult,
)
from providers.base import BaseProvider
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class StepTrace:
    step: int
    type: str  # "model_call" | "tool_execution"
    duration_ms: float = 0.0
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    def __init__(
        self,
        provider: BaseProvider,
        tools: Optional[ToolRegistry] = None,
        contexts: Optional[ContextRegistry] = None,
        max_tool_hops: int = 6,
        tool_timeout: float = 30.0,
    ) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.contexts = contexts or ContextRegistry()
        self.max_tool_hops = max_tool_hops
        self.tool_timeout = tool_timeout

    async def _inject_context(
        self,
        messages: List[NormalizedMessage],
        **ctx_kwargs: Any,
    ) -> List[NormalizedMessage]:
        """Load registered context sources and prepend as a system message."""
        ctx_map = await self.contexts.load_all(**ctx_kwargs)
        if not ctx_map:
            return messages

        ctx_text = "\n\n".join(f"[{name}]\n{text}" for name, text in ctx_map.items() if text)
        if not ctx_text:
            return messages

        ctx_msg = NormalizedMessage(
            role=Role.SYSTEM,
            content=f"Additional context:\n{ctx_text}",
        )

        result = list(messages)
        sys_idx = next((i for i, m in enumerate(result) if m.role == Role.SYSTEM), None)
        if sys_idx is not None:
            result.insert(sys_idx + 1, ctx_msg)
        else:
            result.insert(0, ctx_msg)
        return result

    async def run_conversation(
        self,
        messages: List[NormalizedMessage],
        tool_defs: Optional[List[ToolDefinition]] = None,
        context_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """Run a full conversation loop with tool calling.

        Returns a NormalizedResponse with the final assistant message and
        the full conversation transcript in ``conversation``.
        """
        all_messages = await self._inject_context(list(messages), **(context_kwargs or {}))
        traces: List[StepTrace] = []
        last_response: Optional[NormalizedResponse] = None
        step = 0

        effective_tool_defs = tool_defs
        if effective_tool_defs is None and self.tools.list_names():
            effective_tool_defs = self.tools.list_for_provider()

        for _ in range(self.max_tool_hops):
            t0 = time.monotonic()
            last_response = await self.provider.run(
                messages=all_messages, tools=effective_tool_defs, **kwargs
            )
            elapsed = (time.monotonic() - t0) * 1000
            traces.append(StepTrace(step=step, type="model_call", duration_ms=elapsed))
            step += 1

            assistant_msg = last_response.messages[-1]
            all_messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                break

            tool_results: List[ToolResult] = []
            for tc in assistant_msg.tool_calls:
                t1 = time.monotonic()
                trace = StepTrace(
                    step=step,
                    type="tool_execution",
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                )
                try:
                    reg_tool = self.tools.get(tc.name)
                    result = await reg_tool.handler(**tc.arguments)
                    tool_results.append(ToolResult(tool_call_id=tc.id, output=result))
                except Exception as exc:
                    logger.error("Tool '%s' failed: %s", tc.name, exc, exc_info=True)
                    trace.error = str(exc)
                    tool_results.append(ToolResult(tool_call_id=tc.id, output=f"Error: {exc}"))
                trace.duration_ms = (time.monotonic() - t1) * 1000
                traces.append(trace)
                step += 1

            for tr in tool_results:
                tool_name = next(
                    (tc.name for tc in assistant_msg.tool_calls if tc.id == tr.tool_call_id),
                    None,
                )
                all_messages.append(
                    NormalizedMessage(
                        role=Role.TOOL,
                        content=str(tr.output),
                        tool_call_id=tr.tool_call_id,
                        name=tool_name,
                    )
                )

        if last_response is None:
            last_response = NormalizedResponse(messages=[])

        last_response.conversation = all_messages
        return last_response
