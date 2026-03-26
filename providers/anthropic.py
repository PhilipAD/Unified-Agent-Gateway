from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.types import (
    GatewayError,
    NormalizedMessage,
    NormalizedResponse,
    Role,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from providers.base import BaseProvider


def _to_anthropic_messages(
    messages: List[NormalizedMessage],
) -> List[Dict[str, Any]]:
    """Convert normalized messages to Anthropic Messages API format.

    Handles tool_use (on assistant turns) and tool_result (on user turns)
    correctly via structured content blocks.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            continue

        if m.role == Role.USER:
            out.append({"role": "user", "content": m.content})

        elif m.role == Role.ASSISTANT:
            blocks: List[Dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks or m.content})

        elif m.role == Role.TOOL:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
    return out


def _to_tools(
    tools: Optional[List[ToolDefinition]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.json_schema,
        }
        for t in tools
    ]


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def _client(self) -> AsyncAnthropic:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncAnthropic(**kwargs)

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        system_msg = next((m for m in messages if m.role == Role.SYSTEM), None)
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_anthropic_messages(non_system),
            "max_tokens": kwargs.pop("max_tokens", 4096),
        }
        if system_msg:
            api_kwargs["system"] = system_msg.content

        tool_list = _to_tools(tools)
        if tool_list:
            api_kwargs["tools"] = tool_list

        api_kwargs.update(kwargs)

        client = self._client()
        try:
            resp = await client.messages.create(**api_kwargs)
        except Exception as exc:
            raise GatewayError(
                f"Anthropic API error: {exc}",
                provider=self.name,
            ) from exc

        content = ""
        tool_calls: List[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        out_msg = NormalizedMessage(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return NormalizedResponse(
            messages=[out_msg],
            usage=usage,
            provider=self.name,
            model=self.model,
            raw=resp,
        )

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        system_msg = next((m for m in messages if m.role == Role.SYSTEM), None)
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_anthropic_messages(non_system),
            "max_tokens": kwargs.pop("max_tokens", 4096),
        }
        if system_msg:
            api_kwargs["system"] = system_msg.content

        tool_list = _to_tools(tools)
        if tool_list:
            api_kwargs["tools"] = tool_list

        api_kwargs.update(kwargs)

        client = self._client()
        try:
            async with client.messages.stream(**api_kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield StreamEvent(type="chunk", delta=event.delta.text)
                        elif hasattr(event.delta, "partial_json"):
                            pass  # tool input accumulation handled by SDK
                    elif event.type == "content_block_stop":
                        snapshot = stream.current_message_snapshot
                        for block in snapshot.content:
                            if block.type == "tool_use" and block.input:
                                tc = ToolCall(
                                    id=block.id,
                                    name=block.name,
                                    arguments=block.input if isinstance(block.input, dict) else {},
                                )
                                yield StreamEvent(type="tool_call", tool_call=tc)

                final = await stream.get_final_message()
                usage = {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                }
                yield StreamEvent(type="usage", usage=usage)
        except Exception as exc:
            raise GatewayError(f"Anthropic streaming error: {exc}", provider=self.name) from exc

        yield StreamEvent(type="done")
