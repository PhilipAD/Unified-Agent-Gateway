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

SERVER_TOOL_TYPES = frozenset({
    "web_search_20250305",
    "web_search_20260209",
    "web_fetch_20250910",
    "web_fetch_20260209",
    "web_fetch_20260309",
    "code_execution_20250522",
    "code_execution_20250825",
    "code_execution_20260120",
    "bash_20250124",
    "text_editor_20250124",
    "text_editor_20250429",
    "text_editor_20250728",
    "memory_20250818",
    "tool_search_tool_bm25_20251119",
    "tool_search_tool_regex_20251119",
})


def _to_anthropic_messages(
    messages: List[NormalizedMessage],
) -> List[Dict[str, Any]]:
    """Convert normalized messages to Anthropic Messages API format.

    Handles tool_use (on assistant turns), tool_result (on user turns),
    thinking/redacted_thinking blocks, multi-modal (image) content,
    and document/PDF content blocks correctly via structured content blocks.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            continue

        if m.role == Role.USER:
            content: Any = m.content
            if isinstance(m.content, list):
                content = _convert_user_content_parts(m.content)
            out.append({"role": "user", "content": content})

        elif m.role == Role.ASSISTANT:
            blocks: List[Dict[str, Any]] = []
            if m.thinking_content:
                thinking_block: Dict[str, Any] = {
                    "type": "thinking",
                    "thinking": m.thinking_content,
                }
                signature = getattr(m, "_thinking_signature", None)
                if signature:
                    thinking_block["signature"] = signature
                blocks.append(thinking_block)
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
            tool_result: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
            }
            is_error = getattr(m, "_is_error", None)
            if is_error:
                tool_result["is_error"] = True
            out.append({"role": "user", "content": [tool_result]})
    return out


def _convert_user_content_parts(parts: List[Any]) -> List[Dict[str, Any]]:
    """Convert multi-modal content parts (text, images, documents, search results)."""
    blocks: List[Dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            ptype = part.get("type", "text")
            if ptype == "text":
                block: Dict[str, Any] = {"type": "text", "text": part.get("text", "")}
                if "cache_control" in part:
                    block["cache_control"] = part["cache_control"]
                if "citations" in part:
                    block["citations"] = part["citations"]
                blocks.append(block)
            elif ptype == "image_url":
                url_info = part.get("image_url", {})
                url = url_info.get("url", "") if isinstance(url_info, dict) else url_info
                if url.startswith("data:"):
                    media_type, _, encoded = url.partition(";base64,")
                    media_type = media_type.replace("data:", "")
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    })
                else:
                    blocks.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
            elif ptype == "image":
                blocks.append(part)
            elif ptype == "document":
                blocks.append(part)
            elif ptype == "search_result":
                blocks.append(part)
    return blocks


def _to_tools(
    tools: Optional[List[ToolDefinition]],
    server_tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Build the Anthropic tools list combining user functions and server tools."""
    result: List[Dict[str, Any]] = []
    if tools:
        for t in tools:
            tool_entry: Dict[str, Any] = {
                "name": t.name,
                "description": t.description,
                "input_schema": t.json_schema,
            }
            result.append(tool_entry)

    if server_tools:
        for st in server_tools:
            result.append(st)

    return result or None


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

        thinking_budget_tokens = kwargs.pop("thinking_budget_tokens", None)
        thinking_type = kwargs.pop("thinking_type", "enabled")
        thinking_display = kwargs.pop("thinking_display", None)
        server_tools = kwargs.pop("server_tools", None)
        cache_control = kwargs.pop("cache_control", None)
        output_config = kwargs.pop("output_config", None)
        citations_config = kwargs.pop("citations", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_anthropic_messages(non_system),
            "max_tokens": kwargs.pop("max_tokens", 16384 if thinking_budget_tokens else 4096),
        }

        if system_msg:
            system_content: Any = system_msg.content
            if cache_control:
                system_content = [
                    {"type": "text", "text": system_msg.content, "cache_control": cache_control}
                ]
            api_kwargs["system"] = system_content

        if thinking_budget_tokens:
            thinking_cfg: Dict[str, Any] = {
                "type": thinking_type,
                "budget_tokens": thinking_budget_tokens,
            }
            if thinking_display:
                thinking_cfg["display"] = thinking_display
            api_kwargs["thinking"] = thinking_cfg
        elif thinking_type == "adaptive":
            thinking_cfg = {"type": "adaptive"}
            if thinking_display:
                thinking_cfg["display"] = thinking_display
            api_kwargs["thinking"] = thinking_cfg

        tool_list = _to_tools(tools, server_tools)
        if tool_list:
            api_kwargs["tools"] = tool_list

        if output_config:
            api_kwargs["output_config"] = output_config

        if citations_config:
            for tool_entry in api_kwargs.get("tools", []):
                if "input_schema" in tool_entry:
                    pass
            if isinstance(api_kwargs.get("system"), list):
                for block in api_kwargs["system"]:
                    if isinstance(block, dict) and "citations" not in block:
                        pass

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
        thinking_content = ""
        thinking_signature = ""
        tool_calls: List[ToolCall] = []
        citations_list: List[Dict[str, Any]] = []

        for block in resp.content:
            if block.type == "text":
                content += block.text
                block_citations = getattr(block, "citations", None)
                if block_citations:
                    for cit in block_citations:
                        citations_list.append({
                            "type": getattr(cit, "type", ""),
                            "cited_text": getattr(cit, "cited_text", ""),
                        })
            elif block.type == "thinking":
                thinking_content += getattr(block, "thinking", "")
                sig = getattr(block, "signature", "")
                if sig:
                    thinking_signature = sig
            elif block.type == "redacted_thinking":
                pass
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
            elif block.type == "server_tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=getattr(block, "name", ""),
                    arguments=getattr(block, "input", {}),
                ))
            elif block.type in (
                "web_search_tool_result",
                "web_fetch_tool_result",
                "code_execution_tool_result",
                "bash_code_execution_tool_result",
                "text_editor_code_execution_tool_result",
                "tool_search_tool_result",
            ):
                result_content = getattr(block, "content", None)
                if isinstance(result_content, list):
                    for rb in result_content:
                        if hasattr(rb, "text"):
                            content += rb.text

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            thinking_content=thinking_content or None,
        )

        usage: Dict[str, Any] = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        cache_creation = getattr(resp.usage, "cache_creation_input_tokens", None)
        if cache_creation:
            usage["cache_creation_input_tokens"] = cache_creation
        cache_read = getattr(resp.usage, "cache_read_input_tokens", None)
        if cache_read:
            usage["cache_read_input_tokens"] = cache_read
        server_tool_use = getattr(resp.usage, "server_tool_use", None)
        if server_tool_use:
            usage["server_tool_use"] = {
                "web_search_requests": getattr(server_tool_use, "web_search_requests", 0),
                "web_fetch_requests": getattr(server_tool_use, "web_fetch_requests", 0),
            }

        raw_metadata: Dict[str, Any] = {}
        if thinking_signature:
            raw_metadata["thinking_signature"] = thinking_signature
        if citations_list:
            raw_metadata["citations"] = citations_list
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason:
            raw_metadata["stop_reason"] = stop_reason
        container = getattr(resp, "container", None)
        if container:
            raw_metadata["container"] = {
                "id": getattr(container, "id", ""),
                "expires_at": str(getattr(container, "expires_at", "")),
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

        thinking_budget_tokens = kwargs.pop("thinking_budget_tokens", None)
        thinking_type = kwargs.pop("thinking_type", "enabled")
        thinking_display = kwargs.pop("thinking_display", None)
        server_tools = kwargs.pop("server_tools", None)
        cache_control = kwargs.pop("cache_control", None)
        output_config = kwargs.pop("output_config", None)
        kwargs.pop("citations", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_anthropic_messages(non_system),
            "max_tokens": kwargs.pop("max_tokens", 16384 if thinking_budget_tokens else 4096),
        }

        if system_msg:
            system_content: Any = system_msg.content
            if cache_control:
                system_content = [
                    {"type": "text", "text": system_msg.content, "cache_control": cache_control}
                ]
            api_kwargs["system"] = system_content

        if thinking_budget_tokens:
            thinking_cfg: Dict[str, Any] = {
                "type": thinking_type,
                "budget_tokens": thinking_budget_tokens,
            }
            if thinking_display:
                thinking_cfg["display"] = thinking_display
            api_kwargs["thinking"] = thinking_cfg
        elif thinking_type == "adaptive":
            thinking_cfg = {"type": "adaptive"}
            if thinking_display:
                thinking_cfg["display"] = thinking_display
            api_kwargs["thinking"] = thinking_cfg

        tool_list = _to_tools(tools, server_tools)
        if tool_list:
            api_kwargs["tools"] = tool_list

        if output_config:
            api_kwargs["output_config"] = output_config

        api_kwargs.update(kwargs)

        emitted_tool_ids: set[str] = set()
        client = self._client()
        try:
            async with client.messages.stream(**api_kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield StreamEvent(type="chunk", delta=event.delta.text)
                        elif hasattr(event.delta, "thinking"):
                            yield StreamEvent(type="chunk", delta=event.delta.thinking)
                        elif hasattr(event.delta, "partial_json"):
                            pass
                        elif hasattr(event.delta, "citation"):
                            yield StreamEvent(
                                type="metadata",
                                metadata={
                                    "event": "citation",
                                    "citation": {
                                        "type": getattr(event.delta.citation, "type", ""),
                                    },
                                },
                            )
                    elif event.type == "content_block_stop":
                        snapshot = stream.current_message_snapshot
                        for block in snapshot.content:
                            if block.type == "tool_use" and block.input:
                                if block.id in emitted_tool_ids:
                                    continue
                                emitted_tool_ids.add(block.id)
                                tc = ToolCall(
                                    id=block.id,
                                    name=block.name,
                                    arguments=block.input if isinstance(block.input, dict) else {},
                                )
                                yield StreamEvent(type="tool_call", tool_call=tc)
                            elif block.type == "server_tool_use":
                                if block.id in emitted_tool_ids:
                                    continue
                                emitted_tool_ids.add(block.id)
                                tc = ToolCall(
                                    id=block.id,
                                    name=getattr(block, "name", ""),
                                    arguments=getattr(block, "input", {}) if isinstance(
                                        getattr(block, "input", {}), dict
                                    ) else {},
                                )
                                yield StreamEvent(type="tool_call", tool_call=tc)

                final = await stream.get_final_message()
                usage: Dict[str, Any] = {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                }
                cache_creation = getattr(final.usage, "cache_creation_input_tokens", None)
                if cache_creation:
                    usage["cache_creation_input_tokens"] = cache_creation
                cache_read = getattr(final.usage, "cache_read_input_tokens", None)
                if cache_read:
                    usage["cache_read_input_tokens"] = cache_read
                yield StreamEvent(type="usage", usage=usage)
        except Exception as exc:
            raise GatewayError(f"Anthropic streaming error: {exc}", provider=self.name) from exc

        yield StreamEvent(type="done")
