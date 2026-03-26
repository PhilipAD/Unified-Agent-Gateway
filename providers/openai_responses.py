from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

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

BUILT_IN_TOOL_TYPES = frozenset({
    "web_search",
    "file_search",
    "code_interpreter",
    "computer_use",
    "image_generation",
    "mcp",
})


def _to_input_items(
    messages: List[NormalizedMessage],
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert normalized messages to Responses API input items.

    Returns ``(instructions, items)`` where *instructions* is extracted from
    the first system message (Responses API uses a dedicated field).
    """
    instructions: Optional[str] = None
    items: List[Dict[str, Any]] = []

    for m in messages:
        if m.role == Role.SYSTEM:
            instructions = m.content
            continue

        if m.role == Role.USER:
            items.append({"role": "user", "content": m.content})

        elif m.role == Role.ASSISTANT:
            if m.content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": m.content}],
                })
            for tc in m.tool_calls:
                items.append({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                })

        elif m.role == Role.TOOL:
            items.append({
                "type": "function_call_output",
                "call_id": m.tool_call_id or "",
                "output": m.content,
            })

    return instructions, items


def _to_tools(
    tools: Optional[List[ToolDefinition]],
    built_in_tools: Optional[List[Dict[str, Any]]] = None,
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Build the Responses API tools list.

    Merges user-defined function tools, built-in tool configs (web_search,
    code_interpreter, computer_use, image_generation, file_search, shell,
    local_shell, apply_patch, tool_search), and remote MCP server configs.
    """
    result: List[Dict[str, Any]] = []

    if tools:
        for t in tools:
            result.append({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.json_schema,
                "strict": False,
            })

    if built_in_tools:
        for bt in built_in_tools:
            result.append(bt)

    if mcp_servers:
        for mcp in mcp_servers:
            entry: Dict[str, Any] = {
                "type": "mcp",
                "server_label": mcp.get("server_label", "mcp"),
                "require_approval": mcp.get("require_approval", "never"),
            }
            if "server_url" in mcp:
                entry["server_url"] = mcp["server_url"]
            if "connector_id" in mcp:
                entry["connector_id"] = mcp["connector_id"]
            if "server_description" in mcp:
                entry["server_description"] = mcp["server_description"]
            if "headers" in mcp:
                entry["headers"] = mcp["headers"]
            if "authorization" in mcp:
                entry["authorization"] = mcp["authorization"]
            if "allowed_tools" in mcp:
                entry["allowed_tools"] = mcp["allowed_tools"]
            if "defer_loading" in mcp:
                entry["defer_loading"] = mcp["defer_loading"]
            result.append(entry)

    return result or None


class OpenAIResponsesProvider(BaseProvider):
    """Provider using the OpenAI Responses API (``/v1/responses``).

    Supports built-in tools (web_search, file_search, code_interpreter,
    computer_use, image_generation), remote MCP servers, reasoning effort,
    and stateful sessions via ``previous_response_id``.
    """

    name = "openai_responses"

    def _client(self) -> AsyncOpenAI:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        instructions, input_items = _to_input_items(messages)

        built_in_tools = kwargs.pop("built_in_tools", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning_summary = kwargs.pop("reasoning_summary", None)
        previous_response_id = kwargs.pop("previous_response_id", None)
        store = kwargs.pop("store", None)
        include = kwargs.pop("include", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions:
            api_kwargs["instructions"] = instructions

        tools_list = _to_tools(tools, built_in_tools, mcp_servers)
        if tools_list:
            api_kwargs["tools"] = tools_list

        if reasoning_effort or reasoning_summary:
            reasoning: Dict[str, Any] = {}
            if reasoning_effort:
                reasoning["effort"] = reasoning_effort
            if reasoning_summary:
                reasoning["summary"] = reasoning_summary
            api_kwargs["reasoning"] = reasoning

        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id

        if store is not None:
            api_kwargs["store"] = store

        if include:
            api_kwargs["include"] = include

        api_kwargs.update(kwargs)

        client = self._client()
        try:
            resp = await client.responses.create(**api_kwargs)
        except Exception as exc:
            raise GatewayError(
                f"OpenAI Responses API error: {exc}",
                provider=self.name,
            ) from exc

        content = ""
        tool_calls: List[ToolCall] = []
        thinking_content = ""

        for item in resp.output:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for part in item.content:
                    if getattr(part, "type", None) == "output_text":
                        content += part.text

            elif item_type == "function_call":
                args_raw = getattr(item, "arguments", "{}")
                arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                tool_calls.append(ToolCall(
                    id=getattr(item, "call_id", ""),
                    name=item.name,
                    arguments=arguments,
                ))

            elif item_type == "reasoning":
                for part in getattr(item, "summary", []) or []:
                    if getattr(part, "type", None) == "summary_text":
                        thinking_content += getattr(part, "text", "")

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            thinking_content=thinking_content or None,
        )

        usage: Dict[str, Any] = {}
        raw_usage = getattr(resp, "usage", None)
        if raw_usage:
            usage["input_tokens"] = getattr(raw_usage, "input_tokens", 0)
            usage["output_tokens"] = getattr(raw_usage, "output_tokens", 0)
            usage["total_tokens"] = getattr(raw_usage, "total_tokens", 0)
            input_details = getattr(raw_usage, "input_tokens_details", None)
            if input_details:
                cached = getattr(input_details, "cached_tokens", 0)
                if cached:
                    usage["cached_tokens"] = cached
            output_details = getattr(raw_usage, "output_tokens_details", None)
            if output_details:
                reasoning = getattr(output_details, "reasoning_tokens", 0)
                if reasoning:
                    usage["reasoning_tokens"] = reasoning

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
        instructions, input_items = _to_input_items(messages)

        built_in_tools = kwargs.pop("built_in_tools", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning_summary = kwargs.pop("reasoning_summary", None)
        previous_response_id = kwargs.pop("previous_response_id", None)
        store = kwargs.pop("store", None)
        include = kwargs.pop("include", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": True,
        }

        if instructions:
            api_kwargs["instructions"] = instructions

        tools_list = _to_tools(tools, built_in_tools, mcp_servers)
        if tools_list:
            api_kwargs["tools"] = tools_list

        if reasoning_effort or reasoning_summary:
            reasoning_cfg: Dict[str, Any] = {}
            if reasoning_effort:
                reasoning_cfg["effort"] = reasoning_effort
            if reasoning_summary:
                reasoning_cfg["summary"] = reasoning_summary
            api_kwargs["reasoning"] = reasoning_cfg

        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id

        if store is not None:
            api_kwargs["store"] = store

        if include:
            api_kwargs["include"] = include

        api_kwargs.update(kwargs)

        client = self._client()

        try:
            async with client.responses.stream(**api_kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "response.output_text.delta":
                        yield StreamEvent(type="chunk", delta=event.delta)

                    elif event_type == "response.reasoning_summary_text.delta":
                        yield StreamEvent(type="chunk", delta=event.delta)

                    elif event_type == "response.function_call_arguments.done":
                        args = json.loads(event.arguments) if event.arguments else {}
                        tc = ToolCall(
                            id=getattr(event, "call_id", ""),
                            name=getattr(event, "name", ""),
                            arguments=args,
                        )
                        yield StreamEvent(type="tool_call", tool_call=tc)

                    elif event_type == "response.mcp_call.completed":
                        yield StreamEvent(
                            type="metadata",
                            metadata={
                                "event": "mcp_call_completed",
                                "server_label": getattr(event, "server_label", ""),
                                "name": getattr(event, "name", ""),
                            },
                        )

                    elif event_type == "response.web_search_call.completed":
                        yield StreamEvent(
                            type="metadata",
                            metadata={"event": "web_search_completed"},
                        )

                    elif event_type == "response.completed":
                        final = getattr(event, "response", None)
                        if final:
                            raw_usage = getattr(final, "usage", None)
                            if raw_usage:
                                usage: Dict[str, Any] = {
                                    "input_tokens": getattr(raw_usage, "input_tokens", 0),
                                    "output_tokens": getattr(raw_usage, "output_tokens", 0),
                                    "total_tokens": getattr(raw_usage, "total_tokens", 0),
                                }
                                input_details = getattr(raw_usage, "input_tokens_details", None)
                                if input_details:
                                    cached = getattr(input_details, "cached_tokens", 0)
                                    if cached:
                                        usage["cached_tokens"] = cached
                                output_details = getattr(raw_usage, "output_tokens_details", None)
                                if output_details:
                                    reasoning_tok = getattr(output_details, "reasoning_tokens", 0)
                                    if reasoning_tok:
                                        usage["reasoning_tokens"] = reasoning_tok
                                yield StreamEvent(type="usage", usage=usage)

                    elif event_type in (
                        "response.failed",
                        "response.incomplete",
                    ):
                        err_data = getattr(event, "response", None)
                        error_msg = ""
                        if err_data:
                            err_obj = getattr(err_data, "error", None)
                            if err_obj:
                                error_msg = getattr(err_obj, "message", str(err_obj))
                        yield StreamEvent(
                            type="error",
                            error=error_msg or event_type,
                        )

        except Exception as exc:
            raise GatewayError(
                f"OpenAI Responses streaming error: {exc}",
                provider=self.name,
            ) from exc

        yield StreamEvent(type="done")
