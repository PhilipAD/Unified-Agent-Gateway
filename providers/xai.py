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

XAI_BASE_URL = "https://api.x.ai/v1"

BUILT_IN_TOOL_TYPES = frozenset({
    "web_search",
    "x_search",
    "code_interpreter",
    "code_execution",
    "file_search",
    "collections_search",
    "attachment_search",
    "mcp",
})


def _to_input_items(
    messages: List[NormalizedMessage],
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert normalized messages to Responses API input items."""
    instructions: Optional[str] = None
    items: List[Dict[str, Any]] = []

    for m in messages:
        if m.role == Role.SYSTEM:
            instructions = m.content
            continue

        if m.role == Role.USER:
            if isinstance(m.content, list):
                items.append({"role": "user", "content": m.content})
            else:
                items.append({"role": "user", "content": m.content})

        elif m.role == Role.ASSISTANT:
            if m.content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": m.content},
                    ],
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
    """Build the tools list for xAI Responses API."""
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
            if "server_description" in mcp:
                entry["server_description"] = mcp["server_description"]
            if "headers" in mcp:
                entry["headers"] = mcp["headers"]
            if "authorization" in mcp:
                entry["authorization"] = mcp["authorization"]
            if "allowed_tools" in mcp:
                entry["allowed_tools"] = mcp["allowed_tools"]
            result.append(entry)

    return result or None


class XAIProvider(BaseProvider):
    """Provider for xAI Grok models via OpenAI Responses API.

    Uses the OpenAI SDK pointed at xAI's base URL
    (https://api.x.ai/v1). Supports:
    - Grok reasoning and non-reasoning models
    - Built-in server-side tools: web_search, x_search,
      code_interpreter, collections_search, attachment_search
    - Remote MCP servers
    - Live search with search_parameters
    - Citations and inline citations
    - Cost tracking (cost_in_usd_ticks)
    - Deferred completions
    - Previous response ID for multi-turn
    """

    name = "xai"

    def _client(self) -> AsyncOpenAI:
        base_url = self.base_url or XAI_BASE_URL
        return AsyncOpenAI(api_key=self.api_key, base_url=base_url)

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
        previous_response_id = kwargs.pop(
            "previous_response_id", None,
        )
        store = kwargs.pop("store", None)
        search_parameters = kwargs.pop("search_parameters", None)
        include = kwargs.pop("include", None)
        deferred = kwargs.pop("deferred", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions:
            api_kwargs["instructions"] = instructions

        tools_list = _to_tools(tools, built_in_tools, mcp_servers)
        if tools_list:
            api_kwargs["tools"] = tools_list

        if reasoning_effort:
            api_kwargs["reasoning"] = {"effort": reasoning_effort}

        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id

        if store is not None:
            api_kwargs["store"] = store

        if include:
            api_kwargs["include"] = include

        if search_parameters:
            api_kwargs["search_parameters"] = search_parameters

        if deferred is not None:
            api_kwargs["deferred"] = deferred

        api_kwargs.update(kwargs)

        client = self._client()
        try:
            resp = await client.responses.create(**api_kwargs)
        except Exception as exc:
            raise GatewayError(
                f"xAI API error: {exc}",
                provider=self.name,
            ) from exc

        content = ""
        tool_calls: List[ToolCall] = []
        thinking_content = ""
        citations: List[Dict[str, Any]] = []

        for item in resp.output:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for part in item.content:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        content += part.text
                        annotations = getattr(part, "annotations", None)
                        if annotations:
                            for ann in annotations:
                                citations.append({
                                    "type": getattr(ann, "type", ""),
                                    "url": getattr(ann, "url", ""),
                                    "title": getattr(ann, "title", ""),
                                    "start_index": getattr(
                                        ann, "start_index", 0,
                                    ),
                                    "end_index": getattr(
                                        ann, "end_index", 0,
                                    ),
                                })

            elif item_type == "function_call":
                args_raw = getattr(item, "arguments", "{}")
                arguments = (
                    json.loads(args_raw)
                    if isinstance(args_raw, str)
                    else args_raw
                )
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
            usage["input_tokens"] = getattr(
                raw_usage, "input_tokens", 0,
            )
            usage["output_tokens"] = getattr(
                raw_usage, "output_tokens", 0,
            )
            usage["total_tokens"] = getattr(
                raw_usage, "total_tokens", 0,
            )
            input_det = getattr(raw_usage, "input_tokens_details", None)
            if input_det:
                cached = getattr(input_det, "cached_tokens", 0)
                if cached:
                    usage["cached_tokens"] = cached
            output_det = getattr(
                raw_usage, "output_tokens_details", None,
            )
            if output_det:
                reasoning = getattr(output_det, "reasoning_tokens", 0)
                if reasoning:
                    usage["reasoning_tokens"] = reasoning

        if citations:
            usage["citations"] = citations

        resp_citations = getattr(resp, "citations", None)
        if resp_citations:
            usage["source_urls"] = list(resp_citations)

        cost = getattr(resp, "cost_in_usd_ticks", None)
        if cost is not None:
            usage["cost_in_usd_ticks"] = cost

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
        previous_response_id = kwargs.pop(
            "previous_response_id", None,
        )
        store = kwargs.pop("store", None)
        search_parameters = kwargs.pop("search_parameters", None)
        include = kwargs.pop("include", None)
        kwargs.pop("deferred", None)

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

        if reasoning_effort:
            api_kwargs["reasoning"] = {"effort": reasoning_effort}

        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id

        if store is not None:
            api_kwargs["store"] = store

        if include:
            api_kwargs["include"] = include

        if search_parameters:
            api_kwargs["search_parameters"] = search_parameters

        api_kwargs.update(kwargs)

        client = self._client()

        try:
            async with client.responses.stream(**api_kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "response.output_text.delta":
                        yield StreamEvent(
                            type="chunk", delta=event.delta,
                        )

                    elif event_type in (
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_text.delta",
                    ):
                        yield StreamEvent(
                            type="chunk", delta=event.delta,
                        )

                    elif event_type == (
                        "response.function_call_arguments.done"
                    ):
                        args = (
                            json.loads(event.arguments)
                            if event.arguments
                            else {}
                        )
                        tc = ToolCall(
                            id=getattr(event, "call_id", ""),
                            name=getattr(event, "name", ""),
                            arguments=args,
                        )
                        yield StreamEvent(
                            type="tool_call", tool_call=tc,
                        )

                    elif event_type in (
                        "response.web_search_call.completed",
                        "response.x_search_call.completed",
                    ):
                        yield StreamEvent(
                            type="metadata",
                            metadata={"event": event_type},
                        )

                    elif event_type == "response.completed":
                        final = getattr(event, "response", None)
                        if final:
                            raw_usage = getattr(final, "usage", None)
                            if raw_usage:
                                usage: Dict[str, Any] = {
                                    "input_tokens": getattr(
                                        raw_usage, "input_tokens", 0,
                                    ),
                                    "output_tokens": getattr(
                                        raw_usage, "output_tokens", 0,
                                    ),
                                    "total_tokens": getattr(
                                        raw_usage, "total_tokens", 0,
                                    ),
                                }
                                yield StreamEvent(
                                    type="usage", usage=usage,
                                )

        except Exception as exc:
            raise GatewayError(
                f"xAI streaming error: {exc}",
                provider=self.name,
            ) from exc

        yield StreamEvent(type="done")
