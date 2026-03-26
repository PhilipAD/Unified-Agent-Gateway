from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from groq import AsyncGroq

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

ROLE_MAP = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}

COMPOUND_MODELS = frozenset({"compound-beta", "compound-beta-mini"})

VALID_BUILTIN_TOOLS = frozenset({
    "web_search",
    "code_interpreter",
    "browser_search",
})


class GroqProvider(BaseProvider):
    """Provider for Groq API with compound built-in tools, reasoning, and
    server-side MCP.

    Supports:
    - Standard function calling (all models): OpenAI-compatible chat completions
    - Compound built-in tools (compound-beta/compound-beta-mini): server-side
      web_search, code_interpreter, browser_search via compound_custom
    - Reasoning models: reasoning/reasoning_format/reasoning_effort/include_reasoning
    - Documents: inline text/JSON document context
    - Citations: citation_options with document_citation and function_citation
    - Search settings: country, domains, images
    - Remote MCP via Responses API: server-side MCP tool discovery and execution
    """

    name = "groq"

    def _client(self) -> AsyncGroq:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncGroq(**kwargs)

    def _is_compound(self) -> bool:
        return self.model in COMPOUND_MODELS

    def _msg_to_api(self, m: NormalizedMessage) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "role": ROLE_MAP[m.role],
            "content": m.content,
        }
        if m.role == Role.ASSISTANT and m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in m.tool_calls
            ]
        if m.role == Role.ASSISTANT and m.thinking_content:
            msg["reasoning"] = m.thinking_content
        if m.role == Role.TOOL and m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name
        return msg

    def _build_payload(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]],
        **options: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [self._msg_to_api(m) for m in messages],
            **options,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.json_schema,
                    },
                }
                for t in tools
            ]
        return payload

    @staticmethod
    def _parse_tool_calls(raw_calls: List[Any]) -> List[ToolCall]:
        result: List[ToolCall] = []
        for tc in raw_calls:
            fn = tc.function
            args_raw = fn.arguments or "{}"
            arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            result.append(ToolCall(id=tc.id, name=fn.name, arguments=arguments))
        return result

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        enabled_tools = kwargs.pop("enabled_tools", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        documents = kwargs.pop("documents", None)
        search_settings = kwargs.pop("search_settings", None)
        citation_options = kwargs.pop("citation_options", None)
        reasoning_format = kwargs.pop("reasoning_format", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        include_reasoning = kwargs.pop("include_reasoning", None)
        service_tier = kwargs.pop("service_tier", None)
        compound_models = kwargs.pop("compound_models", None)
        wolfram_settings = kwargs.pop("wolfram_settings", None)
        disable_tool_validation = kwargs.pop("disable_tool_validation", None)

        if mcp_servers:
            return await self._run_responses_mcp(messages, tools, mcp_servers, **kwargs)

        client = self._client()
        payload = self._build_payload(messages, tools, **kwargs)

        if self._is_compound() and enabled_tools:
            compound_custom: Dict[str, Any] = {
                "tools": {"enabled_tools": enabled_tools}
            }
            if wolfram_settings:
                compound_custom["tools"]["wolfram_settings"] = wolfram_settings
            if compound_models:
                compound_custom["models"] = compound_models
            payload["compound_custom"] = compound_custom

        if documents:
            payload["documents"] = documents

        if search_settings:
            payload["search_settings"] = search_settings

        if citation_options:
            payload["citation_options"] = citation_options

        if reasoning_format:
            payload["reasoning_format"] = reasoning_format
        elif include_reasoning is not None:
            payload["include_reasoning"] = include_reasoning

        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        if service_tier:
            payload["service_tier"] = service_tier

        if disable_tool_validation is not None:
            payload["disable_tool_validation"] = disable_tool_validation

        try:
            resp = await client.chat.completions.create(**payload)
        except Exception as exc:
            raise GatewayError(
                f"Groq API error: {exc}",
                provider=self.name,
            ) from exc

        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = self._parse_tool_calls(msg.tool_calls) if msg.tool_calls else []

        reasoning_content = getattr(msg, "reasoning", None)
        executed_tools_data = getattr(msg, "executed_tools", None)
        annotations_data = getattr(msg, "annotations", None)

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            thinking_content=reasoning_content,
        )

        usage: Dict[str, Any] = {}
        if resp.usage:
            usage["input_tokens"] = resp.usage.prompt_tokens or 0
            usage["output_tokens"] = resp.usage.completion_tokens or 0
            ct_details = getattr(resp.usage, "completion_tokens_details", None)
            if ct_details:
                reasoning_tokens = getattr(ct_details, "reasoning_tokens", None)
                if reasoning_tokens:
                    usage["reasoning_tokens"] = reasoning_tokens
            pt_details = getattr(resp.usage, "prompt_tokens_details", None)
            if pt_details:
                cached_tokens = getattr(pt_details, "cached_tokens", None)
                if cached_tokens:
                    usage["cached_tokens"] = cached_tokens
            for timing_field in ("completion_time", "prompt_time", "queue_time", "total_time"):
                val = getattr(resp.usage, timing_field, None)
                if val is not None:
                    usage[timing_field] = val

        if executed_tools_data:
            usage["executed_tools"] = _serialize_executed_tools(executed_tools_data)

        if annotations_data:
            usage["annotations"] = _serialize_annotations(annotations_data)

        usage_breakdown = getattr(resp, "usage_breakdown", None)
        if usage_breakdown:
            models = getattr(usage_breakdown, "models", None)
            if models:
                usage["usage_breakdown"] = [
                    {
                        "model": getattr(m, "model", ""),
                        "prompt_tokens": getattr(
                            getattr(m, "usage", None), "prompt_tokens", 0,
                        ),
                        "completion_tokens": getattr(
                            getattr(m, "usage", None), "completion_tokens", 0,
                        ),
                    }
                    for m in models
                ]

        x_groq = getattr(resp, "x_groq", None)
        if x_groq:
            groq_meta: Dict[str, Any] = {}
            groq_id = getattr(x_groq, "id", None)
            if groq_id:
                groq_meta["id"] = groq_id
            groq_usage = getattr(x_groq, "usage", None)
            if groq_usage:
                groq_meta["dram_cached_tokens"] = getattr(groq_usage, "dram_cached_tokens", 0)
                groq_meta["sram_cached_tokens"] = getattr(groq_usage, "sram_cached_tokens", 0)
            if groq_meta:
                usage["x_groq"] = groq_meta

        mcp_list = getattr(resp, "mcp_list_tools", None)
        if mcp_list:
            usage["mcp_list_tools"] = [
                {
                    "server_label": getattr(mt, "server_label", ""),
                    "tools": [
                        {
                            "name": getattr(t, "name", ""),
                            "description": getattr(t, "description", ""),
                        }
                        for t in (getattr(mt, "tools", None) or [])
                    ],
                }
                for mt in mcp_list
            ]

        return NormalizedResponse(
            messages=[out_msg],
            usage=usage,
            provider=self.name,
            model=self.model,
            raw=resp,
        )

    async def _run_responses_mcp(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]],
        mcp_servers: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> NormalizedResponse:
        """Use the Responses API for server-side MCP tool execution.

        Groq's Responses API is OpenAI-compatible, so we use the openai SDK
        pointed at Groq's base URL.
        """
        from openai import AsyncOpenAI

        base_url = (self.base_url or "https://api.groq.com/openai/v1").rstrip("/")
        oai_client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

        input_items: List[Dict[str, Any]] = []
        instructions: Optional[str] = None
        for m in messages:
            if m.role == Role.SYSTEM:
                instructions = m.content
                continue
            input_items.append({"role": ROLE_MAP[m.role], "content": m.content})

        api_tools: List[Dict[str, Any]] = []
        if tools:
            for t in tools:
                api_tools.append({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.json_schema,
                    "strict": False,
                })

        for mcp in mcp_servers:
            entry: Dict[str, Any] = {
                "type": "mcp",
                "server_label": mcp.get("server_label", "mcp"),
                "server_url": mcp["server_url"],
                "require_approval": mcp.get("require_approval", "never"),
            }
            if "server_description" in mcp:
                entry["server_description"] = mcp["server_description"]
            if "headers" in mcp:
                entry["headers"] = mcp["headers"]
            if "allowed_tools" in mcp:
                entry["allowed_tools"] = mcp["allowed_tools"]
            api_tools.append(entry)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }
        if instructions:
            api_kwargs["instructions"] = instructions
        if api_tools:
            api_kwargs["tools"] = api_tools
        api_kwargs.update(kwargs)

        try:
            resp = await oai_client.responses.create(**api_kwargs)
        except Exception as exc:
            raise GatewayError(
                f"Groq Responses API error: {exc}",
                provider=self.name,
            ) from exc

        content = ""
        tool_calls: List[ToolCall] = []
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

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        usage: Dict[str, Any] = {}
        raw_usage = getattr(resp, "usage", None)
        if raw_usage:
            usage = {
                "input_tokens": getattr(raw_usage, "input_tokens", 0),
                "output_tokens": getattr(raw_usage, "output_tokens", 0),
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
        enabled_tools = kwargs.pop("enabled_tools", None)
        kwargs.pop("mcp_servers", None)
        documents = kwargs.pop("documents", None)
        search_settings = kwargs.pop("search_settings", None)
        citation_options = kwargs.pop("citation_options", None)
        reasoning_format = kwargs.pop("reasoning_format", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        include_reasoning = kwargs.pop("include_reasoning", None)
        service_tier = kwargs.pop("service_tier", None)
        compound_models = kwargs.pop("compound_models", None)
        wolfram_settings = kwargs.pop("wolfram_settings", None)
        disable_tool_validation = kwargs.pop("disable_tool_validation", None)

        client = self._client()
        payload = self._build_payload(messages, tools, stream=True, **kwargs)

        if self._is_compound() and enabled_tools:
            compound_custom: Dict[str, Any] = {
                "tools": {"enabled_tools": enabled_tools}
            }
            if wolfram_settings:
                compound_custom["tools"]["wolfram_settings"] = wolfram_settings
            if compound_models:
                compound_custom["models"] = compound_models
            payload["compound_custom"] = compound_custom

        if documents:
            payload["documents"] = documents

        if search_settings:
            payload["search_settings"] = search_settings

        if citation_options:
            payload["citation_options"] = citation_options

        if reasoning_format:
            payload["reasoning_format"] = reasoning_format
        elif include_reasoning is not None:
            payload["include_reasoning"] = include_reasoning

        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        if service_tier:
            payload["service_tier"] = service_tier

        if disable_tool_validation is not None:
            payload["disable_tool_validation"] = disable_tool_validation

        pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}

        try:
            stream = await client.chat.completions.create(**payload)
            async for chunk in stream:
                if not chunk.choices:
                    if chunk.usage:
                        usage = {
                            "input_tokens": chunk.usage.prompt_tokens or 0,
                            "output_tokens": chunk.usage.completion_tokens or 0,
                        }
                        timing_fields = (
                            "completion_time", "prompt_time",
                            "queue_time", "total_time",
                        )
                        for timing_field in timing_fields:
                            val = getattr(chunk.usage, timing_field, None)
                            if val is not None:
                                usage[timing_field] = val
                    x_groq = getattr(chunk, "x_groq", None)
                    if x_groq:
                        x_usage = getattr(x_groq, "usage_breakdown", None)
                        if x_usage:
                            usage["usage_breakdown"] = x_usage
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta and delta.content:
                    yield StreamEvent(type="chunk", delta=delta.content)

                if delta and getattr(delta, "reasoning", None):
                    yield StreamEvent(type="chunk", delta=delta.reasoning)

                if delta and getattr(delta, "annotations", None):
                    ann_data = _serialize_annotations(delta.annotations)
                    yield StreamEvent(
                        type="metadata",
                        metadata={"event": "annotations", "annotations": ann_data},
                    )

                if delta and getattr(delta, "executed_tools", None):
                    et_data = _serialize_executed_tools(delta.executed_tools)
                    yield StreamEvent(
                        type="metadata",
                        metadata={"event": "executed_tools", "executed_tools": et_data},
                    )

                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = pending_tool_calls[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

                finish = choice.finish_reason
                if finish == "tool_calls":
                    for _idx in sorted(pending_tool_calls):
                        e = pending_tool_calls[_idx]
                        args = json.loads(e["arguments"]) if e["arguments"] else {}
                        tc = ToolCall(id=e["id"], name=e["name"], arguments=args)
                        yield StreamEvent(type="tool_call", tool_call=tc)
                    pending_tool_calls.clear()

        except Exception as exc:
            raise GatewayError(
                f"Groq streaming error: {exc}",
                provider=self.name,
            ) from exc

        if usage:
            yield StreamEvent(type="usage", usage=usage)
        yield StreamEvent(type="done")


def _serialize_executed_tools(executed_tools: Any) -> List[Dict[str, Any]]:
    """Serialize compound model executed_tools into plain dicts."""
    result = []
    for et in executed_tools:
        entry: Dict[str, Any] = {
            "type": getattr(et, "type", ""),
            "arguments": getattr(et, "arguments", ""),
            "index": getattr(et, "index", 0),
        }
        output = getattr(et, "output", None)
        if output:
            entry["output"] = output
        sr = getattr(et, "search_results", None)
        if sr:
            results_list = getattr(sr, "results", [])
            entry["search_results"] = [
                {
                    "title": getattr(r, "title", ""),
                    "url": getattr(r, "url", ""),
                    "content": getattr(r, "content", ""),
                }
                for r in (results_list or [])
            ]
            images = getattr(sr, "images", None)
            if images:
                entry["search_images"] = list(images)
        br = getattr(et, "browser_results", None)
        if br:
            entry["browser_results"] = [
                {
                    "title": getattr(b, "title", ""),
                    "url": getattr(b, "url", ""),
                    "content": getattr(b, "content", ""),
                }
                for b in br
            ]
        cr = getattr(et, "code_results", None)
        if cr:
            entry["code_results"] = [
                {
                    "text": getattr(c, "text", ""),
                    "png": getattr(c, "png", None),
                }
                for c in cr
            ]
        result.append(entry)
    return result


def _serialize_annotations(annotations: Any) -> List[Dict[str, Any]]:
    """Serialize citation annotations into plain dicts."""
    result = []
    for ann in annotations:
        entry: Dict[str, Any] = {"type": getattr(ann, "type", "")}
        dc = getattr(ann, "document_citation", None)
        if dc:
            entry["document_citation"] = {
                "document_id": getattr(dc, "document_id", ""),
                "start_index": getattr(dc, "start_index", 0),
                "end_index": getattr(dc, "end_index", 0),
            }
        fc = getattr(ann, "function_citation", None)
        if fc:
            entry["function_citation"] = {
                "tool_call_id": getattr(fc, "tool_call_id", ""),
                "start_index": getattr(fc, "start_index", 0),
                "end_index": getattr(fc, "end_index", 0),
            }
        result.append(entry)
    return result
