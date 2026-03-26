from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

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

AGENT_TOOL_TYPES = frozenset({
    "web_search",
    "web_search_premium",
    "code_interpreter",
    "image_generation",
    "document_library",
    "connector",
})


def _to_mistral_messages(
    messages: List[NormalizedMessage],
) -> List[Dict[str, Any]]:
    """Convert normalized messages to Mistral Messages format."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        msg: Dict[str, Any] = {
            "role": ROLE_MAP[m.role],
        }

        if m.role == Role.USER and isinstance(m.content, list):
            msg["content"] = _convert_content_parts(m.content)
        else:
            msg["content"] = m.content

        if m.role == Role.ASSISTANT and m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else tc.arguments
                        ),
                    },
                }
                for tc in m.tool_calls
            ]

        if m.role == Role.ASSISTANT and hasattr(m, "prefix") and m.prefix:
            msg["prefix"] = True

        if m.role == Role.TOOL and m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name

        out.append(msg)
    return out


def _convert_content_parts(parts: List[Any]) -> List[Dict[str, Any]]:
    """Convert multi-modal content parts for Mistral."""
    blocks: List[Dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            ptype = part.get("type", "text")
            if ptype == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url_info = part.get("image_url", {})
                url = (
                    url_info.get("url", "")
                    if isinstance(url_info, dict)
                    else url_info
                )
                entry: Dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
                detail = (
                    url_info.get("detail")
                    if isinstance(url_info, dict)
                    else None
                )
                if detail:
                    entry["image_url"]["detail"] = detail
                blocks.append(entry)
            elif ptype == "document_url":
                blocks.append(part)
            elif ptype == "file":
                blocks.append(part)
            elif ptype == "input_audio":
                blocks.append(part)
    return blocks


def _to_tools(
    tools: Optional[List[ToolDefinition]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [
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


class MistralProvider(BaseProvider):
    """Provider for Mistral AI with full SDK integration.

    Supports:
    - Chat completions with function calling
    - Vision/multimodal (images, documents, audio, files)
    - Structured outputs (json_object, json_schema)
    - Reasoning (reasoning_effort, prompt_mode)
    - Guardrails (moderation)
    - Prediction (speculative decoding)
    - Agents API with built-in tools (web_search, code_interpreter,
      image_generation, document_library, connectors)
    - Streaming
    """

    name = "mistral"

    def _client(self):
        from mistralai import Mistral

        kwargs: Dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["server_url"] = self.base_url
        return Mistral(**kwargs)

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        agent_id = kwargs.pop("agent_id", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        prompt_mode = kwargs.pop("prompt_mode", None)
        response_format = kwargs.pop("response_format", None)
        guardrails = kwargs.pop("guardrails", None)
        safe_prompt = kwargs.pop("safe_prompt", None)
        prediction = kwargs.pop("prediction", None)
        kwargs.pop("agent_tools", None)

        mistral_msgs = _to_mistral_messages(messages)
        tool_list = _to_tools(tools)

        client = self._client()

        api_kwargs: Dict[str, Any] = {
            "messages": mistral_msgs,
        }

        if agent_id:
            api_kwargs["agent_id"] = agent_id
        else:
            api_kwargs["model"] = self.model

        if tool_list:
            api_kwargs["tools"] = tool_list
        if reasoning_effort:
            api_kwargs["reasoning_effort"] = reasoning_effort
        if prompt_mode:
            api_kwargs["prompt_mode"] = prompt_mode
        if response_format:
            api_kwargs["response_format"] = response_format
        if guardrails:
            api_kwargs["guardrails"] = guardrails
        if safe_prompt is not None:
            api_kwargs["safe_prompt"] = safe_prompt
        if prediction:
            api_kwargs["prediction"] = prediction

        api_kwargs.update(kwargs)

        try:
            if agent_id:
                resp = await client.agents.complete_async(**api_kwargs)
            else:
                resp = await client.chat.complete_async(**api_kwargs)
        except Exception as exc:
            raise GatewayError(
                f"Mistral API error: {exc}",
                provider=self.name,
            ) from exc

        choice = resp.choices[0]
        msg = choice.message
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            text_parts = []
            for chunk in content:
                if hasattr(chunk, "text"):
                    text_parts.append(chunk.text)
            content = "".join(text_parts)

        tool_calls: List[ToolCall] = []
        raw_tcs = getattr(msg, "tool_calls", None)
        if raw_tcs:
            for tc in raw_tcs:
                fn = tc.function
                args_raw = fn.arguments
                arguments = (
                    json.loads(args_raw)
                    if isinstance(args_raw, str)
                    else (args_raw or {})
                )
                tool_calls.append(ToolCall(
                    id=getattr(tc, "id", "") or "",
                    name=fn.name,
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
            usage["input_tokens"] = getattr(
                raw_usage, "prompt_tokens", 0,
            ) or 0
            usage["output_tokens"] = getattr(
                raw_usage, "completion_tokens", 0,
            ) or 0
            usage["total_tokens"] = getattr(
                raw_usage, "total_tokens", 0,
            ) or 0

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason:
            usage["finish_reason"] = str(finish_reason)

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
        agent_id = kwargs.pop("agent_id", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        prompt_mode = kwargs.pop("prompt_mode", None)
        response_format = kwargs.pop("response_format", None)
        guardrails = kwargs.pop("guardrails", None)
        safe_prompt = kwargs.pop("safe_prompt", None)
        prediction = kwargs.pop("prediction", None)
        kwargs.pop("agent_tools", None)

        mistral_msgs = _to_mistral_messages(messages)
        tool_list = _to_tools(tools)

        client = self._client()

        api_kwargs: Dict[str, Any] = {
            "messages": mistral_msgs,
        }

        if agent_id:
            api_kwargs["agent_id"] = agent_id
        else:
            api_kwargs["model"] = self.model

        if tool_list:
            api_kwargs["tools"] = tool_list
        if reasoning_effort:
            api_kwargs["reasoning_effort"] = reasoning_effort
        if prompt_mode:
            api_kwargs["prompt_mode"] = prompt_mode
        if response_format:
            api_kwargs["response_format"] = response_format
        if guardrails:
            api_kwargs["guardrails"] = guardrails
        if safe_prompt is not None:
            api_kwargs["safe_prompt"] = safe_prompt
        if prediction:
            api_kwargs["prediction"] = prediction

        api_kwargs.update(kwargs)

        pending_tool_calls: Dict[int, Dict[str, Any]] = {}

        try:
            if agent_id:
                stream_resp = await client.agents.stream_async(
                    **api_kwargs,
                )
            else:
                stream_resp = await client.chat.stream_async(
                    **api_kwargs,
                )

            async for event in stream_resp:
                chunk = event.data
                if not chunk.choices:
                    raw_usage = getattr(chunk, "usage", None)
                    if raw_usage:
                        usage: Dict[str, Any] = {
                            "input_tokens": getattr(
                                raw_usage, "prompt_tokens", 0,
                            ) or 0,
                            "output_tokens": getattr(
                                raw_usage, "completion_tokens", 0,
                            ) or 0,
                        }
                        yield StreamEvent(type="usage", usage=usage)
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                content = getattr(delta, "content", None)
                if content:
                    if isinstance(content, str):
                        yield StreamEvent(type="chunk", delta=content)
                    elif isinstance(content, list):
                        for c in content:
                            if hasattr(c, "text"):
                                yield StreamEvent(
                                    type="chunk", delta=c.text,
                                )

                raw_tcs = getattr(delta, "tool_calls", None)
                if raw_tcs:
                    for tc_delta in raw_tcs:
                        idx = getattr(tc_delta, "index", 0) or 0
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": getattr(tc_delta, "id", "") or "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = pending_tool_calls[idx]
                        tc_id = getattr(tc_delta, "id", None)
                        if tc_id:
                            entry["id"] = tc_id
                        fn = getattr(tc_delta, "function", None)
                        if fn:
                            fn_name = getattr(fn, "name", None)
                            if fn_name:
                                entry["name"] = fn_name
                            fn_args = getattr(fn, "arguments", None)
                            if fn_args:
                                if isinstance(fn_args, str):
                                    entry["arguments"] += fn_args
                                else:
                                    entry["arguments"] = json.dumps(
                                        fn_args,
                                    )

                finish = getattr(choice, "finish_reason", None)
                if finish and str(finish) == "tool_calls":
                    for _idx in sorted(pending_tool_calls):
                        e = pending_tool_calls[_idx]
                        args = (
                            json.loads(e["arguments"])
                            if e["arguments"]
                            else {}
                        )
                        tc = ToolCall(
                            id=e["id"], name=e["name"], arguments=args,
                        )
                        yield StreamEvent(type="tool_call", tool_call=tc)
                    pending_tool_calls.clear()

        except Exception as exc:
            raise GatewayError(
                f"Mistral streaming error: {exc}",
                provider=self.name,
            ) from exc

        yield StreamEvent(type="done")
