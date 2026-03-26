from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from core.types import (
    GatewayError,
    NormalizedMessage,
    NormalizedResponse,
    Role,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from providers.openai_compatible import OpenAICompatibleProvider

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

ROLE_MAP = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}


class DeepSeekProvider(OpenAICompatibleProvider):
    """Provider for DeepSeek API with reasoning content capture.

    Extends the OpenAI-compatible provider to capture
    ``reasoning_content`` from deepseek-reasoner responses and support
    the ``thinking`` parameter for toggling thinking mode.

    Supports:
    - Thinking mode: ``thinking=True`` or ``thinking={"type": "enabled"}``
    - Reasoning content capture from non-streaming and streaming responses
    - Multi-turn reasoning passthrough (reasoning_content in assistant messages)
    - Detailed usage with reasoning_tokens, cache_hit/miss_tokens
    - finish_reason capture
    """

    name = "deepseek"

    def _effective_base_url(self) -> str:
        return (self.base_url or DEEPSEEK_BASE_URL).rstrip("/")

    def _msg_to_api(self, m: NormalizedMessage) -> Dict[str, Any]:
        msg = super()._msg_to_api(m)
        if m.role == Role.ASSISTANT and m.thinking_content:
            msg["reasoning_content"] = m.thinking_content
        return msg

    def _normalize_thinking_param(
        self, thinking: Any,
    ) -> Optional[Dict[str, str]]:
        """Convert convenience thinking values to the API format."""
        if thinking is None:
            return None
        if isinstance(thinking, bool):
            return {"type": "enabled"} if thinking else {"type": "disabled"}
        if isinstance(thinking, dict):
            return thinking
        if isinstance(thinking, str):
            return {"type": thinking}
        return None

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        thinking_mode = kwargs.pop("thinking", None)

        base_url = self._effective_base_url()
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = self._build_payload(messages, tools, **kwargs)

        thinking_param = self._normalize_thinking_param(thinking_mode)
        if thinking_param is not None:
            payload["thinking"] = thinking_param

        is_reasoner = "reasoner" in self.model or (
            thinking_param and thinking_param.get("type") == "enabled"
        )
        timeout = 300.0 if is_reasoner else 60.0

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    url, headers=headers, json=payload,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise GatewayError(
                    f"DeepSeek API error {exc.response.status_code}: "
                    f"{exc.response.text}",
                    provider=self.name,
                    status_code=exc.response.status_code,
                ) from exc

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        reasoning_content = msg.get("reasoning_content")
        finish_reason = choice.get("finish_reason")

        tool_calls = (
            self._parse_tool_calls(msg["tool_calls"])
            if "tool_calls" in msg
            else []
        )

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            thinking_content=reasoning_content,
        )

        usage = data.get("usage") or {}
        normalized_usage: Dict[str, Any] = {}
        if usage:
            normalized_usage["input_tokens"] = usage.get(
                "prompt_tokens", 0,
            )
            normalized_usage["output_tokens"] = usage.get(
                "completion_tokens", 0,
            )
            normalized_usage["total_tokens"] = usage.get(
                "total_tokens", 0,
            )
            if "completion_tokens_details" in usage:
                details = usage["completion_tokens_details"]
                reasoning_tokens = details.get("reasoning_tokens")
                if reasoning_tokens is not None:
                    normalized_usage["reasoning_tokens"] = reasoning_tokens
            if "prompt_cache_hit_tokens" in usage:
                normalized_usage["cache_hit_tokens"] = usage[
                    "prompt_cache_hit_tokens"
                ]
                normalized_usage["cache_miss_tokens"] = usage.get(
                    "prompt_cache_miss_tokens", 0,
                )
        if finish_reason:
            normalized_usage["finish_reason"] = finish_reason

        return NormalizedResponse(
            messages=[out_msg],
            usage=normalized_usage,
            provider=self.name,
            model=self.model,
            raw=data,
        )

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        thinking_mode = kwargs.pop("thinking", None)

        base_url = self._effective_base_url()
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        payload = self._build_payload(
            messages, tools, stream=True, **kwargs,
        )

        thinking_param = self._normalize_thinking_param(thinking_mode)
        if thinking_param is not None:
            payload["thinking"] = thinking_param

        payload.setdefault("stream_options", {"include_usage": True})

        pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        raw_usage: Dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST", url, headers=headers, json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk_str = line[len("data:"):].strip()
                        if chunk_str == "[DONE]":
                            break

                        chunk = json.loads(chunk_str)
                        if "usage" in chunk and chunk["usage"]:
                            raw_usage = chunk["usage"]

                        choice = (
                            chunk["choices"][0]
                            if chunk.get("choices")
                            else None
                        )
                        if not choice:
                            continue
                        delta = choice.get("delta", {})

                        if delta.get("reasoning_content"):
                            yield StreamEvent(
                                type="chunk",
                                delta=delta["reasoning_content"],
                            )

                        if delta.get("content"):
                            yield StreamEvent(
                                type="chunk", delta=delta["content"],
                            )

                        if "tool_calls" in delta:
                            for tc_delta in delta["tool_calls"]:
                                idx = tc_delta["index"]
                                if idx not in pending_tool_calls:
                                    pending_tool_calls[idx] = {
                                        "id": tc_delta.get("id", ""),
                                        "name": "",
                                        "arguments": "",
                                    }
                                entry = pending_tool_calls[idx]
                                if tc_delta.get("id"):
                                    entry["id"] = tc_delta["id"]
                                fn = tc_delta.get("function", {})
                                if fn.get("name"):
                                    entry["name"] = fn["name"]
                                if fn.get("arguments"):
                                    entry["arguments"] += fn["arguments"]

                        finish = choice.get("finish_reason")
                        if finish == "tool_calls":
                            for _idx in sorted(pending_tool_calls):
                                e = pending_tool_calls[_idx]
                                args = (
                                    json.loads(e["arguments"])
                                    if e["arguments"]
                                    else {}
                                )
                                tc = ToolCall(
                                    id=e["id"],
                                    name=e["name"],
                                    arguments=args,
                                )
                                yield StreamEvent(
                                    type="tool_call", tool_call=tc,
                                )
                            pending_tool_calls.clear()

            except httpx.HTTPStatusError as exc:
                raise GatewayError(
                    f"DeepSeek streaming error "
                    f"{exc.response.status_code}",
                    provider=self.name,
                    status_code=exc.response.status_code,
                ) from exc

        if raw_usage:
            normalized: Dict[str, Any] = {
                "input_tokens": raw_usage.get("prompt_tokens", 0),
                "output_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }
            if "completion_tokens_details" in raw_usage:
                details = raw_usage["completion_tokens_details"]
                rt = details.get("reasoning_tokens")
                if rt is not None:
                    normalized["reasoning_tokens"] = rt
            if "prompt_cache_hit_tokens" in raw_usage:
                normalized["cache_hit_tokens"] = raw_usage[
                    "prompt_cache_hit_tokens"
                ]
                normalized["cache_miss_tokens"] = raw_usage.get(
                    "prompt_cache_miss_tokens", 0,
                )
            yield StreamEvent(type="usage", usage=normalized)
        yield StreamEvent(type="done")
