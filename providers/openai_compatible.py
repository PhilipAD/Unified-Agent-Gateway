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
from providers.base import BaseProvider

ROLE_MAP = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}


class OpenAICompatibleProvider(BaseProvider):
    name = "openai_compatible"

    def _msg_to_api(self, m: NormalizedMessage) -> Dict[str, Any]:
        """Convert a NormalizedMessage to the OpenAI chat-completions message shape."""
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
    def _parse_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        result: List[ToolCall] = []
        for tc in raw_calls:
            fn = tc["function"]
            args_raw = fn.get("arguments") or "{}"
            arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            result.append(ToolCall(id=tc["id"], name=fn["name"], arguments=arguments))
        return result

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = self._build_payload(messages, tools, **kwargs)

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise GatewayError(
                    f"OpenAI API error {exc.response.status_code}: {exc.response.text}",
                    provider=self.name,
                    status_code=exc.response.status_code,
                ) from exc

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        tool_calls = self._parse_tool_calls(msg["tool_calls"]) if "tool_calls" in msg else []

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        usage = data.get("usage") or {}
        return NormalizedResponse(
            messages=[out_msg],
            usage=usage,
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
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        payload = self._build_payload(messages, tools, stream=True, **kwargs)

        # Accumulators for streamed tool calls (keyed by index)
        pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        full_content = ""
        usage: Dict[str, int] = {}

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk_str = line[len("data:") :].strip()
                        if chunk_str == "[DONE]":
                            break

                        chunk = json.loads(chunk_str)
                        if "usage" in chunk and chunk["usage"]:
                            usage = chunk["usage"]

                        choice = chunk["choices"][0] if chunk.get("choices") else None
                        if not choice:
                            continue
                        delta = choice.get("delta", {})

                        if delta.get("content"):
                            full_content += delta["content"]
                            yield StreamEvent(type="chunk", delta=delta["content"])

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
                                args = json.loads(e["arguments"]) if e["arguments"] else {}
                                tc = ToolCall(id=e["id"], name=e["name"], arguments=args)
                                yield StreamEvent(type="tool_call", tool_call=tc)
                            pending_tool_calls.clear()

            except httpx.HTTPStatusError as exc:
                raise GatewayError(
                    f"OpenAI streaming error {exc.response.status_code}",
                    provider=self.name,
                    status_code=exc.response.status_code,
                ) from exc

        if usage:
            yield StreamEvent(type="usage", usage=usage)
        yield StreamEvent(type="done")
