from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import anyio
from google import genai
from google.genai import types as genai_types

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

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    name = "gemini"

    def _client(self) -> genai.Client:
        return genai.Client(api_key=self.api_key)

    def _build_contents(self, messages: List[NormalizedMessage]) -> List[genai_types.Content]:
        contents: List[genai_types.Content] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                continue

            if m.role == Role.TOOL:
                # Build a function response part with matching id when available
                parts = [
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=m.name or "",
                            response={"result": m.content},
                            **({"id": m.tool_call_id} if m.tool_call_id else {}),
                        )
                    )
                ]
                contents.append(genai_types.Content(role="user", parts=parts))
                continue

            if m.role == Role.ASSISTANT and m.tool_calls:
                parts = []
                if m.content:
                    parts.append(genai_types.Part(text=m.content))
                for tc in m.tool_calls:
                    parts.append(
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name=tc.name,
                                args=tc.arguments,
                                **({"id": tc.id} if tc.id else {}),
                            )
                        )
                    )
                contents.append(genai_types.Content(role="model", parts=parts))
                continue

            role = "user" if m.role == Role.USER else "model"
            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=m.content)],
                )
            )
        return contents

    def _build_tools(
        self, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[genai_types.Tool]]:
        if not tools:
            return None
        declarations = []
        for t in tools:
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.json_schema,
                )
            )
        return [genai_types.Tool(function_declarations=declarations)]

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        client = self._client()
        contents = self._build_contents(messages)
        tools_cfg = self._build_tools(tools)

        system_msg = next((m for m in messages if m.role == Role.SYSTEM), None)
        config_kwargs: Dict[str, Any] = {}
        if tools_cfg:
            config_kwargs["tools"] = tools_cfg
        if system_msg:
            config_kwargs["system_instruction"] = system_msg.content
        config_kwargs.update(kwargs)
        config = genai_types.GenerateContentConfig(**config_kwargs)

        try:
            resp = await anyio.to_thread.run_sync(
                lambda: client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            )
        except Exception as exc:
            raise GatewayError(f"Gemini API error: {exc}", provider=self.name) from exc

        content_text = resp.text or ""
        tool_calls: List[ToolCall] = []

        if resp.candidates:
            cand = resp.candidates[0]
            for part in cand.content.parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    tool_calls.append(
                        ToolCall(
                            id=getattr(fc, "id", "") or "",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT, content=content_text, tool_calls=tool_calls
        )

        usage: Dict[str, int] = {}
        raw_usage = getattr(resp, "usage_metadata", None)
        if raw_usage:
            usage = {
                "input_tokens": getattr(raw_usage, "prompt_token_count", 0),
                "output_tokens": getattr(raw_usage, "candidates_token_count", 0),
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
        client = self._client()
        contents = self._build_contents(messages)
        tools_cfg = self._build_tools(tools)

        system_msg = next((m for m in messages if m.role == Role.SYSTEM), None)
        config_kwargs: Dict[str, Any] = {}
        if tools_cfg:
            config_kwargs["tools"] = tools_cfg
        if system_msg:
            config_kwargs["system_instruction"] = system_msg.content
        config_kwargs.update(kwargs)
        config = genai_types.GenerateContentConfig(**config_kwargs)

        try:
            stream_iter = await anyio.to_thread.run_sync(
                lambda: client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                )
            )
        except Exception as exc:
            raise GatewayError(f"Gemini streaming error: {exc}", provider=self.name) from exc

        def _next_chunk(it):
            try:
                return next(it)
            except StopIteration:
                return None

        while True:
            chunk = await anyio.to_thread.run_sync(lambda: _next_chunk(stream_iter))
            if chunk is None:
                break

            if chunk.text:
                yield StreamEvent(type="chunk", delta=chunk.text)

            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc:
                        tc = ToolCall(
                            id=getattr(fc, "id", "") or "",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                        yield StreamEvent(type="tool_call", tool_call=tc)

        yield StreamEvent(type="done")
