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

            if m.role == Role.USER and isinstance(m.content, list):
                parts = _convert_user_content_parts(m.content)
                contents.append(genai_types.Content(role="user", parts=parts))
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
        self,
        tools: Optional[List[ToolDefinition]],
        built_in_tools: Optional[List[str]] = None,
        built_in_tool_configs: Optional[List[Dict[str, Any]]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[List[genai_types.Tool]]:
        result: List[genai_types.Tool] = []

        if tools:
            declarations = []
            for t in tools:
                declarations.append(
                    genai_types.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=t.json_schema,
                    )
                )
            result.append(genai_types.Tool(function_declarations=declarations))

        for bt_name in (built_in_tools or []):
            if bt_name == "code_execution":
                result.append(
                    genai_types.Tool(code_execution=genai_types.ToolCodeExecution())
                )
            elif bt_name == "google_search":
                search_kwargs: Dict[str, Any] = {}
                if built_in_tool_configs:
                    for cfg in built_in_tool_configs:
                        if cfg.get("type") == "google_search":
                            if "time_range_filter" in cfg:
                                search_kwargs["time_range_filter"] = cfg["time_range_filter"]
                            if "exclude_domains" in cfg:
                                search_kwargs["exclude_domains"] = cfg["exclude_domains"]
                result.append(
                    genai_types.Tool(google_search=genai_types.GoogleSearch(**search_kwargs))
                )
            elif bt_name == "url_context":
                result.append(
                    genai_types.Tool(url_context=genai_types.UrlContext())
                )
            elif bt_name == "google_maps":
                result.append(
                    genai_types.Tool(google_maps=genai_types.GoogleMaps())
                )
            elif bt_name == "computer_use":
                cu_kwargs: Dict[str, Any] = {}
                if built_in_tool_configs:
                    for cfg in built_in_tool_configs:
                        if cfg.get("type") == "computer_use":
                            if "environment" in cfg:
                                cu_kwargs["environment"] = cfg["environment"]
                result.append(
                    genai_types.Tool(computer_use=genai_types.ComputerUse(**cu_kwargs))
                )
            elif bt_name == "file_search":
                fs_kwargs: Dict[str, Any] = {}
                if built_in_tool_configs:
                    for cfg in built_in_tool_configs:
                        if cfg.get("type") == "file_search":
                            if "file_search_store_names" in cfg:
                                names = cfg["file_search_store_names"]
                                fs_kwargs["file_search_store_names"] = names
                            if "top_k" in cfg:
                                fs_kwargs["top_k"] = cfg["top_k"]
                result.append(
                    genai_types.Tool(file_search=genai_types.FileSearch(**fs_kwargs))
                )

        if mcp_servers:
            mcp_list = []
            for mcp in mcp_servers:
                transport_kwargs: Dict[str, Any] = {"url": mcp["server_url"]}
                if "headers" in mcp:
                    transport_kwargs["headers"] = mcp["headers"]
                if "timeout" in mcp:
                    transport_kwargs["timeout"] = mcp["timeout"]
                mcp_list.append(
                    genai_types.McpServer(
                        name=mcp.get("server_label", "mcp"),
                        streamable_http_transport=genai_types.StreamableHttpTransport(
                            **transport_kwargs
                        ),
                    )
                )
            result.append(genai_types.Tool(mcp_servers=mcp_list))

        return result or None

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        client = self._client()
        contents = self._build_contents(messages)

        built_in_tools = kwargs.pop("built_in_tools", None)
        built_in_tool_configs = kwargs.pop("built_in_tool_configs", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        thinking_level = kwargs.pop("thinking_level", None)
        tool_config = kwargs.pop("tool_config", None)
        safety_settings = kwargs.pop("safety_settings", None)
        response_schema = kwargs.pop("response_schema", None)
        response_mime_type = kwargs.pop("response_mime_type", None)

        tools_cfg = self._build_tools(tools, built_in_tools, built_in_tool_configs, mcp_servers)

        system_msg = next((m for m in messages if m.role == Role.SYSTEM), None)
        config_kwargs: Dict[str, Any] = {}
        if tools_cfg:
            config_kwargs["tools"] = tools_cfg
        if system_msg:
            config_kwargs["system_instruction"] = system_msg.content

        if thinking_budget is not None or thinking_level is not None:
            tc_kwargs: Dict[str, Any] = {}
            if thinking_budget is not None:
                tc_kwargs["include_thoughts"] = True
                tc_kwargs["thinking_budget"] = thinking_budget
            if thinking_level is not None:
                tc_kwargs["thinking_level"] = thinking_level
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(**tc_kwargs)

        if tool_config is not None:
            if isinstance(tool_config, dict):
                fc_config_kwargs: Dict[str, Any] = {}
                if "mode" in tool_config:
                    fc_config_kwargs["mode"] = tool_config["mode"]
                if "allowed_function_names" in tool_config:
                    names = tool_config["allowed_function_names"]
                    fc_config_kwargs["allowed_function_names"] = names
                config_kwargs["tool_config"] = genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(**fc_config_kwargs)
                )
            else:
                config_kwargs["tool_config"] = tool_config

        if safety_settings is not None:
            config_kwargs["safety_settings"] = safety_settings

        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
            if not response_mime_type:
                config_kwargs["response_mime_type"] = "application/json"

        if response_mime_type is not None:
            config_kwargs["response_mime_type"] = response_mime_type

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

        content_text = ""
        thinking_content = ""
        tool_calls: List[ToolCall] = []
        code_execution_result = ""

        if resp.candidates:
            cand = resp.candidates[0]
            for part in cand.content.parts:
                if getattr(part, "thought", False) and part.text:
                    thinking_content += part.text
                    continue

                fc = getattr(part, "function_call", None)
                if fc:
                    tool_calls.append(
                        ToolCall(
                            id=getattr(fc, "id", "") or "",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )
                    continue

                exec_result = getattr(part, "executable_code", None)
                if exec_result:
                    code_execution_result += getattr(exec_result, "code", "")
                    continue

                code_output = getattr(part, "code_execution_result", None)
                if code_output:
                    code_execution_result += getattr(code_output, "output", "")
                    continue

                if part.text:
                    content_text += part.text

        if not content_text:
            content_text = getattr(resp, "text", None) or ""

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content_text,
            tool_calls=tool_calls,
            thinking_content=thinking_content or None,
        )

        usage: Dict[str, Any] = {}
        raw_usage = getattr(resp, "usage_metadata", None)
        if raw_usage:
            usage["input_tokens"] = getattr(raw_usage, "prompt_token_count", 0)
            usage["output_tokens"] = getattr(raw_usage, "candidates_token_count", 0)
            total = getattr(raw_usage, "total_token_count", None)
            if total:
                usage["total_tokens"] = total
            thoughts_tokens = getattr(raw_usage, "thoughts_token_count", None)
            if thoughts_tokens:
                usage["thoughts_tokens"] = thoughts_tokens
            cached_content_tokens = getattr(raw_usage, "cached_content_token_count", None)
            if cached_content_tokens:
                usage["cached_tokens"] = cached_content_tokens

        grounding_metadata = _extract_grounding_metadata(resp)
        if grounding_metadata:
            usage["grounding_metadata"] = grounding_metadata

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

        built_in_tools = kwargs.pop("built_in_tools", None)
        built_in_tool_configs = kwargs.pop("built_in_tool_configs", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        thinking_level = kwargs.pop("thinking_level", None)
        tool_config = kwargs.pop("tool_config", None)
        safety_settings = kwargs.pop("safety_settings", None)
        response_schema = kwargs.pop("response_schema", None)
        response_mime_type = kwargs.pop("response_mime_type", None)

        tools_cfg = self._build_tools(tools, built_in_tools, built_in_tool_configs, mcp_servers)

        system_msg = next((m for m in messages if m.role == Role.SYSTEM), None)
        config_kwargs: Dict[str, Any] = {}
        if tools_cfg:
            config_kwargs["tools"] = tools_cfg
        if system_msg:
            config_kwargs["system_instruction"] = system_msg.content

        if thinking_budget is not None or thinking_level is not None:
            tc_kwargs: Dict[str, Any] = {}
            if thinking_budget is not None:
                tc_kwargs["include_thoughts"] = True
                tc_kwargs["thinking_budget"] = thinking_budget
            if thinking_level is not None:
                tc_kwargs["thinking_level"] = thinking_level
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(**tc_kwargs)

        if tool_config is not None:
            if isinstance(tool_config, dict):
                fc_config_kwargs: Dict[str, Any] = {}
                if "mode" in tool_config:
                    fc_config_kwargs["mode"] = tool_config["mode"]
                if "allowed_function_names" in tool_config:
                    names = tool_config["allowed_function_names"]
                    fc_config_kwargs["allowed_function_names"] = names
                config_kwargs["tool_config"] = genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(**fc_config_kwargs)
                )
            else:
                config_kwargs["tool_config"] = tool_config

        if safety_settings is not None:
            config_kwargs["safety_settings"] = safety_settings

        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
            if not response_mime_type:
                config_kwargs["response_mime_type"] = "application/json"

        if response_mime_type is not None:
            config_kwargs["response_mime_type"] = response_mime_type

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

            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False) and part.text:
                        yield StreamEvent(type="chunk", delta=part.text)
                        continue

                    fc = getattr(part, "function_call", None)
                    if fc:
                        tc = ToolCall(
                            id=getattr(fc, "id", "") or "",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                        yield StreamEvent(type="tool_call", tool_call=tc)
                        continue

                    if part.text:
                        yield StreamEvent(type="chunk", delta=part.text)
            elif chunk.text:
                yield StreamEvent(type="chunk", delta=chunk.text)

            raw_usage = getattr(chunk, "usage_metadata", None)
            if raw_usage and getattr(raw_usage, "total_token_count", 0):
                usage: Dict[str, Any] = {
                    "input_tokens": getattr(raw_usage, "prompt_token_count", 0),
                    "output_tokens": getattr(raw_usage, "candidates_token_count", 0),
                    "total_tokens": getattr(raw_usage, "total_token_count", 0),
                }
                thoughts = getattr(raw_usage, "thoughts_token_count", None)
                if thoughts:
                    usage["thoughts_tokens"] = thoughts
                yield StreamEvent(type="usage", usage=usage)

        yield StreamEvent(type="done")


def _convert_user_content_parts(parts: List[Any]) -> List[genai_types.Part]:
    """Convert multi-modal content parts to Gemini Part objects."""
    result: List[genai_types.Part] = []
    for part in parts:
        if isinstance(part, str):
            result.append(genai_types.Part(text=part))
        elif isinstance(part, dict):
            ptype = part.get("type", "text")
            if ptype == "text":
                result.append(genai_types.Part(text=part.get("text", "")))
            elif ptype == "image_url":
                url_info = part.get("image_url", {})
                url = url_info.get("url", "") if isinstance(url_info, dict) else url_info
                if url.startswith("data:"):
                    media_type, _, encoded = url.partition(";base64,")
                    media_type = media_type.replace("data:", "")
                    import base64
                    result.append(genai_types.Part(
                        inline_data=genai_types.Blob(
                            data=base64.b64decode(encoded),
                            mime_type=media_type,
                        )
                    ))
                else:
                    result.append(genai_types.Part(
                        file_data=genai_types.FileData(file_uri=url, mime_type="image/*")
                    ))
            elif ptype == "file":
                file_info = part.get("file", {})
                result.append(genai_types.Part(
                    file_data=genai_types.FileData(
                        file_uri=file_info.get("uri", ""),
                        mime_type=file_info.get("mime_type", ""),
                    )
                ))
    return result


def _extract_grounding_metadata(resp: Any) -> Optional[Dict[str, Any]]:
    """Extract grounding metadata from response candidates."""
    if not resp.candidates:
        return None
    cand = resp.candidates[0]
    gm = getattr(cand, "grounding_metadata", None)
    if not gm:
        return None

    result: Dict[str, Any] = {}
    queries = getattr(gm, "web_search_queries", None)
    if queries:
        result["web_search_queries"] = list(queries)

    chunks = getattr(gm, "grounding_chunks", None)
    if chunks:
        chunk_list = []
        for gc in chunks:
            web = getattr(gc, "web", None)
            if web:
                chunk_list.append({
                    "uri": getattr(web, "uri", ""),
                    "title": getattr(web, "title", ""),
                })
        if chunk_list:
            result["grounding_chunks"] = chunk_list

    supports = getattr(gm, "grounding_supports", None)
    if supports:
        support_list = []
        for gs in supports:
            segment = getattr(gs, "segment", None)
            support_list.append({
                "text": getattr(segment, "text", "") if segment else "",
                "confidence_scores": list(getattr(gs, "confidence_scores", [])),
                "grounding_chunk_indices": list(getattr(gs, "grounding_chunk_indices", [])),
            })
        if support_list:
            result["grounding_supports"] = support_list

    return result or None
