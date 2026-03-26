from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config.settings import (
    GatewaySettings,
    NamedContextPreset,
    ProviderSettings,
)
from context.registry import ContextRegistry, ContextSource, RegisteredContext
from core.agent_loop import AgentLoop
from core.types import GatewayError, NormalizedMessage, Role
from runtime.router import (
    ProviderConfig,
    create_provider,
    merge_provider_config_overrides,
    resolve_agent_profile,
    resolve_provider_config,
)
from runtime.sse import format_sse
from tools.mcp_http_client import InlineMCPClient
from tools.mcp_loader import load_mcp_tools_from_server
from tools.registry import ToolRegistry, ToolSource


class AppState:
    tool_registry: ToolRegistry = ToolRegistry()
    context_registry: ContextRegistry = ContextRegistry()
    provider_settings: ProviderSettings = ProviderSettings()
    gateway_settings: GatewaySettings = GatewaySettings()


_state = AppState()


async def _lifespan(app: FastAPI):  # noqa: ARG001
    yield


app = FastAPI(title="Unified Agent Gateway", version="0.1.0", lifespan=_lifespan)


def configure(
    tool_registry: Optional[ToolRegistry] = None,
    context_registry: Optional[ContextRegistry] = None,
    provider_settings: Optional[ProviderSettings] = None,
    gateway_settings: Optional[GatewaySettings] = None,
) -> None:
    if tool_registry is not None:
        _state.tool_registry = tool_registry
    if context_registry is not None:
        _state.context_registry = context_registry
    if provider_settings is not None:
        _state.provider_settings = provider_settings
    if gateway_settings is not None:
        _state.gateway_settings = gateway_settings


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class DynamicHTTPTool(BaseModel):
    name: str
    description: str
    json_schema: Dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    url: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 20.0
    argument_mode: Literal["json", "query"] = "json"


class DynamicMCPServer(BaseModel):
    """Inline MCP server wired per-request with full connection details."""

    url: str = Field(..., description="Base URL of the MCP server endpoint")
    namespace: str = Field(..., description="Prefix for tools from this server")
    transport: Literal["sse", "streamable_http"] = Field(
        "streamable_http",
        description="MCP transport: 'streamable_http' (recommended) or 'sse' (legacy)",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Headers forwarded to the MCP server (e.g. Authorization)",
    )
    timeout_seconds: float = Field(30.0)


class DynamicContext(BaseModel):
    name: str
    source: ContextSource = ContextSource.STATIC
    mode: Literal["static", "http"] = "static"
    text: str = ""
    url: Optional[str] = None
    method: Literal["GET", "POST"] = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    payload_template: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 10.0
    required: bool = False
    max_chars: Optional[int] = None


class RuntimeRegistryConfig(BaseModel):
    """Per-request registry overrides.

    Three ways to add tools / contexts for a single call:

    1. **Named presets** — reference keys from ``MCP_SERVERS`` / ``NAMED_CONTEXTS``
       in your ``.env``.  Credentials stay server-side.

       .. code-block:: json

           {"mcp_namespaces": ["search", "files"], "context_names": ["company_info"]}

    2. **Inline MCP servers** — full connection spec in the request body.

       .. code-block:: json

           {"mcp_servers": [{"url": "http://my-mcp/mcp", "namespace": "ext",
                              "headers": {"Authorization": "Bearer sk-..."}}]}

    3. **Inline HTTP tools / static contexts** — arbitrary tools and context
       text defined directly in the request.
    """

    use_global_tools: bool = True
    use_global_contexts: bool = True
    namespace: Optional[str] = None

    # --- Preset references (resolved from settings) ---
    mcp_namespaces: List[str] = Field(
        default_factory=list,
        description="Keys from MCP_SERVERS in .env — connect at request time",
    )
    context_names: List[str] = Field(
        default_factory=list,
        description="Keys from NAMED_CONTEXTS in .env — injected per request",
    )

    # --- Inline full specs ---
    tools: List[DynamicHTTPTool] = Field(default_factory=list)
    mcp_servers: List[DynamicMCPServer] = Field(
        default_factory=list,
        description="MCP servers to connect to inline (full spec in the request)",
    )
    contexts: List[DynamicContext] = Field(default_factory=list)


class ProviderRequestCredentials(BaseModel):
    """Optional per-request LLM credentials.

    Only accepted when ``ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=true`` in
    gateway settings.  Prefer .env / profiles for production; use this for
    bring-your-own-key (BYOK) behind your own auth layer.
    """

    api_key: Optional[str] = None
    base_url: Optional[str] = Field(
        None,
        description="OpenAI-compatible base URL only (ignored for Anthropic/Gemini)",
    )
    model: Optional[str] = None


class AgentQueryRequest(BaseModel):
    input: str
    context: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str = "default"
    profile: str = "default"
    options: Dict[str, Any] = Field(default_factory=dict)
    runtime: Optional[RuntimeRegistryConfig] = None
    provider_credentials: Optional[ProviderRequestCredentials] = None


class AgentQueryResponse(BaseModel):
    output: str
    tool_traces: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Dict[str, int] = Field(default_factory=dict)
    provider: Optional[str] = None
    model: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_format(value: str, variables: Dict[str, Any]) -> str:
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return value.format_map(_SafeDict(**variables))


def _render_template(obj: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _render_template(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_render_template(v, variables) for v in obj]
    if isinstance(obj, str):
        return _safe_format(obj, variables)
    return obj


def _context_source_from_string(value: str) -> ContextSource:
    try:
        return ContextSource(value)
    except ValueError:
        return ContextSource.STATIC


def _register_named_context_preset(
    contexts: ContextRegistry,
    name: str,
    preset: NamedContextPreset,
) -> Optional[str]:
    """Register a named preset into *contexts*.  Returns a warning string on failure."""
    from runtime.bootstrap import _register_named_context

    try:
        _register_named_context(contexts, name, preset)
        return None
    except Exception as exc:
        return f"Named context '{name}' could not be registered: {exc}"


def _effective_runtime(
    body: AgentQueryRequest,
) -> RuntimeRegistryConfig:
    """Merge profile-level preset names into the request's RuntimeRegistryConfig.

    If the resolved agent profile defines ``mcp_namespaces`` or
    ``context_names``, they are prepended to whatever the caller supplied so
    profile-level presets are always active without the caller having to repeat
    them every request.
    """
    profile = resolve_agent_profile(
        _state.gateway_settings,
        agent_id=body.agent_id,
        profile=body.profile,
    )
    base = body.runtime or RuntimeRegistryConfig()

    # Merge profile defaults — profile presets come first, caller overrides last
    merged_mcp = list(dict.fromkeys(profile.mcp_namespaces + base.mcp_namespaces))
    merged_ctx = list(dict.fromkeys(profile.context_names + base.context_names))

    return base.model_copy(update={"mcp_namespaces": merged_mcp, "context_names": merged_ctx})


async def _compose_registries(
    runtime_cfg: RuntimeRegistryConfig,
    stack: AsyncExitStack,
    warnings: List[str],
) -> Tuple[ToolRegistry, ContextRegistry]:
    """Build request-scoped ToolRegistry and ContextRegistry.

    *warnings* is mutated in-place with any non-fatal issues encountered.
    MCP sessions are entered into *stack* so they stay alive for the request.
    """
    has_dynamic = bool(
        runtime_cfg.tools
        or runtime_cfg.mcp_servers
        or runtime_cfg.mcp_namespaces
        or runtime_cfg.contexts
        or runtime_cfg.context_names
    )
    if has_dynamic and not _state.gateway_settings.ALLOW_DYNAMIC_RUNTIME_REGISTRATION:
        raise HTTPException(
            status_code=403,
            detail="Dynamic runtime registration is disabled by gateway settings.",
        )

    tools = _state.tool_registry.copy() if runtime_cfg.use_global_tools else ToolRegistry()
    contexts = (
        _state.context_registry.copy() if runtime_cfg.use_global_contexts else ContextRegistry()
    )

    ns_prefix = f"{runtime_cfg.namespace}." if runtime_cfg.namespace else ""

    # --- Named MCP preset references (from MCP_SERVERS in settings) ---
    for ns_name in runtime_cfg.mcp_namespaces:
        preset = _state.gateway_settings.MCP_SERVERS.get(ns_name)
        if preset is None:
            warnings.append(
                f"MCP namespace '{ns_name}' not found in MCP_SERVERS settings — skipped"
            )
            continue
        qualified_ns = f"{ns_prefix}{ns_name}"
        client = InlineMCPClient(
            url=preset.url,
            transport=preset.transport,
            headers=preset.headers,
            timeout=preset.timeout_seconds,
        )
        try:
            connected = await stack.enter_async_context(client)
            count = await load_mcp_tools_from_server(
                registry=tools, client=connected, namespace=qualified_ns
            )
            if count == 0:
                warnings.append(f"MCP preset '{ns_name}' at '{preset.url}' reported zero tools")
        except Exception as exc:
            warnings.append(f"MCP preset '{ns_name}' at '{preset.url}' failed to connect: {exc}")

    # --- Inline MCP servers (full spec in request) ---
    for spec in runtime_cfg.mcp_servers:
        qualified_ns = f"{ns_prefix}{spec.namespace}"
        client = InlineMCPClient(
            url=spec.url,
            transport=spec.transport,
            headers=spec.headers,
            timeout=spec.timeout_seconds,
        )
        try:
            connected = await stack.enter_async_context(client)
            count = await load_mcp_tools_from_server(
                registry=tools, client=connected, namespace=qualified_ns
            )
            if count == 0:
                warnings.append(
                    f"Inline MCP server at '{spec.url}' (namespace '{qualified_ns}') "
                    "reported zero tools"
                )
        except Exception as exc:
            warnings.append(
                f"Inline MCP server at '{spec.url}' (namespace '{qualified_ns}') "
                f"failed to connect: {exc}"
            )

    # --- Inline HTTP tools ---
    for spec in runtime_cfg.tools:
        tool_name = f"{ns_prefix}{spec.name}"

        async def handler(_spec: DynamicHTTPTool = spec, **kwargs: Any) -> Any:
            async with httpx.AsyncClient(timeout=_spec.timeout_seconds) as http:
                if _spec.argument_mode == "query":
                    resp = await http.request(
                        _spec.method, _spec.url, params=kwargs, headers=_spec.headers
                    )
                else:
                    resp = await http.request(
                        _spec.method, _spec.url, json=kwargs, headers=_spec.headers
                    )
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "application/json" in ctype:
                    return resp.json()
                return resp.text

        tools.register(
            name=tool_name,
            description=spec.description,
            json_schema=spec.json_schema,
            source=ToolSource.HTTP,
            handler=handler,
            metadata={"dynamic": True, "url": spec.url},
        )

    # --- Named context preset references (from NAMED_CONTEXTS in settings) ---
    for ctx_name in runtime_cfg.context_names:
        preset = _state.gateway_settings.NAMED_CONTEXTS.get(ctx_name)
        if preset is None:
            warnings.append(
                f"Context name '{ctx_name}' not found in NAMED_CONTEXTS settings — skipped"
            )
            continue
        qualified_name = f"{ns_prefix}{ctx_name}"
        warn = _register_named_context_preset(contexts, qualified_name, preset)
        if warn:
            warnings.append(warn)

    # --- Inline dynamic contexts ---
    for spec in runtime_cfg.contexts:
        ctx_name = f"{ns_prefix}{spec.name}"

        if spec.mode == "static":

            async def fetch(_spec: DynamicContext = spec, **kwargs: Any) -> str:
                return _safe_format(_spec.text, {k: str(v) for k, v in kwargs.items()})
        else:
            if not spec.url:
                warnings.append(f"Context '{ctx_name}' skipped: mode=http but no url provided")
                continue

            async def fetch(_spec: DynamicContext = spec, **kwargs: Any) -> str:
                variables = {k: str(v) for k, v in kwargs.items()}
                payload = _render_template(
                    _spec.payload_template or {"input": "{input}"},
                    variables,
                )
                async with httpx.AsyncClient(timeout=_spec.timeout_seconds) as http:
                    if _spec.method == "GET":
                        resp = await http.get(
                            _spec.url or "", params=payload, headers=_spec.headers
                        )
                    else:
                        resp = await http.post(_spec.url or "", json=payload, headers=_spec.headers)
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if "application/json" in ctype:
                        data = resp.json()
                        return str(data.get("context", data))
                    return resp.text

        contexts.register(
            RegisteredContext(
                name=ctx_name,
                source=_context_source_from_string(spec.source.value),
                fetch=fetch,
                required=spec.required,
                max_chars=spec.max_chars,
                metadata={"dynamic": True, "mode": spec.mode},
            )
        )

    return tools, contexts


def _resolve_provider_config_for_request(body: AgentQueryRequest) -> ProviderConfig:
    """Resolve provider config, applying optional per-request credentials."""
    cfg = resolve_provider_config(
        _state.provider_settings,
        _state.gateway_settings,
        agent_id=body.agent_id,
        profile=body.profile,
    )
    creds = body.provider_credentials
    if creds is None:
        return cfg
    has_any = creds.api_key is not None or creds.model is not None or creds.base_url is not None
    if not has_any:
        return cfg
    if not _state.gateway_settings.ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS:
        raise HTTPException(
            status_code=403,
            detail="Per-request provider credentials are disabled by gateway settings.",
        )
    return merge_provider_config_overrides(
        cfg,
        api_key=creds.api_key,
        model=creds.model,
        base_url=creds.base_url,
    )


async def _inject_context_messages(
    messages: List[NormalizedMessage],
    contexts: ContextRegistry,
    context_kwargs: Dict[str, Any],
) -> List[NormalizedMessage]:
    ctx_map = await contexts.load_all(**context_kwargs)
    if not ctx_map:
        return messages

    ctx_text = "\n\n".join(f"[{k}]\n{v}" for k, v in ctx_map.items() if v)
    if not ctx_text:
        return messages

    context_msg = NormalizedMessage(
        role=Role.SYSTEM,
        content=f"Additional context:\n{ctx_text}",
    )
    result = list(messages)
    sys_idx = next((i for i, m in enumerate(result) if m.role == Role.SYSTEM), None)
    if sys_idx is not None:
        result.insert(sys_idx + 1, context_msg)
    else:
        result.insert(0, context_msg)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/agent-query", response_model=AgentQueryResponse)
async def agent_query(body: AgentQueryRequest):
    cfg = _resolve_provider_config_for_request(body)
    provider = create_provider(cfg)
    runtime_cfg = _effective_runtime(body)

    async with AsyncExitStack() as stack:
        warnings: List[str] = []
        tools, contexts = await _compose_registries(runtime_cfg, stack=stack, warnings=warnings)

        loop = AgentLoop(
            provider=provider,
            tools=tools,
            contexts=contexts,
            max_tool_hops=_state.gateway_settings.MAX_TOOL_HOPS,
            tool_timeout=_state.gateway_settings.TOOL_TIMEOUT_SECONDS,
        )

        messages = [NormalizedMessage(role=Role.USER, content=body.input)]
        if body.context.get("system_prompt"):
            messages.insert(
                0,
                NormalizedMessage(role=Role.SYSTEM, content=body.context["system_prompt"]),
            )

        try:
            result = await loop.run_conversation(
                messages,
                context_kwargs={"input": body.input, **body.context},
                **body.options,
            )
        except GatewayError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=AgentQueryResponse(
                    output="",
                    errors=[str(exc)],
                    provider=exc.provider,
                    warnings=warnings,
                ).model_dump(),
            )

    answer = result.messages[-1].content if result.messages else ""
    return AgentQueryResponse(
        output=answer,
        usage=result.usage,
        provider=result.provider,
        model=result.model,
        warnings=warnings,
    )


@app.post("/agent-query/stream")
async def agent_query_stream(request: Request, body: AgentQueryRequest):
    cfg = _resolve_provider_config_for_request(body)
    provider = create_provider(cfg)
    runtime_cfg = _effective_runtime(body)

    # Open MCP sessions before the streaming generator starts; keep them
    # alive via the stack until the generator finishes.
    stack = AsyncExitStack()
    try:
        await stack.__aenter__()
        warnings: List[str] = []
        tools, contexts = await _compose_registries(runtime_cfg, stack=stack, warnings=warnings)

        messages: List[NormalizedMessage] = []
        if body.context.get("system_prompt"):
            messages.append(
                NormalizedMessage(role=Role.SYSTEM, content=body.context["system_prompt"])
            )
        messages.append(NormalizedMessage(role=Role.USER, content=body.input))
        messages = await _inject_context_messages(
            messages,
            contexts,
            context_kwargs={"input": body.input, **body.context},
        )
    except Exception:
        await stack.aclose()
        raise

    async def event_stream():
        try:
            if warnings:
                yield format_sse("warning", {"warnings": warnings})
            async for event in provider.stream(
                messages=messages,
                tools=tools.list_for_provider() or None,
            ):
                if await request.is_disconnected():
                    break
                yield format_sse(event.type, event.to_dict())
        except GatewayError as exc:
            yield format_sse("error", {"message": str(exc)})
        finally:
            await stack.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
