from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config.settings import (
    AgentHarnessSettings,
    GatewaySettings,
    NamedContextPreset,
    ProviderSettings,
)
from context.registry import ContextRegistry, ContextSource, RegisteredContext
from core.agent_loop import AgentLoop
from core.types import GatewayError, NormalizedMessage, Role
from providers.cursor_cloud_agent import CursorCloudAgentProvider, verify_cursor_webhook_signature
from runtime.cursor_webhook import signal_cursor_agent_event
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

logger = logging.getLogger(__name__)


async def _lifespan(app: FastAPI):  # noqa: ARG001
    yield


lifespan = _lifespan

app = FastAPI(title="Unified Agents SDK", version="0.1.0", lifespan=_lifespan)


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
    """Per-request context source specification.

    Modes
    -----
    static       : inline text (optionally with ``{variable}`` placeholders).
    http         : fetch text from an HTTP endpoint at call time.
    md_hierarchy : walk *cwd* + ancestor dirs (optionally system/user dirs too)
                   looking for files whose name matches any entry in *filenames*.
                   Mirrors what the harness bootstraps globally, but fully
                   configurable per request.
    md_files     : read an explicit list of *paths* (absolute or relative to cwd).
    md_glob      : collect every file matching *glob_pattern* under each directory
                   listed in *glob_dirs*.
    """

    name: str
    source: ContextSource = ContextSource.STATIC
    mode: Literal["static", "http", "md_hierarchy", "md_files", "md_glob"] = "static"

    # --- static / http shared ---
    text: str = ""
    url: Optional[str] = None
    method: Literal["GET", "POST"] = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    payload_template: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 10.0

    # --- md_hierarchy ---
    filenames: List[str] = Field(
        default_factory=list,
        description="File names to search for (e.g. ['AGENTS.md', 'MY_RULES.md'])",
    )
    cwd: str = Field(".", description="Base directory for hierarchy walk")
    system_dirs: List[str] = Field(
        default_factory=list,
        description="Extra system-level directories to scan before cwd walk",
    )
    user_dirs: List[str] = Field(
        default_factory=list,
        description="Extra user-level directories to scan (e.g. ['~/.myagent'])",
    )
    stop_at_git_root: bool = True
    resolve_imports: bool = Field(
        True,
        description="Inline @file.md import directives (Gemini-style)",
    )

    # --- md_files ---
    paths: List[str] = Field(
        default_factory=list,
        description="Explicit file paths to read (absolute or relative to cwd)",
    )

    # --- md_glob ---
    glob_dirs: List[str] = Field(
        default_factory=list,
        description="Directories to scan with glob_pattern",
    )
    glob_pattern: str = Field("*.md", description="Glob pattern applied inside each glob_dir")

    # --- shared ---
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

        elif spec.mode == "http":
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

        elif spec.mode == "md_hierarchy":
            if not spec.filenames:
                warnings.append(
                    f"Context '{ctx_name}' skipped: mode=md_hierarchy requires filenames list"
                )
                continue

            async def fetch(_spec: DynamicContext = spec, **kwargs: Any) -> str:
                from context.md_hierarchy import collect_md_hierarchy

                cwd = str(kwargs.get("cwd") or _spec.cwd or ".")
                return collect_md_hierarchy(
                    cwd=cwd,
                    filenames=_spec.filenames,
                    system_dirs=_spec.system_dirs,
                    user_dirs=_spec.user_dirs,
                    stop_at_git_root=_spec.stop_at_git_root,
                    resolve_imports=_spec.resolve_imports,
                )

        elif spec.mode == "md_files":
            if not spec.paths:
                warnings.append(
                    f"Context '{ctx_name}' skipped: mode=md_files requires paths list"
                )
                continue

            async def fetch(_spec: DynamicContext = spec, **kwargs: Any) -> str:
                import os
                from pathlib import Path

                from context.md_hierarchy import _read_file_safe  # noqa: PLC2701

                base = Path(str(kwargs.get("cwd") or _spec.cwd or ".")).resolve()
                parts: List[str] = []
                for raw_path in _spec.paths:
                    p = Path(os.path.expandvars(os.path.expanduser(raw_path)))
                    if not p.is_absolute():
                        p = base / p
                    p = p.resolve()
                    text = _read_file_safe(p)
                    if text is not None:
                        parts.append(f"--- file: {p} ---\n{text.rstrip()}")
                    else:
                        logger.warning("md_files context '%s': cannot read %s", _spec.name, p)
                return "\n\n".join(parts)

        elif spec.mode == "md_glob":
            if not spec.glob_dirs:
                warnings.append(
                    f"Context '{ctx_name}' skipped: mode=md_glob requires glob_dirs list"
                )
                continue

            async def fetch(_spec: DynamicContext = spec, **kwargs: Any) -> str:
                import os
                from pathlib import Path

                from context.md_hierarchy import collect_glob_files_in_dirs

                base = Path(str(kwargs.get("cwd") or _spec.cwd or ".")).resolve()
                dirs = [
                    Path(os.path.expandvars(os.path.expanduser(d)))
                    if Path(os.path.expanduser(d)).is_absolute()
                    else base / d
                    for d in _spec.glob_dirs
                ]
                return collect_glob_files_in_dirs(
                    dirs,
                    _spec.glob_pattern,
                    resolve_imports=_spec.resolve_imports,
                )

        else:
            warnings.append(f"Context '{ctx_name}' skipped: unknown mode '{spec.mode}'")
            continue

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


def _merged_run_options(body: AgentQueryRequest) -> Dict[str, Any]:
    profile = resolve_agent_profile(
        _state.gateway_settings, agent_id=body.agent_id, profile=body.profile
    )
    return {**dict(profile.extra), **body.options}


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
                **_merged_run_options(body),
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
                **_merged_run_options(body),
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


# ---------------------------------------------------------------------------
# Cursor Cloud Agent webhook + proxies
# ---------------------------------------------------------------------------


class CursorWebhookPayload(BaseModel):
    event: Optional[str] = None
    timestamp: Optional[str] = None
    id: Optional[str] = None
    status: Optional[str] = None
    source: Dict[str, Any] = Field(default_factory=dict)
    target: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None


@app.post("/webhooks/cursor")
async def cursor_webhook_handler(request: Request, _background: BackgroundTasks):
    raw = await request.body()
    harness = AgentHarnessSettings()
    secret = harness.CURSOR_WEBHOOK_SECRET or ""
    sig = (
        request.headers.get("X-Webhook-Signature")
        or request.headers.get("x-webhook-signature")
        or ""
    )
    if secret:
        if not verify_cursor_webhook_signature(raw, sig, secret):
            raise HTTPException(status_code=401, detail="invalid webhook signature")
    try:
        data = json.loads(raw.decode("utf-8")) if raw else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid json body")
    agent_id = data.get("id") or data.get("agentId")
    if agent_id:
        signal_cursor_agent_event(str(agent_id))
    try:
        received = CursorWebhookPayload.model_validate(data).model_dump()
    except Exception:
        received = data
    return {"ok": True, "received": received}


def _cursor_api() -> CursorCloudAgentProvider:
    ps = _state.provider_settings
    return CursorCloudAgentProvider(
        api_key=ps.CURSOR_API_KEY or "",
        model=ps.DEFAULT_CURSOR_MODEL,
    )


@app.get("/cursor-agent/{agent_id}/status")
async def cursor_agent_status_proxy(agent_id: str):
    try:
        return await _cursor_api().get_agent_status(agent_id)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/cursor-agent/{agent_id}/conversation")
async def cursor_agent_conversation_proxy(agent_id: str):
    try:
        return await _cursor_api().get_conversation(agent_id)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/cursor-agent/{agent_id}/artifacts")
async def cursor_agent_artifacts_proxy(agent_id: str):
    try:
        return await _cursor_api().list_agent_artifacts(agent_id)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


class CursorFollowupBody(BaseModel):
    text: str
    images: Optional[List[str]] = None


@app.post("/cursor-agent/{agent_id}/followup")
async def cursor_agent_followup_proxy(agent_id: str, body: CursorFollowupBody):
    try:
        return await _cursor_api().followup(agent_id, body.text, images=body.images)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/cursor-agent/{agent_id}/stop")
async def cursor_agent_stop_proxy(agent_id: str):
    try:
        return await _cursor_api().stop_agent(agent_id)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.delete("/cursor-agent/{agent_id}")
async def cursor_agent_delete_proxy(agent_id: str):
    try:
        await _cursor_api().delete_agent(agent_id)
        return {"ok": True}
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Windsurf Enterprise Analytics proxy
# ---------------------------------------------------------------------------


class WindsurfCascadeAnalyticsRequest(BaseModel):
    service_key: Optional[str] = None
    start_timestamp: str
    end_timestamp: str
    query_requests: List[Dict[str, Any]] = Field(
        default_factory=lambda: [{"cascade_runs": {}}],
    )
    group_name: Optional[str] = None
    emails: Optional[List[str]] = None
    ide_types: Optional[List[str]] = None


@app.post("/windsurf/analytics/cascade")
async def windsurf_cascade_analytics_proxy(body: WindsurfCascadeAnalyticsRequest):
    from api.windsurf_analytics import post_cascade_analytics

    harness = AgentHarnessSettings()
    key = body.service_key or harness.WINDSURF_ANALYTICS_SERVICE_KEY
    if not key:
        raise HTTPException(
            status_code=400,
            detail="service_key required (body or WINDSURF_ANALYTICS_SERVICE_KEY)",
        )
    try:
        return await post_cascade_analytics(
            key,
            start_timestamp=body.start_timestamp,
            end_timestamp=body.end_timestamp,
            query_requests=body.query_requests,
            group_name=body.group_name,
            emails=body.emails,
            ide_types=body.ide_types,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
