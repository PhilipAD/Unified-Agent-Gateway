# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

_Changes that are merged to `main` but not yet released._

---

## [0.1.0] — 2025-03-26

Initial public release.

### Added

**Core**
- `core/types.py` — Normalized `NormalizedMessage`, `ToolCall`, `ToolDefinition`, `NormalizedResponse`, `StreamEvent`, `GatewayError`.
- `core/agent_loop.py` — Provider-agnostic multi-hop tool-calling loop with context injection, per-step tracing, configurable `max_tool_hops` and `tool_timeout`.
- `core/execution.py` — Durable execution primitives: `RunRecord`, `StepRecord`, `RunStore`, `RetryPolicy`.
- `core/handoff.py` — Agent handoff meta-tool (`call_agent`) for multi-agent delegation.

**Providers**
- `providers/openai_compatible.py` — Full OpenAI API adapter (indexed tool-call accumulation, streaming, `GatewayError` mapping). Also covers Groq, DeepSeek, Together, Ollama, Mistral, Azure OpenAI.
- `providers/anthropic.py` — Anthropic Claude adapter (`tool_use`/`tool_result` blocks, streaming via `messages.stream`).
- `providers/gemini.py` — Google Gemini adapter (function call IDs, `anyio` async bridge, streaming).

**Tools**
- `tools/registry.py` — `ToolRegistry` with `ToolSource` enum (python, mcp, http, context_forge), `RegisteredTool`, per-request `copy()`.
- `tools/mcp_loader.py` — Auto-discover and register tools from any MCP client (`list_tools` / `call_tool` protocol).
- `tools/mcp_http_client.py` — `InlineMCPClient` async context manager over `streamable_http` or `sse` transport (official `mcp` SDK).

**Context**
- `context/registry.py` — `ContextRegistry` with `ContextSource` enum (context_forge, rag, static, kv), `RegisteredContext`, per-request `copy()`.
- `context/contextforge.py` — HTTP adapter for ContextForge context injection.

**Config**
- `config/settings.py` — Pydantic-settings with `ProviderSettings`, `IntegrationSettings`, `GatewaySettings`.
  - `AgentProfile` — per-profile provider + model + preset binding (`mcp_namespaces`, `context_names`).
  - `MCPServerPreset` — named MCP server (url, transport, headers, timeout).
  - `NamedContextPreset` — named context source (static or HTTP fetch).
  - `OAICompatibleProviderPreset` — named extra OpenAI-compatible provider (Groq, DeepSeek, …).

**Runtime**
- `runtime/router.py` — Profile/agent-id routing, named OAI-compatible provider resolution, `merge_provider_config_overrides` for BYOK.
- `runtime/bootstrap.py` — Startup wiring: MCP client connection, ContextForge registration, named context preset registration.
- `runtime/sse.py` — SSE frame formatting helper.

**API**
- `api/http.py` — FastAPI `/agent-query` (sync) and `/agent-query/stream` (SSE) with:
  - `RuntimeRegistryConfig` — per-request tool/context overrides.
  - `DynamicHTTPTool` — inline HTTP tool spec.
  - `DynamicMCPServer` — inline MCP server spec (full connection details).
  - `DynamicContext` — inline context source spec.
  - `ProviderRequestCredentials` — BYOK api_key / model / base_url.
  - Named preset resolution (`mcp_namespaces`, `context_names` from settings).
  - Profile-level preset auto-merging.
  - `ALLOW_DYNAMIC_RUNTIME_REGISTRATION` and `ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS` gates.

**Tests**
- 101 pytest tests covering: types, agent loop, all three providers, tool registry, context registry, MCP loader, MCP HTTP client, SSE, router, settings, API (sync + stream, dynamic registration, named presets, BYOK), execution, handoff.

**Docs and tooling**
- `docs/ARCHITECTURE.md` — Full architecture reference.
- `docs/API_SPEC.md` — Endpoint spec with all fields.
- `postman/unified-agent-gateway.postman_collection.json` — 52 requests across 10 folders.
- `.env.example` — Fully documented environment variable reference.
- `pyproject.toml` — Hatchling build, ruff config, pytest config.

[Unreleased]: https://github.com/PhilipAD/Unified-Agent-Gateway/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PhilipAD/Unified-Agent-Gateway/releases/tag/v0.1.0
