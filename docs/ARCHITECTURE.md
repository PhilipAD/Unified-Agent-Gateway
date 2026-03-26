# Unified Agents SDK -- Architecture

## Purpose

A **product-agnostic** agent runtime that normalizes communication with
multiple LLM providers behind one interface.  Any backend service can embed
this gateway to get:

* Consistent tool calling across OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral, and xAI/Grok.
* A single agent loop that owns retries, tool hops, and tracing.
* Pluggable context injection (ContextForge, RAG, static, etc.).
* MCP tool auto-discovery and namespaced registration.
* A thin FastAPI HTTP/SSE layer for direct use or embedding.

## Directory Layout

```
core/
  types.py          Normalized messages, tools, responses, stream events
  agent_loop.py     Provider-agnostic tool-calling loop

providers/
  base.py               BaseProvider ABC (run + stream)
  openai_compatible.py   Generic OpenAI-compatible adapter (Together, Ollama, Azure, etc.)
  openai_responses.py    OpenAI Responses API (built-in tools, MCP, reasoning, stateful)
  anthropic.py           Claude via Anthropic SDK (thinking, server tools, citations)
  gemini.py              Gemini via google-genai SDK (built-in tools, native MCP, thinking)
  groq.py                Groq API (compound models, built-in search/code tools, reasoning)
  deepseek.py            DeepSeek API (reasoning content, thinking, multi-turn passthrough)
  mistral.py             Mistral AI SDK (chat + agents API, multimodal, guardrails)
  xai.py                 xAI/Grok via OpenAI Responses compat (x_search, citations, cost)

tools/
  registry.py       ToolRegistry with ToolSource enum
  mcp_loader.py   Auto-register MCP server tools
  mcp_http_client.py  InlineMCPClient (streamable_http / sse) for per-request MCP

context/
  registry.py       ContextRegistry with ContextSource enum
  contextforge.py   ContextForge HTTP integration

runtime/
  router.py         Provider factory, profile routing, named OAI presets, BYOK merge
  sse.py            SSE formatting helpers
  bootstrap.py      Startup wiring for tools, context, named presets from settings

api/
  http.py           FastAPI endpoints (/agent-query, /agent-query/stream)

config/
  settings.py       Pydantic-settings for env-based configuration

tests/              pytest suite (unit + optional integration)
```

## Key Concepts

### Normalized Types

All providers map to and from:

* `NormalizedMessage` -- role, content, optional tool_calls, tool_call_id
* `ToolCall` -- id, name, arguments (parsed dict)
* `ToolDefinition` -- name, description, JSON Schema
* `NormalizedResponse` -- messages, conversation transcript, usage (Dict[str, Any]), raw
* `StreamEvent` -- type (chunk/tool_call/usage/metadata/error/done), delta, tool_call, metadata

### Provider Adapters

Each adapter translates NormalizedMessage lists into provider-native API
calls and parses responses back.  Providers never see MCP, ContextForge,
or tool source metadata.

### Agent Loop

`AgentLoop.run_conversation()`:

1. Loads context from `ContextRegistry`.
2. Calls the provider.
3. If the assistant returns `tool_calls`, executes each via `ToolRegistry`.
4. Appends tool results and loops (up to `max_tool_hops`).
5. Returns the final response plus full conversation transcript.

### Tool Registry

Tools carry a `ToolSource` tag (python, mcp, http, context_forge).
`list_for_provider()` strips metadata and returns plain `ToolDefinition`
objects so providers remain source-agnostic.

### Context Registry

Passive context sources (`ContextSource`: context_forge, rag, static, kv)
are fetched before the first model call and injected as system messages.
Each source has optional `max_chars` truncation and `required` flags.

### MCP Integration

`tools/mcp_loader.py` accepts any client implementing `list_tools()` and
`call_tool()`.  Tools are registered under `namespace.tool_name` to avoid
collisions across multiple MCP servers.

`tools/mcp_http_client.py` wraps the official `mcp` SDK's `ClientSession` over
**streamable HTTP** (recommended) or **SSE** (legacy) for remote MCP servers.
The HTTP API can connect per-request via:

* **Named presets** — `runtime.mcp_namespaces` resolves keys from `MCP_SERVERS` in settings (credentials stay server-side).
* **Inline** — `runtime.mcp_servers` carries full `url`, `namespace`, `transport`, `headers` for ad-hoc servers.

`runtime.namespace` adds a prefix to every qualified MCP namespace for multi-tenant isolation.

### Routing

`(agent_id, profile)` maps to a `ProviderConfig` via `AGENT_PROFILES`
in settings.  Consuming products define their own routing rules through
environment or configuration without touching gateway code.

Dedicated providers (Groq, DeepSeek, Mistral, xAI) each have their own
adapter that surfaces provider-specific capabilities (reasoning, built-in
tools, citations, etc.).  Generic OpenAI-compatible providers (Together,
Ollama, Azure) are referenced via `OPENAI_COMPATIBLE_PROVIDERS` entries
and resolve to the `openai_compatible` adapter with preset `base_url` /
`model` / `api_key`.

`AgentProfile` may attach `mcp_namespaces` and `context_names` so every
request using that profile automatically includes those presets without
repeating them in the JSON body.

### Bring-your-own-key (BYOK)

Optional `provider_credentials` on the HTTP request overrides the resolved
API key, model, and (for OpenAI-compatible) `base_url` for a single call.
Gated by `ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS` (default **false**).

## Security

* Default: API keys are loaded server-side from environment variables and profiles.
* Optional BYOK: clients may send keys in the JSON body when explicitly enabled — keys can appear in logs and proxies; use only behind trusted auth.
* **Dynamic runtime** (`runtime` tools/MCP/contexts) is gated by `ALLOW_DYNAMIC_RUNTIME_REGISTRATION`.
* Provider adapters do not log request/response bodies at INFO level.

## Extensibility

* **New provider:** subclass `BaseProvider`, register in `runtime/router.py`.
* **New tool source:** add to `ToolSource` enum, register handlers normally.
* **New context source:** add to `ContextSource` enum, register a fetcher.
* **Handoffs:** reserve a `call_agent` meta-tool to route to another
  agent profile without re-architecting.

## Durable Execution (Future)

* Per-step timeouts and retry policies are configurable.
* The agent loop records step traces (model calls, tool executions).
* Storage is pluggable (in-memory now, DB-backed later) for resume.
