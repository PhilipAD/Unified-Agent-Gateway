# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

_Changes that are merged to `main` but not yet released._

### Fixed

- Pin `ruff==0.15.8` in dev dependencies and bump `ruff-pre-commit` to `v0.15.8` so `ruff format --check` in CI matches local/pre-commit output (avoids failures when `ruff>=0.6,<1` resolves to a newer formatter).

---

## [0.3.0] — 2026-03-27

### Added

- `providers/_shared.py` — Shared helpers: `CHAT_ROLE_MAP`, `normalize_openai_usage`, `normalize_responses_usage`, `msg_to_openai_chat`, `build_openai_chat_tools`, streaming tool deltas, `to_responses_input_items` / `to_responses_tools` / `parse_responses_output` for OpenAI Responses–style providers.
- `providers/gemini.py` — Optional Vertex AI and `http_options` via provider `extra` (`vertex_ai`, `vertex_project`, `vertex_location`, `vertex_credentials`).

### Changed

- **Breaking:** `openai_compatible` (and streaming) now returns normalised usage with `input_tokens` / `output_tokens` instead of passing through raw `prompt_tokens` / `completion_tokens`. Clients that parsed only the old keys should read the new fields (or use `normalize_openai_usage` logic).
- `providers/openai_responses.py`, `providers/xai.py` — Refactored to use `_shared` parsers and usage helpers; xAI streaming usage now includes cached/reasoning/cost fields where available.
- `providers/groq.py`, `providers/mistral.py`, `providers/deepseek.py` — Use shared chat message / tool builders or `CHAT_ROLE_MAP` where applicable.
- `providers/anthropic.py` — Removed no-op citation block; `citations` kwarg is documented as per-block on documents, not top-level.

### Documentation

- `README.md` — Full **MCP — Model Context Protocol** section (gateway Path 1 vs provider Path 2, bridges, Claude Agent / Codex / Copilot); curated harness table clarified; architecture diagram updated; Postman pointer from MCP section.
- `postman/unified-agents-sdk.postman_collection.json` — **62 requests** in **11 folders**: OpenAI Responses / xAI / Mistral profiles (folder 2); dynamic contexts `md_hierarchy`, `md_files`, `md_glob` (folder 6); **MCP README scenarios** (folder 11). Collection `info.description` cross-links the README.

---

## [0.2.1] — 2026-03-26

### Fixed

- `README.md` — Convert relative `assets/logo.jpg` image path to absolute raw GitHub URL so the logo renders correctly on PyPI.

---

## [0.2.0] — 2026-03-26

Full provider coverage, pip-installable package with `uag` CLI, and OSS hardening.

### Added

**Package and CLI**
- `pyproject.toml` — Bumped to v0.2.0; added `typer`, `rich` to dependencies; added `[project.scripts]` entry `uag = "cli:app"` for CLI access after `pip install`; added `pytest-cov`, `coverage`, `pre-commit` to dev deps; added `force-include` for `cli.py` and `py.typed` in wheel.
- `cli.py` — Typer-based CLI with three commands: `uag serve` (start HTTP gateway with host/port/reload/workers options), `uag chat` (in-process query with `--profile`, `--stream`, `--json`, `--system` flags), `uag providers` (list registered providers with config status table).
- `py.typed` — PEP 561 typed package marker.
- `Makefile` — Dev task runner with `install`, `test`, `test-cov`, `lint`, `format`, `serve`, `clean`, `build` targets.

**CI/CD and OSS**
- `.github/workflows/publish.yml` — PyPI publish workflow using OIDC trusted publishing, triggered on `v*.*.*` tags.
- `.github/workflows/ci.yml` — Added `pytest-cov` coverage reporting with Codecov upload.
- `.github/dependabot.yml` — Weekly automated dependency updates for pip and GitHub Actions.
- `.pre-commit-config.yaml` — Pre-commit hooks for ruff lint + format on every commit.
- `CONTRIBUTING.md` — Added `pre-commit install` to setup steps; updated commands to use `make` targets.

**New Providers**
- `providers/openai_responses.py` — OpenAI Responses API adapter with built-in tools (web search, file search, code interpreter, computer use, image generation), remote MCP support, reasoning effort/summary, stateful sessions (`previous_response_id`), and detailed usage tracking (cached/reasoning tokens).
- `providers/groq.py` — Dedicated Groq adapter with compound models (`compound-beta`/`compound-beta-mini`), built-in tools (browser_search, code_interpreter, web_search), reasoning (format, effort, include), documents, search settings, citations/annotations, `executed_tools` structured data, `usage_breakdown`, and `x_groq` metadata.
- `providers/deepseek.py` — Dedicated DeepSeek adapter with `reasoning_content` extraction for `deepseek-reasoner`, `thinking` parameter with DX helpers (bool/string/dict normalization), multi-turn reasoning passthrough, `finish_reason` capture, streaming usage via `stream_options`, and extended timeout for reasoning models.
- `providers/mistral.py` — Mistral AI SDK adapter supporting chat and Agents API (`agent_id`), multimodal inputs (images, documents, audio, files), structured outputs (JSON schema), reasoning effort, guardrails, safe prompt, speculative decoding (prediction), and streaming.
- `providers/xai.py` — xAI/Grok adapter via OpenAI Responses API compatibility with provider-specific built-in tools (`x_search`, `collections_search`, `attachment_search`), reasoning, live search parameters, inline citation extraction, cost tracking (`cost_in_usd_ticks`), and deferred completions.

**Enhanced Existing Providers**
- `providers/openai_responses.py` — Extended `_to_tools` for MCP `connector_id`/`authorization`/`defer_loading`; added `reasoning_summary` and `include` parameters; extract `cached_tokens` and `reasoning_tokens` from usage; stream `reasoning_summary_text` and `mcp_call`/`web_search_call` events; handle `response.failed`/`response.incomplete` as error events.
- `providers/anthropic.py` — Added `thinking_signature` in assistant messages; `document` and `search_result` user content types; server-side tools support; `thinking_type` including "adaptive"; `thinking_display`, `cache_control`, `output_config`, `citations_config`; citations extraction; `stop_reason` and `container` metadata; detailed usage (cache creation/read, server tool use).
- `providers/gemini.py` — Added multimodal user input (base64 images, file URIs); `url_context`, `google_maps`, `computer_use`, `file_search` built-in tools; native `McpServer` support; `built_in_tool_configs`; `thinking_level`; `tool_config`, `safety_settings`, `response_schema`, `response_mime_type`; grounding metadata; `thoughts_token_count` and `cached_content_token_count` in usage.

**Core Types**
- `core/types.py` — Added `metadata` and `error` fields to `StreamEvent`; changed `NormalizedResponse.usage` from `Dict[str, int]` to `Dict[str, Any]` to support structured usage details (reasoning tokens, cached tokens, timing info, tool execution metadata).

**Configuration**
- `config/settings.py` — Added `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEFAULT_DEEPSEEK_MODEL`, `MISTRAL_API_KEY`, `MISTRAL_BASE_URL`, `DEFAULT_MISTRAL_MODEL`, `XAI_API_KEY`, `XAI_BASE_URL`, `DEFAULT_XAI_MODEL` to `ProviderSettings`.
- `runtime/router.py` — Registered `OpenAIResponsesProvider`, `GroqProvider`, `DeepSeekProvider`, `MistralProvider`, `XAIProvider` in provider factory; updated profile resolution for all new providers.

**Dependencies**
- `pyproject.toml` — Added `groq>=0.9`, `mistralai>=2.0` as dependencies; updated keywords.

**Tests**
- 200 pytest tests (up from 101), all passing, all offline.
- `tests/test_openai_responses_provider.py` — Tests for MCP tool conversion, connector fields, built-in tool types.
- `tests/test_groq_provider.py` — Tests for compound model detection, message conversion (including reasoning), tool definitions.
- `tests/test_deepseek_provider.py` — Tests for `_normalize_thinking_param` (bool/dict/string/None), reasoning passthrough in messages.
- `tests/test_mistral_provider.py` — Tests for message conversion, multimodal content parts, tool definitions, agent tool types.
- `tests/test_xai_provider.py` — Tests for input item conversion, tool definitions (including x_search), built-in tool types.
- `tests/test_gemini_provider.py` — Tests for built-in tools (url_context, google_maps, file_search, computer_use, mcp_servers).
- Updated `tests/test_anthropic_provider.py` — Tests for document/search_result content, cache_control, server-side tools.
- Updated `tests/test_router.py` — Tests for Mistral and xAI provider creation and profile resolution.

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
- `postman/unified-agents-sdk.postman_collection.json` — 52 requests across 10 folders.
- `.env.example` — Fully documented environment variable reference.
- `pyproject.toml` — Hatchling build, ruff config, pytest config.

[Unreleased]: https://github.com/PhilipAD/Unified-Agents-SDK/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/PhilipAD/Unified-Agents-SDK/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/PhilipAD/Unified-Agents-SDK/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/PhilipAD/Unified-Agents-SDK/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/PhilipAD/Unified-Agents-SDK/releases/tag/v0.1.0
