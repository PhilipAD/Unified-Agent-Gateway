<h1 align="center">
  <img src="https://raw.githubusercontent.com/PhilipAD/Unified-Agents-SDK/main/assets/logo.jpg" alt="Unified Agents SDK" width="80" style="vertical-align: middle; border-radius: 16px;"><br><br>
  Unified Agents SDK
</h1>

<p align="center">
  <strong>One API. Every LLM. Any tool.</strong><br>
  Normalize OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral, xAI/Grok — and any OpenAI-compatible endpoint — behind a single REST/SSE interface with MCP, RAG, and bring-your-own-key support.
</p>

<p align="center">
  <a href="https://pypi.org/project/unified-agents-sdk/"><img src="https://img.shields.io/pypi/v/unified-agents-sdk?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-3_min-blue?style=for-the-badge" alt="Quick Start"></a>
  <a href="#-api-usage"><img src="https://img.shields.io/badge/API-REST_%2B_SSE-green?style=for-the-badge" alt="API"></a>
  <a href="postman/unified-agents-sdk.postman_collection.json"><img src="https://img.shields.io/badge/Postman-52_examples-orange?style=for-the-badge&logo=postman&logoColor=white" alt="Postman"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.10-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/providers-OpenAI_%7C_Anthropic_%7C_Gemini_%7C_Groq_%7C_DeepSeek_%7C_Mistral_%7C_xAI-blueviolet" alt="Providers">
  <img src="https://img.shields.io/badge/MCP-streamable__http_%7C_sse-orange" alt="MCP">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
  <a href="https://github.com/PhilipAD/Unified-Agents-SDK/actions/workflows/ci.yml">
    <img src="https://github.com/PhilipAD/Unified-Agents-SDK/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

---

## 🤔 Why?

Every LLM provider speaks a different dialect. Tool calling, streaming, context — all incompatible. You end up writing provider-specific glue code everywhere.

**Unified Agents SDK solves it once. Three ways to use it:**

<table>
<tr>
<td width="50%">

### Before — provider-specific glue everywhere

```python
# Different tool schema for every provider
openai_client.chat.completions.create(
    model="gpt-4o",
    tools=[{"type": "function", "function": {
        "name": "search", "parameters": {...}
    }}],
)

anthropic_client.messages.create(
    model="claude-opus-4-5",
    tools=[{"name": "search", "input_schema": {...}}],
)

groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    tools=[{"type": "function", "function": {...}}],
)
# ...repeat for every provider, every format change
```

</td>
<td width="50%">

### After — one interface, any provider

**Mode 1 — CLI** (no server, no code)
```bash
uag chat "Search and summarise" --profile claude
uag chat "Explain ML" --stream
uag providers
```

**Mode 2 — Python** (embed in your app, swap providers with one string)
```python
from runtime.router import create_provider, resolve_provider_config
from config.settings import ProviderSettings, GatewaySettings
from core.agent_loop import AgentLoop
from core.types import NormalizedMessage, Role

provider_settings = ProviderSettings()
gateway_settings = GatewaySettings()

messages = [NormalizedMessage(role=Role.USER, content="Search and summarise")]

# Switch provider by changing one argument — code is identical
for profile in ["default", "claude", "gemini", "fast", "deep", "grok"]:
    cfg = resolve_provider_config(
        provider_settings, gateway_settings, profile=profile
    )
    loop = AgentLoop(provider=create_provider(cfg))
    response = await loop.run_conversation(messages)
    print(f"[{cfg.provider_name}] {response.messages[-1].content}")
```

**Mode 3 — HTTP / REST** (language-agnostic)
```bash
curl http://localhost:8000/agent-query \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Search and summarise",
    "profile": "claude",
    "runtime": {"mcp_namespaces": ["search"]}
  }'
```

</td>
</tr>
</table>

---

## ✨ What You Get

<table>
<tr>
<td width="25%" align="center" style="vertical-align: top; padding: 12px;">

### 🔌 Provider Adapters
OpenAI · Anthropic · Gemini<br>
Groq · DeepSeek · Mistral · xAI/Grok<br>
Together · Ollama · Azure · any OAI endpoint

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 12px;">

### 🛠️ MCP Native
Connect any [MCP](https://modelcontextprotocol.io) server per-request.<br>
Named presets or full inline spec.<br>
Both `streamable_http` and `sse`.

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 12px;">

### 💉 Context Injection
RAG · KV · static text · ContextForge<br>
Injected before every model call.<br>
Per-profile or per-request.

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 12px;">

### 📡 SSE Streaming
Real-time token streaming<br>
for every provider.<br>
Same event shape, always.

</td>
</tr>
</table>

---

## 🗺️ Feature Matrix

**Every provider runs the same unified agent runtime** (multi-hop tool loop, shared HTTP API, normalized types). Each column is one **vendor integration** through that runtime—not a comparison of who “has an agent.” Rows below call out **optional vendor-specific surfaces** (built-in tools, how PDFs are passed, Mistral’s separate Agents HTTP API, etc.).

> **Using OpenAI?** Use the **OpenAI Responses** column (`openai_responses` adapter / `"openai-r"` profile). OpenAI’s [Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses) is a superset of Chat Completions and is recommended for all new projects—it adds built-in tools, MCP, stateful context, and better reasoning. The **OAI-compat¹** column is the *generic* adapter for third-party OpenAI-compatible endpoints (Together AI, Ollama, Azure OpenAI, local models, etc.) that do not expose the Responses API.

**Legend**

| Symbol | Meaning |
|--------|--------|
| ✅ | Supported and wired in this SDK for that column’s adapter. |
| — | Genuine gap in this SDK for that column, or a real vendor limitation. See [When the matrix shows a dash](#when-the-matrix-shows-a-dash). |
| †  | Vendor-specific flow — see **Footnotes (†)** below the table. |
| ‡  | Automatic platform behaviour — no request change needed; see note below the table. |

| Feature | OAI-compat¹ | OpenAI Responses | Anthropic | Gemini | Groq | DeepSeek | Mistral | xAI/Grok |
|---------|:----------:|:----------------:|:---------:|:------:|:----:|:--------:|:-------:|:--------:|
| Sync chat | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SSE streaming | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool / function calling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multi-hop tool loops | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Context injection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MCP tool auto-discovery | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| Server-side built-in tools | — | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Extended thinking / reasoning | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vision / multimodal input | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Structured outputs (JSON schema) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Prompt caching | — | ✅ | ✅ | ✅ | ✅‡ | ✅ | — | ✅ |
| Citations | — | ✅ | ✅ | ✅† | ✅ | — | — | ✅ |
| Document / PDF input | — | ✅† | ✅ | ✅† | ✅ | — | ✅ | — |
| Live web search | — | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| Mistral Agents API (`agent_id`) | — | — | — | — | — | — | ✅ | — |
| Inline BYOK credentials | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**¹ OAI-compat** = `openai_compatible` adapter—the generic Chat Completions path for any OpenAI-format endpoint (Together AI, Ollama, Azure OpenAI, local models, etc.). For OpenAI itself, use the **OpenAI Responses** column.

**‡ Groq prompt caching** is automatic (prefix reuse, no request change needed). `cached_tokens` is reported in the usage response via `prompt_tokens_details` and `x_groq` DRAM/SRAM breakdown.

**Mistral Agents API (`agent_id`):** This row is **only** [Mistral's separate Agents HTTP API](https://docs.mistral.ai/agents/agents)—optional `agent_id` routing to `agents.complete` / server-defined agents and built-in tools. **All other providers** still run full agent loops through their own chat or Responses-style APIs; they do not expose Mistral's branded Agents API, which is why the column is empty outside Mistral.

### Footnotes (†)

**Citations · Gemini (✅†):** [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search) returns `grounding_metadata` (web chunks, queries, support references). This SDK extracts and surfaces it in the `usage` response key `grounding_metadata` — see `providers/gemini.py`. It is citation-like grounding, not an inline text-annotation object like Anthropic's.

**Document / PDF · Gemini (✅†):** Upload files with the [Gemini Files API](https://ai.google.dev/gemini-api/docs/files), then include the file URI in user message parts (`type: "file"`, `uri`, `mime_type`) — see `_convert_user_content_parts` in `providers/gemini.py`.

**Document / PDF · OpenAI Responses (✅†):** Pass `{"type": "file_search", "vector_store_ids": [...]}` in `built_in_tools`. This SDK wires it through `_to_tools` in `providers/openai_responses.py`. See OpenAI's [file inputs guide](https://developers.openai.com/api/docs/guides/file-inputs?api-mode=responses) for creating files and passing them inline.

### When the matrix shows a dash

Every **—** is a genuine gap in **this SDK** or a real vendor limitation.

| Row · column | Reason |
|--------------|--------|
| **MCP · Mistral** | Mistral [supports MCP](https://docs.mistral.ai/agents/tools/mcp) via `RunContext` + Agents SDK — a different API pattern. This gateway has not yet bridged it to the unified `MCP_SERVERS` / `mcp_namespaces` path. |
| **Server-side built-in tools · OAI-compat¹** | Hosted tools require the **Responses API**. The `openai_compatible` column is generic Chat Completions — bring your own function tools. |
| **Server-side built-in tools · DeepSeek** | DeepSeek has no hosted tool service. All tools are user-supplied via function calling. |
| **Extended thinking · OAI-compat¹** | Reasoning parameters are wired only in the **OpenAI Responses** adapter. The generic `openai_compatible` column does not map them (though extra kwargs are forwarded). |
| **Vision · DeepSeek** | The official [DeepSeek API](https://api-docs.deepseek.com/) (`deepseek-chat`, `deepseek-reasoner`) does not support image input. DeepSeek VL is a separate product. |
| **Prompt caching · OAI-compat¹** | The `openai_compatible` adapter is generic Chat Completions. OpenAI's server-side prefix cache is implicit; no explicit cache-control API on this path. |
| **Prompt caching · Mistral** | Mistral has no documented explicit prompt-cache API. |
| **Citations · OAI-compat¹** | Inline `url_citation` annotations require **Responses + web search tool**. Generic Chat Completions returns no citation annotation objects. |
| **Citations · DeepSeek / Mistral** | Neither API returns structured citation annotation objects. |
| **Document / PDF · OAI-compat¹** | The generic `openai_compatible` Chat Completions path has no native document block. Use the `openai_responses` column with `file_search` (see † above). |
| **Document / PDF · DeepSeek** | DeepSeek's chat API has no native document or PDF block. |
| **Document / PDF · xAI** | xAI Grok has no documented inline file/PDF input on the Responses path used by this SDK. |
| **Live web search · OAI-compat¹** | Hosted web search requires the **Responses API**. Generic Chat Completions uses function tools you supply. |
| **Live web search · DeepSeek** | No hosted search tool; implement search as a custom function. |

### Curated agent harnesses

These are **additional** surfaces beyond the core LLM columns above. They reuse the same UAG patterns: `BaseProvider` + `NormalizedResponse` / `StreamEvent` where a programmatic driver exists, `ContextRegistry` for on-disk rules, and `GatewaySettings.MCP_SERVERS` for HTTP/SSE MCP bridges.

Shared utilities (no duplicated walk logic): `context/md_hierarchy.py`, `tools/mcp_config_loader.py`, one `AgentHarnessSettings` class in `config/settings.py`, and optional `AGENTS.md` via `context/agents_md.py`.

| Harness | Provider in `PROVIDERS` | MCP / tools bridge | Context bridge | Notes |
|---------|:----------------------:|:------------------:|:--------------:|-------|
| **Claude Agent SDK** | `claude_agent` | In-process MCP via SDK (`pip install 'unified-agents-sdk[claude-agent]'`) | Claude loads `CLAUDE.md` / skills when `setting_sources` set in `AgentProfile.extra` | Wraps `claude_agent_sdk.query` (+ options); UAG tools can be attached via `mcp_servers` in `extra`. |
| **Gemini CLI** | — | `GEMINI_CLI_MCP_BRIDGE` merges HTTP/SSE from `settings.json` into `MCP_SERVERS` | `GEMINI_CLI_MD_ENABLED`, `GEMINI_CLI_SKILLS_ENABLED` register `gemini_md` / `gemini_skills` | No Gemini CLI Python SDK; parity via file + preset bridge. |
| **Cursor Cloud Agents** | `cursor_cloud_agent` | API does not expose MCP for callers | — | REST job runner; `POST /webhooks/cursor` + `/cursor-agent/{id}/*` proxies; set `CURSOR_API_KEY`, `repository` in `AgentProfile.extra`. |
| **Codex CLI** | `codex` | `CODEX_MCP_ENABLED` runs `codex mcp-server` (stdio) at bootstrap | `AGENTS_MD_ENABLED` or Codex `--project-doc` via `extra` | Default: `codex -q` subprocess; optional `codex app-server` JSON-RPC via `use_app_server`. |
| **Windsurf Cascade** | — | `WINDSURF_MCP_BRIDGE` merges `~/.codeium/windsurf/mcp_config.json` | `WINDSURF_RULES_ENABLED` | No headless Cascade API; Enterprise analytics: `POST /windsurf/analytics/cascade`. |
| **Cline** | — | — | `CLINE_RULES_ENABLED` loads `.clinerules` | VS Code extension only; rules file bridge only. |
| **GitHub Copilot** | `copilot` | `COPILOT_MCP_BRIDGE` adds GitHub remote MCP preset | — | `pip install 'unified-agents-sdk[copilot]'` (preview SDK); token `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN`. |

**Optional install groups:** `[claude-agent]`, `[codex]` (binary separate), `[copilot]`.

---

## 📦 Dependencies

All constraints are declared in [`pyproject.toml`](pyproject.toml). Upper bounds prevent silent breakage from future major-version API changes.

### Provider SDKs

| Provider | Package | Tested version | Min required | Adapter |
|----------|---------|---------------|-------------|---------|
| OpenAI / OAI-compat | [`openai`](https://pypi.org/project/openai/) | **2.30.0** | `>=2.0,<3` | `openai_compatible`, `openai_responses`, `xai` |
| Anthropic | [`anthropic`](https://pypi.org/project/anthropic/) | **0.86.0** | `>=0.86,<1` | `anthropic` |
| Google Gemini | [`google-genai`](https://pypi.org/project/google-genai/) | **1.47.0** | `>=1.47,<2` | `gemini` |
| Groq | [`groq`](https://pypi.org/project/groq/) | **1.0.0** | `>=1.0,<2` | `groq` |
| Mistral | [`mistralai`](https://pypi.org/project/mistralai/) | **1.10.0** | `>=1.10,<2` | `mistral` |
| MCP | [`mcp`](https://pypi.org/project/mcp/) | **1.26.0** | `>=1.26,<2` | All (tool auto-discovery) |
| Claude Agent SDK | [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/) | optional | `[claude-agent]` extra | `claude_agent` provider |
| GitHub Copilot SDK | [`github-copilot-sdk`](https://pypi.org/project/github-copilot-sdk/) | optional | `[copilot]` extra | `copilot` provider |

> **Note:** `openai>=2.0` is required — the Responses API (`client.responses.create`) only exists in the v2 SDK. The `xai` adapter uses the same SDK pointed at `https://api.x.ai/v1`.

### Framework & transport

| Package | Tested version | Constraint | Purpose |
|---------|---------------|-----------|---------|
| [`fastapi`](https://pypi.org/project/fastapi/) | 0.135.2 | `>=0.115,<1` | HTTP / SSE server |
| [`uvicorn[standard]`](https://pypi.org/project/uvicorn/) | 0.42.0 | `>=0.30,<1` | ASGI server |
| [`httpx`](https://pypi.org/project/httpx/) | 0.28.1 | `>=0.27,<1` | Async HTTP (OAI-compat, DeepSeek) |
| [`pydantic`](https://pypi.org/project/pydantic/) | 2.12.5 | `>=2.0,<3` | Data models & validation |
| [`pydantic-settings`](https://pypi.org/project/pydantic-settings/) | 2.13.1 | `>=2.0,<3` | Env-based config |
| [`anyio`](https://pypi.org/project/anyio/) | 4.13.0 | `>=4.0,<5` | Async primitives |
| [`typer`](https://pypi.org/project/typer/) | 0.24.1 | `>=0.12,<1` | CLI (`uag serve`) |
| [`rich`](https://pypi.org/project/rich/) | 14.3.3 | `>=13.0,<15` | CLI output formatting |

---

## 🚀 Quick Start

### Option A — pip install (recommended)

```bash
pip install unified-agents-sdk
```

### Option B — from source

```bash
git clone https://github.com/PhilipAD/Unified-Agents-SDK.git
cd unified-agents-sdk
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Add at minimum one provider key:
#   OPENAI_API_KEY=sk-...
```

### Start the server

```bash
uag serve
# or: uag serve --reload --port 8000
```

Server: `http://localhost:8000` · Swagger UI: `http://localhost:8000/docs`

### First call (HTTP)

```bash
curl -s http://localhost:8000/agent-query \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the capital of France?"}' | python -m json.tool
```

```json
{
  "output": "The capital of France is Paris.",
  "tool_traces": [],
  "usage": {"input_tokens": 14, "output_tokens": 9},
  "provider": "openai_compatible",
  "model": "gpt-4o",
  "warnings": [],
  "errors": []
}
```

### First call (CLI -- no server needed)

```bash
uag chat "What is the capital of France?"
uag chat "Explain quantum computing" --profile claude --stream
uag chat "2+2?" --json
```

---

## 🏗️ Architecture

```
  Your app / curl / Postman
         │
         ▼
  ┌────────────────────────────────────────────────────────┐
  │   POST /agent-query   ·   POST /agent-query/stream     │
  │          FastAPI HTTP + SSE layer  (api/http.py)       │
  └────────────────────────┬───────────────────────────────┘
                           │
               ┌───────────▼───────────┐
               │       AgentLoop       │  core/agent_loop.py
               │  1. Inject context    │
               │  2. Call provider     │
               │  3. Execute tools     │
               │  4. Loop until done   │
               └──────┬────────┬───────┘
                      │        │
        ┌─────────────▼──┐  ┌──▼──────────────┐
        │  ToolRegistry  │  │ ContextRegistry  │
        │   tools/       │  │   context/       │
        │  • Python fns  │  │  • Static text   │
        │  • MCP servers │  │  • RAG / HTTP    │
        │  • HTTP tools  │  │  • ContextForge  │
        └────────────────┘  └─────────────────┘
                      │
        ┌──────────────────────▼────────────────────────────────────┐
        │                  Provider Adapters                       │
        │  openai_compatible │ openai_responses │ anthropic        │
        │  gemini │ groq │ deepseek │ mistral │ xai               │
        └─────────────────────────────────────────────────────────┘
```

### Layer reference

| Layer | Package | Responsibility |
|-------|---------|---------------|
| Config | `config/` | Pydantic-settings: env-loaded profiles, MCP presets, context presets |
| Core types | `core/` | Normalized messages, tool calls, responses, stream events |
| Providers | `providers/` | 8 adapters: translate normalized types to/from each provider's native API |
| Tools | `tools/` | Registry, MCP loader, inline MCP HTTP client |
| Context | `context/` | Registry, ContextForge adapter, RAG/KV fetch |
| Runtime | `runtime/` | Router, profile resolution, bootstrap, SSE helpers |
| API | `api/` | FastAPI endpoints, dynamic registry composition |

---

## ⚙️ Configuration

All settings are env vars (`.env` file supported). Full reference in [`.env.example`](.env.example).

<details open>
<summary><strong>Provider selection</strong></summary>

```env
# Built-in providers — set whichever you use
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk-...
DEEPSEEK_API_KEY=sk-ds-...
MISTRAL_API_KEY=...
XAI_API_KEY=xai-...

# Extra OpenAI-compatible providers (Together, Ollama, Azure…)
OPENAI_COMPATIBLE_PROVIDERS={"together":{"api_key":"...","base_url":"https://api.together.xyz/v1","model":"meta-llama/Llama-3.3-70B-Instruct-Turbo"}}
```

</details>

<details>
<summary><strong>Agent profiles</strong></summary>

Select provider + model + presets per request with a single `"profile"` field.

```env
AGENT_PROFILES={
  "default":    {"provider_name": "openai_compatible"},
  "openai-r":   {"provider_name": "openai_responses", "model": "gpt-4o"},
  "claude":     {"provider_name": "anthropic", "model": "claude-opus-4-5"},
  "gemini":     {"provider_name": "gemini", "model": "gemini-2.5-pro"},
  "fast":       {"provider_name": "groq", "model": "llama-3.3-70b-versatile"},
  "deep":       {"provider_name": "deepseek", "model": "deepseek-reasoner"},
  "mistral":    {"provider_name": "mistral", "model": "mistral-large-latest"},
  "grok":       {"provider_name": "xai", "model": "grok-4.20-reasoning"},
  "researcher": {
    "provider_name": "openai_responses",
    "mcp_namespaces": ["search"],
    "context_names":  ["company_info"]
  }
}
```

Profile-level `mcp_namespaces` and `context_names` are automatically merged into every request using that profile — callers don't need to repeat them.

</details>

<details>
<summary><strong>Named MCP servers</strong></summary>

Define once in `.env`, reference by name in API calls. Credentials never leave the server.

```env
MCP_SERVERS={
  "search": {"url": "http://search-mcp.internal/mcp",
             "transport": "streamable_http",
             "headers": {"Authorization": "Bearer sk-xyz"}},
  "files":  {"url": "http://files-mcp.internal/sse", "transport": "sse"}
}
```

</details>

<details>
<summary><strong>Named context sources</strong></summary>

```env
NAMED_CONTEXTS={
  "company_info": {"mode": "static",
                   "text": "We are Acme Corp, a global e-commerce platform."},
  "product_faq":  {"mode": "http", "source": "rag",
                   "url": "http://kb.internal/search",
                   "payload_template": {"query": "{input}"},
                   "max_chars": 4000}
}
```

</details>

---

## 📡 API Usage

Both endpoints accept the same JSON body.

### Sync query

```bash
curl -s http://localhost:8000/agent-query \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Summarise the refund policy.",
    "profile": "default",
    "context": {"system_prompt": "You are a helpful support agent."},
    "runtime": {
      "context_names": ["company_info"],
      "mcp_namespaces": ["ticketing"]
    }
  }'
```

### Streaming (SSE)

```bash
curl -N http://localhost:8000/agent-query/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Write a haiku about distributed systems."}'
```

```
event: chunk
data: {"type": "chunk", "delta": "Nodes whisper in time\n"}

event: chunk
data: {"type": "chunk", "delta": "Consensus blooms like spring rain\n"}

event: done
data: {"type": "done", "usage": {"input_tokens": 12, "output_tokens": 17}}
```

### Inline MCP server (per-request)

```bash
curl -s http://localhost:8000/agent-query \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Create a GitHub issue for the pagination bug.",
    "runtime": {
      "mcp_servers": [{
        "url": "https://mcp.github.example.com/mcp",
        "namespace": "github",
        "transport": "streamable_http",
        "headers": {"Authorization": "Bearer ghp_..."}
      }]
    }
  }'
```

### Inline HTTP tool

```bash
curl -s http://localhost:8000/agent-query \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is the weather in London?",
    "runtime": {
      "tools": [{
        "name": "weather",
        "description": "Get current weather for a city",
        "json_schema": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        },
        "url": "https://api.weather.example.com/current",
        "method": "GET",
        "argument_mode": "query"
      }]
    }
  }'
```

### Bring-your-own-key (BYOK)

Requires `ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=true` in `.env`.

```bash
curl -s http://localhost:8000/agent-query \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello!",
    "provider_credentials": {
      "api_key": "sk-user-supplied",
      "model": "gpt-4o-mini"
    }
  }'
```

---

## 📋 Request Schema

<details open>
<summary><strong>Top-level fields</strong></summary>

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `input` | string | **yes** | User message / instruction |
| `profile` | string | no | Agent profile name (default: `"default"`) |
| `agent_id` | string | no | Agent identifier; combined with profile for lookup |
| `context` | object | no | `system_prompt` + any template variables |
| `options` | object | no | Extra kwargs forwarded to provider (`temperature`, `max_tokens`, …) |
| `runtime` | object | no | Per-request tool / context overrides |
| `provider_credentials` | object | no | BYOK: `api_key`, `model`, `base_url` |

</details>

<details>
<summary><strong><code>runtime</code> fields</strong></summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_global_tools` | bool | `true` | Include globally registered tools |
| `use_global_contexts` | bool | `true` | Include globally registered contexts |
| `namespace` | string | — | Prefix for all tools/contexts registered in this request |
| `mcp_namespaces` | string[] | `[]` | Keys from `MCP_SERVERS` in `.env` — connect per-request |
| `context_names` | string[] | `[]` | Keys from `NAMED_CONTEXTS` in `.env` — inject per-request |
| `mcp_servers` | object[] | `[]` | Inline MCP: `url`, `namespace`, `transport`, `headers` |
| `tools` | object[] | `[]` | Inline HTTP tools: `name`, `description`, `json_schema`, `url`, `method` |
| `contexts` | object[] | `[]` | Inline contexts: `name`, `mode`, `text` / `url` |

</details>

Full schema and SSE event reference: [`docs/API_SPEC.md`](docs/API_SPEC.md)

---

## 📬 Postman Collection

Import [`postman/unified-agents-sdk.postman_collection.json`](postman/unified-agents-sdk.postman_collection.json) for **52 ready-made requests** across **10 folders** covering every feature and error case.

| Folder | Requests | What it covers |
|--------|:--------:|----------------|
| 1 — Basic Queries | 3 | Minimal, system prompt, agent_id targeting |
| 2 — Provider Profiles | 6 | OpenAI, Anthropic, Gemini, Groq, DeepSeek, options |
| 3 — Named Presets | 5 | MCP namespaces, context names, profile-baked presets |
| 4 — Inline MCP Servers | 5 | Streamable HTTP, SSE, auth headers, multi-server |
| 5 — Inline HTTP Tools | 5 | POST/JSON, GET/query, multi-tool, auth, isolated |
| 6 — Inline Contexts | 7 | Static, templates, RAG, GET, multi-context, ContextForge |
| 7 — Kitchen Sink | 3 | Full combined, sandboxed, real-world DevOps scenario |
| 8 — Streaming (SSE) | 5 | Minimal, profile, context, MCP, warning event |
| 9 — Error Cases | 6 | 403, unknown namespace/context, MCP failure, bad key |
| 10 — BYOK | 7 | api_key, model, base_url, Groq, Anthropic, stream, disabled |

Set the `base_url` collection variable to your server address.

---

## 🔧 Extending the SDK

<details>
<summary><strong>Add a new LLM provider</strong></summary>

1. Create `providers/myprovider.py` subclassing `BaseProvider`.
2. Implement `run()` (sync) and `stream()` (async generator of `StreamEvent`).
3. Register in `runtime/router.py`:

```python
from providers.myprovider import MyProvider
PROVIDERS["myprovider"] = MyProvider
```

</details>

<details>
<summary><strong>Add a new tool source</strong></summary>

Register any async callable into `ToolRegistry`:

```python
from tools.registry import ToolRegistry, ToolSource

registry.register(
    name="my_tool",
    description="Does something useful",
    json_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    source=ToolSource.PYTHON,
    handler=my_async_fn,
)
```

</details>

<details>
<summary><strong>Add a new context source</strong></summary>

```python
from context.registry import ContextRegistry, ContextSource, RegisteredContext

async def fetch_my_context(**kwargs) -> str:
    return "relevant background information"

registry.register(RegisteredContext(
    name="my_context",
    source=ContextSource.RAG,
    fetch=fetch_my_context,
))
```

</details>

---

## 🖥️ CLI Reference

After `pip install`, the `uag` command is available globally.

```
uag serve            Start the HTTP gateway
uag chat "prompt"    Send a query directly (no server needed)
uag providers        List registered providers and config status
```

<details>
<summary><strong><code>uag serve</code></strong></summary>

```bash
uag serve                          # defaults: 0.0.0.0:8000
uag serve --port 3000 --reload     # dev mode
uag serve --workers 4              # production
```

</details>

<details>
<summary><strong><code>uag chat</code></strong></summary>

```bash
uag chat "What is 2+2?"                         # default profile
uag chat "Explain ML" --profile claude           # specific provider
uag chat "Write a poem" --stream                 # stream tokens live
uag chat "Summarise this" --json                 # raw JSON output
uag chat "Be brief" --system "You are terse."    # custom system prompt
```

</details>

<details>
<summary><strong><code>uag providers</code></strong></summary>

```bash
uag providers
# Prints a table of all providers, their adapter class, env key, and whether configured
```

</details>

---

## 🧪 Running Tests

```bash
# Full suite — no live API keys needed
make test

# With coverage
make test-cov

# Or directly with pytest
pytest -q -m "not integration"
```

200 tests, all passing, all offline.

---

## 📁 Project Structure

```
unified-agents-sdk/
├── api/               FastAPI HTTP/SSE endpoints + dynamic registry composition
├── config/            Pydantic-settings: providers, profiles, MCP & context presets
├── context/           Context registry, ContextForge adapter
├── core/              Normalized types, agent loop, durable execution primitives
├── docs/              Architecture reference and API specification
├── postman/           Postman collection (52 requests, 10 folders)
├── providers/         OpenAI, OpenAI Responses, Anthropic, Gemini, Groq, DeepSeek, Mistral, xAI adapters
├── runtime/           Router, profile resolution, bootstrap, SSE helpers
├── tests/             pytest test suite (200 tests, all offline)
├── tools/             Tool registry, MCP loader, inline MCP HTTP client
├── cli.py             Typer CLI (uag serve / chat / providers)
├── main.py            Application entry point (uvicorn)
├── Makefile           Dev task runner (make test, make lint, make serve, ...)
├── .env.example       Fully documented environment variable reference
├── py.typed           PEP 561 typed package marker
└── pyproject.toml     Package metadata, build config, ruff + pytest config
```

---

## 🗺️ Roadmap

| Version | What | Status |
|---------|------|--------|
| **v0.1.0** | Core gateway: 3 providers, tool loop, SSE, MCP, named presets, BYOK, 101 tests | ✅ Shipped |
| **v0.2.0** | Full provider coverage: 8 dedicated adapters (OpenAI, OpenAI Responses, Anthropic, Gemini, Groq, DeepSeek, Mistral, xAI/Grok), extended thinking/reasoning, server-side tools, multimodal I/O, citations, 200 tests | ✅ Shipped |
| **v0.3** | Auth middleware, rate limiting, request logging | 🔜 Planned |
| **v0.4** | Agent handoffs — native multi-agent delegation via `call_agent` meta-tool | 🔜 Planned |
| **v0.5** | Durable execution — resume interrupted runs, persistent step records | 💡 Exploring |
| **v0.6** | Provider marketplace — plug-in registry for community adapters | 💡 Exploring |
| **v1.0** | Production-grade — auth, permissions, audit logs, HA deployment guide | 💡 Exploring |

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, standards, and the PR process.

Ideas especially wanted:
- 🔌 **New provider adapters** — Cohere, Bedrock, Azure OpenAI
- 🛠️ **New tool sources** — gRPC, GraphQL, database queries
- 💉 **New context sources** — vector stores, custom KV, document stores
- 📖 **Documentation** — tutorials, examples, integration guides

---

## 🔒 Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

**Safe defaults:** API keys load from `.env` server-side. BYOK (`provider_credentials`) and dynamic runtime registration require explicit opt-in via environment flags.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Unified Agents SDK** — *One API. Every LLM. Any tool.*

<br>

[![Star this repo](https://img.shields.io/github/stars/PhilipAD/Unified-Agents-SDK?style=social)](https://github.com/PhilipAD/Unified-Agents-SDK)

</div>
