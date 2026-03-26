# Unified Agent Gateway -- API Specification

## Base URL

```
http://localhost:8000
```

---

## POST /agent-query

Synchronous agent invocation.  Runs the full tool-calling loop and returns
the final answer.

### Request

```json
{
  "input": "What is the weather in London?",
  "context": {
    "system_prompt": "You are a helpful assistant.",
    "user_id": "u-123"
  },
  "agent_id": "default",
  "profile": "default",
  "options": {}
}
```

| Field                   | Type            | Required | Description                              |
|-------------------------|-----------------|----------|------------------------------------------|
| input                   | string          | yes      | The user query / instruction             |
| context                 | object          | no       | Arbitrary context (`system_prompt`, template vars, etc.) |
| agent_id                | string          | no       | Agent identifier for routing (`agent_id:profile` lookup) |
| profile                 | string          | no       | Profile name for routing (default `default`) |
| options                 | object          | no       | Extra kwargs forwarded to provider (temperature, max_tokens, …) |
| runtime                 | object          | no       | Per-request tools / contexts / MCP — see **Runtime** below |
| provider_credentials    | object          | no       | BYOK: `api_key`, optional `model`, optional `base_url` (OpenAI-compatible). Requires `ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=true` |

### Runtime (`runtime` object)

Omitted or empty means: use global tool/context registries only (from bootstrap + `.env`).

| Field                  | Type    | Default | Description |
|------------------------|---------|---------|-------------|
| use_global_tools       | bool    | `true`  | Start from a copy of globally registered tools |
| use_global_contexts    | bool    | `true`  | Start from a copy of globally registered contexts |
| namespace              | string  | —       | Prefix for every tool/context name registered in this request (e.g. `tenant_a.search.tool`) |
| mcp_namespaces         | string[]| `[]`    | Keys from `MCP_SERVERS` in `.env` — connect each MCP server per request |
| context_names          | string[]| `[]`    | Keys from `NAMED_CONTEXTS` in `.env` — inject each named context |
| mcp_servers            | object[]| `[]`    | Inline MCP servers: `url`, `namespace`, `transport` (`streamable_http` \| `sse`), optional `headers`, `timeout_seconds` |
| tools                  | object[]| `[]`    | Inline HTTP tools: `name`, `description`, `json_schema`, `url`, `method`, `headers`, `timeout_seconds`, `argument_mode` (`json` \| `query`) |
| contexts               | object[]| `[]`    | Inline contexts: `name`, `source`, `mode` (`static` \| `http`), `text`, `url`, `method`, `payload_template`, etc. |

Profile-level `mcp_namespaces` and `context_names` from `AGENT_PROFILES` are **merged** with `runtime` (profile first, then request; duplicates deduped).

If `runtime` includes any inline tools, MCP servers, named MCP/context references, or inline contexts, `ALLOW_DYNAMIC_RUNTIME_REGISTRATION` must be `true` or the gateway returns **403**.

### Provider credentials (`provider_credentials` object)

| Field     | Type   | Description |
|-----------|--------|-------------|
| api_key   | string | Overrides resolved API key for this request |
| model     | string | Overrides resolved model id |
| base_url  | string | Overrides base URL for **OpenAI-compatible** adapter only |

Sending any non-null field requires `ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=true` (403 otherwise). An empty `{}` is allowed and does not trigger the gate.

### Response (200)

```json
{
  "output": "The weather in London is 15C and cloudy.",
  "tool_traces": [],
  "usage": {"input_tokens": 42, "output_tokens": 18},
  "provider": "openai_compatible",
  "model": "gpt-4o",
  "warnings": [],
  "errors": []
}
```

| Field        | Type          | Description                                  |
|--------------|---------------|----------------------------------------------|
| output       | string        | Final assistant response text                |
| tool_traces  | array         | Normalized traces of tool calls executed     |
| usage        | object        | Token usage from the provider                |
| provider     | string|null   | Provider name used                           |
| model        | string|null   | Model identifier used                        |
| warnings     | array[string] | Non-fatal warnings                           |
| errors       | array[string] | Error messages (non-empty on failure)        |

### Error Responses

* **403** — Dynamic runtime registration disabled (`runtime` tools/MCP/contexts when `ALLOW_DYNAMIC_RUNTIME_REGISTRATION=false`), or per-request credentials disabled (`provider_credentials` when `ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=false`).
* **422** — Validation error (malformed request body).
* **4xx/5xx** — Provider or tool errors may be returned as JSON with `errors` populated, or as streamed `error` SSE events on `/agent-query/stream`.

---

## POST /agent-query/stream

Streaming agent invocation via Server-Sent Events (SSE).

### Request

Same schema as `/agent-query`.

### SSE Event Types

Each SSE frame has an `event:` line and a `data:` line containing JSON.

| Event     | Data Fields              | Description                        |
|-----------|--------------------------|------------------------------------|
| chunk     | `{"type":"chunk","delta":"..."}` | Incremental text from the model |
| tool_call | `{"type":"tool_call","tool_call":{...}}` | Completed tool call      |
| usage     | `{"type":"usage","usage":{...}}`         | Token usage summary      |
| done      | `{"type":"done"}`                        | Stream complete          |
| warning   | `{"warnings":[...]}`                     | Non-fatal issues (e.g. MCP skipped) |
| error     | `{"message":"..."}`                      | Error during streaming   |

### Example

```
event: chunk
data: {"type":"chunk","delta":"The weather"}

event: chunk
data: {"type":"chunk","delta":" in London is 15C."}

event: usage
data: {"type":"usage","usage":{"input_tokens":42,"output_tokens":18}}

event: done
data: {"type":"done"}
```

### Headers

Response includes:

```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

---

## Downstream Adapter Pattern

The gateway returns a generic envelope.  Consuming backends transform it
into their own domain schema.  Example adapter (not part of gateway):

```python
def to_sa_response(gw_resp: dict) -> dict:
    return {
        "question": original_question,
        "answer": gw_resp["output"],
        "data": [],
        "totals": None,
        "confidence": 0.85,
        "metadata": {"provider": gw_resp["provider"]},
        "warnings": gw_resp["warnings"],
        "errors": gw_resp["errors"],
    }
```

---

## Configuration

All settings are loaded from environment variables (see `.env.example`).

| Variable                | Default                     | Description                    |
|-------------------------|-----------------------------|--------------------------------|
| OPENAI_API_KEY          | --                          | OpenAI / compatible API key    |
| OPENAI_BASE_URL         | https://api.openai.com/v1   | Override for compatible APIs   |
| ANTHROPIC_API_KEY       | --                          | Anthropic API key              |
| GOOGLE_API_KEY          | --                          | Google Gemini API key          |
| DEFAULT_OPENAI_MODEL    | gpt-4o                      | Default OpenAI model           |
| DEFAULT_ANTHROPIC_MODEL | claude-sonnet-4-20250514    | Default Anthropic model        |
| DEFAULT_GEMINI_MODEL    | gemini-2.5-flash            | Default Gemini model           |
| MAX_TOOL_HOPS           | 6                           | Max tool-calling iterations    |
| MODEL_TIMEOUT_SECONDS   | 60                          | Timeout for model API calls    |
| TOOL_TIMEOUT_SECONDS    | 30                          | Timeout for tool execution     |
| ALLOW_DYNAMIC_RUNTIME_REGISTRATION | true (default)     | Allow `runtime` overrides in API body |
| ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS | false (default) | Allow `provider_credentials` in API body |
| AGENT_PROFILES          | (see `.env.example`)        | JSON: named profiles with provider, model, `mcp_namespaces`, `context_names` |
| MCP_SERVERS             | `{}`                        | JSON: named MCP presets (`url`, `transport`, `headers`) |
| NAMED_CONTEXTS          | `{}`                        | JSON: named context presets |
| OPENAI_COMPATIBLE_PROVIDERS | `{}`                    | JSON: extra OAI-compatible backends (Groq, DeepSeek, …) |
