"""Gateway configuration.

All settings are loaded from environment variables (and an optional .env file).
Complex types (dicts of presets) are expressed as JSON strings in the env.

Quick reference
---------------
Provider selection::

    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_API_KEY=AI...

    # Point any agent profile at a specific provider + model
    AGENT_PROFILES={"default":{"provider_name":"openai_compatible"},
                    "claude":{"provider_name":"anthropic","model":"claude-opus-4-5"},
                    "fast":{"provider_name":"groq","model":"llama-3.3-70b-versatile"}}

    # Register extra OpenAI-compatible providers (DeepSeek, Groq, Together, Ollama...)
    OPENAI_COMPATIBLE_PROVIDERS={"groq":{"api_key":"gsk-...","base_url":"https://api.groq.com/openai/v1","model":"llama-3.3-70b-versatile"}}

Named MCP server presets (referenced by namespace in API calls)::

    MCP_SERVERS={"search":{"url":"http://search-mcp/mcp",
        "transport":"streamable_http",
        "headers":{"Authorization":"Bearer sk-xyz"}},
        "files":{"url":"http://files-mcp/sse","transport":"sse","timeout_seconds":60}}

Named context presets (referenced by name in API calls)::

    NAMED_CONTEXTS={"company_info":{"mode":"static","text":"We are Acme Corp."},
                    "product_faq":{"mode":"http","url":"http://kb.internal/search","payload_template":{"query":"{input}"}}}

    # Attach presets permanently to a profile so callers never need to specify them
    AGENT_PROFILES={"researcher":{"provider_name":"openai_compatible",
                                   "mcp_namespaces":["search"],
                                   "context_names":["company_info"]}}
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

_ENV_CFG = {
    "env_file": ".env",
    "env_file_encoding": "utf-8",
    "extra": "ignore",
}


# ---------------------------------------------------------------------------
# Sub-models (not settings classes — just Pydantic models used as field types)
# ---------------------------------------------------------------------------


class MCPServerPreset(BaseModel):
    """A named MCP server that can be referenced by namespace in API calls."""

    url: str
    transport: Literal["sse", "streamable_http"] = "streamable_http"
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 30.0


class NamedContextPreset(BaseModel):
    """A named context source that can be referenced by name in API calls."""

    mode: Literal["static", "http"] = "static"
    source: str = "static"
    text: str = ""
    url: Optional[str] = None
    method: Literal["GET", "POST"] = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    payload_template: Dict[str, Any] = Field(default_factory=dict)
    required: bool = False
    max_chars: Optional[int] = None


class OAICompatibleProviderPreset(BaseModel):
    """An extra OpenAI-compatible provider (DeepSeek, Groq, Together, Ollama…).

    Reference it by its key name as ``provider_name`` in an ``AgentProfile``.
    """

    api_key: str = ""
    base_url: str
    model: str


class AgentProfile(BaseModel):
    """Full per-profile configuration.

    Profiles are keyed by name in ``AGENT_PROFILES``.  The profile name is
    sent as the ``profile`` field in every agent API request.
    """

    provider_name: str = "openai_compatible"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    mcp_namespaces: List[str] = Field(
        default_factory=list,
        description="Named MCP servers (keys of MCP_SERVERS) always active for this profile",
    )
    context_names: List[str] = Field(
        default_factory=list,
        description="Named contexts (keys of NAMED_CONTEXTS) always active for this profile",
    )


# ---------------------------------------------------------------------------
# Settings classes (loaded from env / .env file)
# ---------------------------------------------------------------------------


class ProviderSettings(BaseSettings):
    """API keys and default models for the built-in providers."""

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    DEFAULT_OPENAI_MODEL: str = "gpt-4o"

    ANTHROPIC_API_KEY: Optional[str] = None
    DEFAULT_ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    GOOGLE_API_KEY: Optional[str] = None
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"

    model_config = _ENV_CFG


class IntegrationSettings(BaseSettings):
    """Shorthand env vars for ContextForge (legacy, still supported)."""

    CONTEXTFORGE_URL: Optional[str] = None
    CONTEXTFORGE_API_KEY: Optional[str] = None

    model_config = _ENV_CFG


class GatewaySettings(BaseSettings):
    """Top-level gateway and routing settings."""

    MODEL_TIMEOUT_SECONDS: float = 60.0
    TOOL_TIMEOUT_SECONDS: float = 30.0
    MAX_TOOL_HOPS: int = 6
    ALLOW_DYNAMIC_RUNTIME_REGISTRATION: bool = True

    # When false (default), reject any request that sends provider_credentials.
    # Enable only for trusted callers — keys in JSON bodies are logged by proxies.
    ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS: bool = False

    # --- Named presets (JSON env vars) ---

    MCP_SERVERS: Dict[str, MCPServerPreset] = Field(
        default_factory=dict,
        description="Named MCP server presets keyed by namespace",
    )

    NAMED_CONTEXTS: Dict[str, NamedContextPreset] = Field(
        default_factory=dict,
        description="Named context presets keyed by name",
    )

    OPENAI_COMPATIBLE_PROVIDERS: Dict[str, OAICompatibleProviderPreset] = Field(
        default_factory=dict,
        description="Extra OpenAI-compatible providers (DeepSeek, Groq, Ollama…)",
    )

    AGENT_PROFILES: Dict[str, AgentProfile] = Field(
        default_factory=lambda: {"default": AgentProfile()},
        description="Per-profile provider and preset configuration",
    )

    model_config = _ENV_CFG
