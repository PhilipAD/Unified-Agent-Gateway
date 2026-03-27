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
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool allow/deny lists and other bridge-specific fields",
    )


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
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific kwargs merged into provider constructor and run()",
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

    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: Optional[str] = None
    DEFAULT_GROQ_MODEL: str = "llama-3.3-70b-versatile"

    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: Optional[str] = None
    DEFAULT_DEEPSEEK_MODEL: str = "deepseek-chat"

    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_BASE_URL: Optional[str] = None
    DEFAULT_MISTRAL_MODEL: str = "mistral-large-latest"

    XAI_API_KEY: Optional[str] = None
    XAI_BASE_URL: Optional[str] = None
    DEFAULT_XAI_MODEL: str = "grok-4-1-fast-reasoning"

    CURSOR_API_KEY: Optional[str] = None
    DEFAULT_CURSOR_MODEL: str = "default"

    CODEX_API_KEY: Optional[str] = None
    DEFAULT_CODEX_MODEL: str = "codex-mini-latest"

    COPILOT_GITHUB_TOKEN: Optional[str] = None
    DEFAULT_COPILOT_MODEL: str = "default"

    model_config = _ENV_CFG


class IntegrationSettings(BaseSettings):
    """Shorthand env vars for ContextForge (legacy, still supported)."""

    CONTEXTFORGE_URL: Optional[str] = None
    CONTEXTFORGE_API_KEY: Optional[str] = None

    model_config = _ENV_CFG


class AgentHarnessSettings(BaseSettings):
    """Feature flags and paths for curated agent harness integrations."""

    # ---------------------------------------------------------------------------
    # Generic / user-defined custom MD loader
    # Lets operators load any project-specific .md file(s) without writing code.
    # ---------------------------------------------------------------------------
    CUSTOM_MD_ENABLED: bool = False
    CUSTOM_MD_CWD: str = "."
    CUSTOM_MD_FILENAMES: List[str] = Field(
        default_factory=list,
        description=(
            "File names to search for when CUSTOM_MD_ENABLED is true "
            "(e.g. MY_RULES.md, TEAM_STANDARDS.md)"
        ),
    )
    CUSTOM_MD_SYSTEM_DIRS: List[str] = Field(
        default_factory=list,
        description="System-level directories scanned before the cwd walk",
    )
    CUSTOM_MD_USER_DIRS: List[str] = Field(
        default_factory=list,
        description="User-level directories scanned before the cwd walk (~ expanded)",
    )
    CUSTOM_MD_STOP_AT_GIT_ROOT: bool = True
    CUSTOM_MD_RESOLVE_IMPORTS: bool = True
    CUSTOM_MD_MAX_CHARS: Optional[int] = None

    AGENTS_MD_ENABLED: bool = False
    AGENTS_MD_CWD: str = "."

    GEMINI_CLI_MD_ENABLED: bool = False
    GEMINI_CLI_MD_CWD: str = "."
    GEMINI_CLI_MD_FILENAMES: List[str] = Field(
        default_factory=lambda: ["GEMINI.md", "AGENTS.md", "CLAUDE.md"]
    )
    GEMINI_CLI_MD_STRIP_AUTO_MEMORY: bool = False
    GEMINI_CLI_MD_MAX_CHARS: Optional[int] = None
    GEMINI_CLI_SYSTEM_CONFIG_DIR: Optional[str] = None
    GEMINI_CLI_SKILLS_ENABLED: bool = False
    GEMINI_CLI_SKILLS_WORKSPACE_DIR: str = "."
    GEMINI_CLI_MCP_BRIDGE: bool = False
    GEMINI_CLI_MCP_WORKSPACE_DIR: str = "."

    WINDSURF_RULES_ENABLED: bool = False
    WINDSURF_RULES_WORKSPACE_DIR: str = "."
    WINDSURF_MCP_BRIDGE: bool = False
    WINDSURF_MCP_CONFIG_PATH: Optional[str] = None
    WINDSURF_ANALYTICS_SERVICE_KEY: Optional[str] = None

    CLINE_RULES_ENABLED: bool = False
    CLINE_RULES_WORKSPACE_DIR: str = "."

    CODEX_PROVIDER: str = "openai"
    CODEX_APPROVAL_POLICY: str = "on-request"
    CODEX_SANDBOX_MODE: str = "workspace-write"
    CODEX_REASONING_EFFORT: str = "medium"
    CODEX_NO_PROJECT_DOC: bool = False
    CODEX_MCP_ENABLED: bool = False
    CODEX_USE_APP_SERVER: bool = False
    CODEX_BINARY: str = "codex"

    COPILOT_MCP_BRIDGE: bool = False
    COPILOT_MCP_TRANSPORT: str = "remote"
    COPILOT_MCP_TOOLSETS: Optional[List[str]] = None
    COPILOT_MCP_URL: str = "https://api.github.com/copilot/mcp"
    GITHUB_MCP_URL: Optional[str] = None

    CLAUDE_AGENT_DEFAULT_PERMISSION_MODE: str = "acceptEdits"
    CLAUDE_AGENT_DEFAULT_EFFORT: Optional[str] = None
    CLAUDE_AGENT_SETTING_SOURCES: List[str] = Field(default_factory=list)
    CLAUDE_AGENT_ENABLE_FILE_CHECKPOINTING: bool = False
    CLAUDE_AGENT_USE_CLIENT: bool = False

    CURSOR_WEBHOOK_SECRET: Optional[str] = None
    CURSOR_POLL_INTERVAL_SECONDS: float = 15.0
    CURSOR_MAX_WAIT_SECONDS: float = 600.0
    CURSOR_DEFAULT_REPOSITORY: Optional[str] = None
    CURSOR_DEFAULT_REF: str = "main"
    CURSOR_AUTO_CREATE_PR: bool = False

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
