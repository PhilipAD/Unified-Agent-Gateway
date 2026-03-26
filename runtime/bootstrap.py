"""Startup bootstrap: builds ToolRegistry and ContextRegistry from settings.

Call ``bootstrap()`` once at application startup (e.g. inside FastAPI lifespan)
to wire MCP servers, context providers, and any built-in tools into the global
registries that every request uses by default.

Named presets defined in the environment (``MCP_SERVERS``, ``NAMED_CONTEXTS``)
are available per-request via the ``runtime.mcp_namespaces`` and
``runtime.context_names`` API fields — they do **not** have to be loaded
globally at startup.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from config.settings import (
    GatewaySettings,
    IntegrationSettings,
    NamedContextPreset,
    ProviderSettings,
)
from context.contextforge import register_contextforge
from context.registry import ContextRegistry, ContextSource, RegisteredContext
from tools.mcp_loader import MCPClient, load_mcp_tools_from_server
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _register_named_context(
    contexts: ContextRegistry,
    name: str,
    preset: NamedContextPreset,
) -> None:
    """Register a ``NamedContextPreset`` into a ``ContextRegistry``."""
    import httpx

    try:
        source = ContextSource(preset.source)
    except ValueError:
        source = ContextSource.STATIC

    if preset.mode == "static":

        async def fetch_static(_preset: NamedContextPreset = preset, **_kwargs: Any) -> str:
            return _preset.text

        contexts.register(
            RegisteredContext(
                name=name,
                source=source,
                fetch=fetch_static,
                required=preset.required,
                max_chars=preset.max_chars,
                metadata={"preset": True, "mode": "static"},
            )
        )
    else:
        if not preset.url:
            logger.warning("Named context '%s' has mode=http but no url — skipped", name)
            return

        async def fetch_http(_preset: NamedContextPreset = preset, **kwargs: Any) -> str:
            variables = {k: str(v) for k, v in kwargs.items()}

            def _render(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: _render(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_render(v) for v in obj]
                if isinstance(obj, str):
                    _fmt = type(
                        "_SD",
                        (dict,),
                        {"__missing__": lambda self, k: "{" + k + "}"},
                    )(**variables)
                    return obj.format_map(_fmt)
                return obj

            payload = _render(_preset.payload_template or {"input": "{input}"})
            async with httpx.AsyncClient() as client:
                url = _preset.url or ""
                hdrs = _preset.headers
                if _preset.method == "GET":
                    resp = await client.get(url, params=payload, headers=hdrs)
                else:
                    resp = await client.post(url, json=payload, headers=hdrs)
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "application/json" in ctype:
                    data = resp.json()
                    return str(data.get("context", data))
                return resp.text

        contexts.register(
            RegisteredContext(
                name=name,
                source=source,
                fetch=fetch_http,
                required=preset.required,
                max_chars=preset.max_chars,
                metadata={"preset": True, "mode": "http", "url": preset.url},
            )
        )


async def bootstrap(
    mcp_clients: Optional[List[Tuple[MCPClient, str]]] = None,
    integration_settings: Optional[IntegrationSettings] = None,
    gateway_settings: Optional[GatewaySettings] = None,
) -> Tuple[ToolRegistry, ContextRegistry]:
    """Build and return fully wired global registries.

    Parameters
    ----------
    mcp_clients:
        Optional pre-connected ``(client, namespace)`` pairs.  Pass these if
        you want to load certain MCP servers globally at startup instead of
        per-request.
    integration_settings:
        Loaded from env if not supplied.
    gateway_settings:
        Loaded from env if not supplied.  ``NAMED_CONTEXTS`` defined here are
        registered into the global ``ContextRegistry`` so every request sees
        them (unless ``use_global_contexts=False`` is set per-request).
    """
    int_settings = integration_settings or IntegrationSettings()
    gw_settings = gateway_settings or GatewaySettings()

    tools = ToolRegistry()
    contexts = ContextRegistry()

    # -- Pre-connected MCP clients (optional startup wiring) -----------------
    if mcp_clients:
        for client, namespace in mcp_clients:
            count = await load_mcp_tools_from_server(tools, client, namespace)
            logger.info("Loaded %d MCP tools from namespace '%s'", count, namespace)

    # -- ContextForge (legacy shorthand env vars) ----------------------------
    if int_settings.CONTEXTFORGE_URL and int_settings.CONTEXTFORGE_API_KEY:
        register_contextforge(
            contexts,
            base_url=int_settings.CONTEXTFORGE_URL,
            api_key=int_settings.CONTEXTFORGE_API_KEY,
        )
        logger.info("Registered ContextForge context source")

    # -- Named context presets from NAMED_CONTEXTS ---------------------------
    for name, preset in gw_settings.NAMED_CONTEXTS.items():
        _register_named_context(contexts, name, preset)
        logger.info("Registered named context '%s' (mode=%s)", name, preset.mode)

    return tools, contexts


async def bootstrap_and_configure_app() -> None:
    """Convenience wrapper that bootstraps and configures the FastAPI app state."""
    from api.http import configure

    provider_settings = ProviderSettings()
    gateway_settings = GatewaySettings()
    tools, contexts = await bootstrap(gateway_settings=gateway_settings)

    configure(
        tool_registry=tools,
        context_registry=contexts,
        provider_settings=provider_settings,
        gateway_settings=gateway_settings,
    )
