from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.http import app, configure
from config.settings import GatewaySettings, ProviderSettings
from context.registry import ContextRegistry
from core.types import NormalizedMessage, NormalizedResponse, Role
from tools.registry import ToolRegistry


class StubProvider:
    name = "stub"

    def __init__(self, *a, **kw):
        self.model = "stub-model"

    async def run(self, messages, tools=None, **kwargs):
        return NormalizedResponse(
            messages=[NormalizedMessage(role=Role.ASSISTANT, content="stub answer")],
            usage={"input_tokens": 1, "output_tokens": 1},
            provider="stub",
            model="stub-model",
        )

    async def stream(self, messages, tools=None, **kwargs):
        from core.types import StreamEvent

        yield StreamEvent(type="chunk", delta="stub ")
        yield StreamEvent(type="chunk", delta="answer")
        yield StreamEvent(type="done")


class ContextEchoProvider:
    name = "context_echo"

    def __init__(self, *a, **kw):
        self.model = "context-echo-model"

    async def run(self, messages, tools=None, **kwargs):
        system_chunks = [m.content for m in messages if m.role == Role.SYSTEM]
        joined = "\n---\n".join(system_chunks)
        return NormalizedResponse(
            messages=[NormalizedMessage(role=Role.ASSISTANT, content=joined)],
            usage={},
            provider="context_echo",
            model="context-echo-model",
        )

    async def stream(self, messages, tools=None, **kwargs):
        from core.types import StreamEvent

        yield StreamEvent(type="done")


@pytest.fixture(autouse=True)
def _configure_stub(monkeypatch):
    """Patch the router to always use StubProvider."""
    import runtime.router as router_mod

    original_providers = router_mod.PROVIDERS.copy()
    router_mod.PROVIDERS["stub"] = StubProvider
    router_mod.PROVIDERS["context_echo"] = ContextEchoProvider

    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(AGENT_PROFILES={"default": {"provider_name": "stub"}}),
    )
    yield
    router_mod.PROVIDERS = original_providers


@pytest.mark.asyncio
async def test_agent_query_returns_output():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={"input": "hello"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["output"] == "stub answer"
    assert data["provider"] == "stub"


@pytest.mark.asyncio
async def test_agent_query_stream_returns_sse():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query/stream",
            json={"input": "hello"},
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert "event: chunk" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_dynamic_context_registration_per_request():
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "context_echo"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={
                "input": "hello world",
                "runtime": {
                    "use_global_contexts": False,
                    "contexts": [
                        {
                            "name": "rag_inline",
                            "source": "rag",
                            "mode": "static",
                            "text": "RAG says: {input}",
                        }
                    ],
                },
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "context_echo"
    assert "Additional context:" in data["output"]
    assert "[rag_inline]" in data["output"]
    assert "RAG says: hello world" in data["output"]


@pytest.mark.asyncio
async def test_dynamic_registration_can_be_disabled():
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=False,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={
                "input": "hello",
                "runtime": {
                    "contexts": [
                        {
                            "name": "ctx",
                            "source": "static",
                            "mode": "static",
                            "text": "x",
                        }
                    ]
                },
            },
        )

    assert resp.status_code == 403
    assert "disabled" in resp.text


@pytest.mark.asyncio
async def test_provider_credentials_rejected_when_disabled():
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=False,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={
                "input": "hi",
                "provider_credentials": {"api_key": "sk-user-brought-key"},
            },
        )

    assert resp.status_code == 403
    assert "credentials" in resp.text.lower()


@pytest.mark.asyncio
async def test_provider_credentials_allowed_when_enabled():
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=True,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={
                "input": "hi",
                "provider_credentials": {
                    "api_key": "sk-ignored-by-stub",
                    "model": "custom-model-id",
                },
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["output"] == "stub answer"


@pytest.mark.asyncio
async def test_empty_provider_credentials_object_ignored_without_allow_check():
    """All-null provider_credentials does not trigger the allow gate."""
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_PER_REQUEST_PROVIDER_CREDENTIALS=False,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={"input": "hi", "provider_credentials": {}},
        )

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Inline MCP server tests
# ---------------------------------------------------------------------------


def _make_mock_mcp_tool(name="remote_tool", description="Remote tool"):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}
    return t


def _make_fake_inline_mcp_client(tools=None, call_result="mcp result"):
    """Return a class that behaves like InlineMCPClient but uses mocks."""
    mock_tools = tools or [_make_mock_mcp_tool()]

    class FakeMCPClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def list_tools(self):
            from tools.mcp_http_client import _MCPToolAdapter

            return [_MCPToolAdapter(t) for t in mock_tools]

        async def call_tool(self, name, arguments):
            return call_result

    return FakeMCPClient


@pytest.mark.asyncio
async def test_inline_mcp_server_tools_are_registered():
    """Tools from an inline MCP server are discovered and available to the agent."""
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
        ),
    )

    FakeMCPClient = _make_fake_inline_mcp_client(
        tools=[_make_mock_mcp_tool("weather", "Get weather")]
    )

    with patch("api.http.InlineMCPClient", FakeMCPClient):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/agent-query",
                json={
                    "input": "hi",
                    "runtime": {
                        "use_global_tools": False,
                        "mcp_servers": [
                            {
                                "url": "http://fake-mcp/mcp",
                                "namespace": "ext",
                                "transport": "streamable_http",
                            }
                        ],
                    },
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["output"] == "stub answer"
    assert data["warnings"] == []


@pytest.mark.asyncio
async def test_inline_mcp_server_connection_failure_becomes_warning():
    """A failing MCP connection adds a warning but does not fail the request."""
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
        ),
    )

    class FailingMCPClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            raise ConnectionRefusedError("no server")

        async def __aexit__(self, *args):
            pass

    with patch("api.http.InlineMCPClient", FailingMCPClient):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/agent-query",
                json={
                    "input": "hi",
                    "runtime": {
                        "mcp_servers": [
                            {
                                "url": "http://not-real/mcp",
                                "namespace": "broken",
                            }
                        ]
                    },
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert any("broken" in w or "not-real" in w for w in data["warnings"])


@pytest.mark.asyncio
async def test_inline_mcp_server_respects_disabled_dynamic_registration():
    """mcp_servers are blocked when ALLOW_DYNAMIC_RUNTIME_REGISTRATION=False."""
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=False,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={
                "input": "hi",
                "runtime": {
                    "mcp_servers": [
                        {
                            "url": "http://mcp/mcp",
                            "namespace": "ns",
                        }
                    ]
                },
            },
        )

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_inline_mcp_server_namespace_with_global_ns_prefix():
    """mcp_servers get the top-level namespace prefix applied."""
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
        ),
    )

    registered_names: list[str] = []

    class CapturingMCPClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def list_tools(self):
            from tools.mcp_http_client import _MCPToolAdapter

            t = _make_mock_mcp_tool("do_thing", "Does a thing")
            return [_MCPToolAdapter(t)]

        async def call_tool(self, name, arguments):
            return "done"

    original_register = ToolRegistry.register

    def capturing_register(self, name, **kwargs):
        registered_names.append(name)
        return original_register(self, name=name, **kwargs)

    with (
        patch("api.http.InlineMCPClient", CapturingMCPClient),
        patch.object(ToolRegistry, "register", capturing_register),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/agent-query",
                json={
                    "input": "hi",
                    "runtime": {
                        "namespace": "myapp",
                        "mcp_servers": [
                            {
                                "url": "http://fake/mcp",
                                "namespace": "search",
                            }
                        ],
                    },
                },
            )

    assert any("myapp.search.do_thing" in n for n in registered_names)


# ---------------------------------------------------------------------------
# Named preset tests (MCP_SERVERS / NAMED_CONTEXTS in GatewaySettings)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_named_mcp_preset_resolved_from_settings():
    """mcp_namespaces: ['search'] looks up MCP_SERVERS['search'] and connects."""
    from config.settings import MCPServerPreset

    FakeMCPClient = _make_fake_inline_mcp_client(tools=[_make_mock_mcp_tool("find", "Find things")])
    registered_names: list[str] = []
    original_register = ToolRegistry.register

    def capturing_register(self, name, **kwargs):
        registered_names.append(name)
        return original_register(self, name=name, **kwargs)

    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
            MCP_SERVERS={"search": MCPServerPreset(url="http://search-mcp/mcp")},
        ),
    )

    with (
        patch("api.http.InlineMCPClient", FakeMCPClient),
        patch.object(ToolRegistry, "register", capturing_register),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/agent-query",
                json={"input": "hi", "runtime": {"mcp_namespaces": ["search"]}},
            )

    assert resp.status_code == 200
    assert any("search.find" in n for n in registered_names)


@pytest.mark.asyncio
async def test_unknown_mcp_preset_becomes_warning():
    """A mcp_namespace that is not in MCP_SERVERS produces a warning."""
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
            MCP_SERVERS={},
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={"input": "hi", "runtime": {"mcp_namespaces": ["nonexistent"]}},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert any("nonexistent" in w for w in data["warnings"])


@pytest.mark.asyncio
async def test_named_context_preset_injected_from_settings():
    """context_names: ['company_info'] looks up NAMED_CONTEXTS and injects it."""
    from config.settings import NamedContextPreset

    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "context_echo"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
            NAMED_CONTEXTS={
                "company_info": NamedContextPreset(mode="static", text="We are Acme Corp")
            },
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={
                "input": "hi",
                "runtime": {
                    "use_global_contexts": False,
                    "context_names": ["company_info"],
                },
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "Acme Corp" in data["output"]


@pytest.mark.asyncio
async def test_unknown_context_name_becomes_warning():
    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={"default": {"provider_name": "stub"}},
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
            NAMED_CONTEXTS={},
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={"input": "hi", "runtime": {"context_names": ["missing"]}},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert any("missing" in w for w in data["warnings"])


@pytest.mark.asyncio
async def test_profile_mcp_namespaces_automatically_merged():
    """mcp_namespaces defined in a profile are applied without the caller specifying them."""
    from config.settings import AgentProfile, MCPServerPreset

    FakeMCPClient = _make_fake_inline_mcp_client(
        tools=[_make_mock_mcp_tool("profile_tool", "Tool from profile")]
    )
    registered_names: list[str] = []
    original_register = ToolRegistry.register

    def capturing_register(self, name, **kwargs):
        registered_names.append(name)
        return original_register(self, name=name, **kwargs)

    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={
                "default": {"provider_name": "stub"},
                "researcher": AgentProfile(
                    provider_name="stub",
                    mcp_namespaces=["search"],
                ),
            },
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
            MCP_SERVERS={"search": MCPServerPreset(url="http://search-mcp/mcp")},
        ),
    )

    with (
        patch("api.http.InlineMCPClient", FakeMCPClient),
        patch.object(ToolRegistry, "register", capturing_register),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # No "runtime" field at all — profile supplies the namespace
            resp = await client.post(
                "/agent-query",
                json={"input": "hi", "profile": "researcher"},
            )

    assert resp.status_code == 200
    assert any("search.profile_tool" in n for n in registered_names)


@pytest.mark.asyncio
async def test_profile_context_names_automatically_merged():
    """context_names defined in a profile are injected without the caller specifying them."""
    from config.settings import AgentProfile, NamedContextPreset

    configure(
        tool_registry=ToolRegistry(),
        context_registry=ContextRegistry(),
        provider_settings=ProviderSettings(),
        gateway_settings=GatewaySettings(
            AGENT_PROFILES={
                "default": {"provider_name": "stub"},
                "informed": AgentProfile(
                    provider_name="context_echo",
                    context_names=["brand"],
                ),
            },
            ALLOW_DYNAMIC_RUNTIME_REGISTRATION=True,
            NAMED_CONTEXTS={"brand": NamedContextPreset(mode="static", text="Brand: Acme")},
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/agent-query",
            json={
                "input": "hi",
                "profile": "informed",
                "runtime": {"use_global_contexts": False},
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "Brand: Acme" in data["output"]
