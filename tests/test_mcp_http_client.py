"""Tests for tools/mcp_http_client.py.

We mock the MCP SDK internals so no live MCP server is required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_http_client import InlineMCPClient, _extract_text_from_content, _MCPToolAdapter

# ---------------------------------------------------------------------------
# _extract_text_from_content
# ---------------------------------------------------------------------------


def test_extract_text_from_none():
    assert _extract_text_from_content(None) == ""


def test_extract_text_from_string():
    assert _extract_text_from_content("hello") == "hello"


def test_extract_text_from_text_block_object():
    block = MagicMock()
    block.text = "block text"
    assert _extract_text_from_content(block) == "block text"


def test_extract_text_from_list_of_blocks():
    b1 = MagicMock()
    b1.text = "first"
    b2 = MagicMock()
    b2.text = "second"
    assert _extract_text_from_content([b1, b2]) == "first\nsecond"


def test_extract_text_from_list_of_dicts():
    blocks = [{"text": "from dict"}]
    assert _extract_text_from_content(blocks) == "from dict"


# ---------------------------------------------------------------------------
# _MCPToolAdapter
# ---------------------------------------------------------------------------


def test_mcp_tool_adapter_with_dict_schema():
    raw = MagicMock()
    raw.name = "search"
    raw.description = "Search the web"
    raw.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}
    adapter = _MCPToolAdapter(raw)
    assert adapter.name == "search"
    assert adapter.description == "Search the web"
    assert adapter.schema["properties"]["q"] == {"type": "string"}


def test_mcp_tool_adapter_no_schema_defaults():
    raw = MagicMock()
    raw.name = "noop"
    raw.description = ""
    raw.inputSchema = None
    raw.input_schema = None
    adapter = _MCPToolAdapter(raw)
    assert adapter.schema == {"type": "object"}


def test_mcp_tool_adapter_pydantic_schema():
    raw = MagicMock()
    raw.name = "thing"
    raw.description = "desc"
    pydantic_model = MagicMock()
    pydantic_model.model_dump.return_value = {"type": "object", "properties": {}}
    raw.inputSchema = pydantic_model
    adapter = _MCPToolAdapter(raw)
    assert adapter.schema == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# InlineMCPClient — context manager and tool calls (mocked transport)
# ---------------------------------------------------------------------------


def _make_mock_tool(name="my_tool", description="A tool", schema=None):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object"}
    return t


def _make_mock_session(tools=None, call_result_text="ok"):
    session = MagicMock()
    tools = tools or [_make_mock_tool()]

    list_result = MagicMock()
    list_result.tools = tools
    session.list_tools = AsyncMock(return_value=list_result)

    call_result = MagicMock()
    content_block = MagicMock()
    content_block.text = call_result_text
    call_result.content = [content_block]
    call_result.isError = False
    session.call_tool = AsyncMock(return_value=call_result)

    session.initialize = AsyncMock()
    return session


class _FakeTransportCM:
    """Yields (read, write); mocks sse_client / streamablehttp_client."""

    def __init__(self, triple=False):
        self._triple = triple  # streamable_http yields 3-tuple

    async def __aenter__(self):
        if self._triple:
            return (MagicMock(), MagicMock(), lambda: None)
        return (MagicMock(), MagicMock())

    async def __aexit__(self, *args):
        pass


class _FakeSessionCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_inline_mcp_client_streamable_http_list_tools():
    mock_session = _make_mock_session(tools=[_make_mock_tool("calc", "Calculator")])

    with (
        patch(
            "tools.mcp_http_client.streamablehttp_client",
            return_value=_FakeTransportCM(triple=True),
        ),
        patch(
            "tools.mcp_http_client.ClientSession",
            return_value=_FakeSessionCM(mock_session),
        ),
    ):
        async with InlineMCPClient("http://mcp/mcp", transport="streamable_http") as client:
            tools = await client.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "calc"
    assert tools[0].description == "Calculator"


@pytest.mark.asyncio
async def test_inline_mcp_client_sse_list_tools():
    mock_session = _make_mock_session(tools=[_make_mock_tool("writer", "Write stuff")])

    with (
        patch(
            "tools.mcp_http_client.sse_client",
            return_value=_FakeTransportCM(triple=False),
        ),
        patch(
            "tools.mcp_http_client.ClientSession",
            return_value=_FakeSessionCM(mock_session),
        ),
    ):
        async with InlineMCPClient("http://mcp/sse", transport="sse") as client:
            tools = await client.list_tools()

    assert tools[0].name == "writer"


@pytest.mark.asyncio
async def test_inline_mcp_client_call_tool():
    mock_session = _make_mock_session(call_result_text="42")

    with (
        patch(
            "tools.mcp_http_client.streamablehttp_client",
            return_value=_FakeTransportCM(triple=True),
        ),
        patch(
            "tools.mcp_http_client.ClientSession",
            return_value=_FakeSessionCM(mock_session),
        ),
    ):
        async with InlineMCPClient("http://mcp/mcp") as client:
            result = await client.call_tool("add", {"a": 1, "b": 2})

    assert result == "42"
    mock_session.call_tool.assert_called_once_with("add", {"a": 1, "b": 2})


@pytest.mark.asyncio
async def test_inline_mcp_client_call_tool_error_raises():
    mock_session = _make_mock_session()
    error_result = MagicMock()
    error_result.isError = True
    content_block = MagicMock()
    content_block.text = "tool failed badly"
    error_result.content = [content_block]
    mock_session.call_tool = AsyncMock(return_value=error_result)

    with (
        patch(
            "tools.mcp_http_client.streamablehttp_client",
            return_value=_FakeTransportCM(triple=True),
        ),
        patch(
            "tools.mcp_http_client.ClientSession",
            return_value=_FakeSessionCM(mock_session),
        ),
    ):
        async with InlineMCPClient("http://mcp/mcp") as client:
            with pytest.raises(RuntimeError, match="tool failed badly"):
                await client.call_tool("bad_tool", {})


@pytest.mark.asyncio
async def test_inline_mcp_client_connect_failure_raises():
    class _FailingCM:
        async def __aenter__(self):
            raise ConnectionRefusedError("refused")

        async def __aexit__(self, *args):
            pass

    with patch(
        "tools.mcp_http_client.streamablehttp_client",
        return_value=_FailingCM(),
    ):
        with pytest.raises(RuntimeError, match="Failed to connect"):
            async with InlineMCPClient("http://mcp/mcp"):
                pass


@pytest.mark.asyncio
async def test_list_tools_without_connect_raises():
    client = InlineMCPClient("http://mcp/mcp")
    with pytest.raises(RuntimeError, match="not connected"):
        await client.list_tools()


@pytest.mark.asyncio
async def test_call_tool_without_connect_raises():
    client = InlineMCPClient("http://mcp/mcp")
    with pytest.raises(RuntimeError, match="not connected"):
        await client.call_tool("foo", {})
