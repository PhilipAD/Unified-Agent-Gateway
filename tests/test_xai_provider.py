from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition
from providers.xai import (
    BUILT_IN_TOOL_TYPES,
    XAI_BASE_URL,
    XAIProvider,
    _to_input_items,
    _to_tools,
)


def _provider(**kw):
    defaults = {"api_key": "xai-test", "model": "grok-4-1-fast-reasoning"}
    defaults.update(kw)
    return XAIProvider(**defaults)


def test_provider_init():
    p = _provider()
    assert p.name == "xai"
    assert p.model == "grok-4-1-fast-reasoning"
    assert p.api_key == "xai-test"


def test_base_url_default():
    assert XAI_BASE_URL == "https://api.x.ai/v1"


def test_to_input_items_basic():
    msgs = [
        NormalizedMessage(role=Role.SYSTEM, content="You are Grok"),
        NormalizedMessage(role=Role.USER, content="hello"),
    ]
    instructions, items = _to_input_items(msgs)
    assert instructions == "You are Grok"
    assert len(items) == 1
    assert items[0] == {"role": "user", "content": "hello"}


def test_to_input_items_assistant_with_function_call():
    tc = ToolCall(id="call-1", name="search", arguments={"q": "test"})
    msgs = [
        NormalizedMessage(
            role=Role.ASSISTANT, content="searching", tool_calls=[tc],
        ),
    ]
    _, items = _to_input_items(msgs)
    assert len(items) == 2
    assert items[0]["type"] == "message"
    assert items[1]["type"] == "function_call"
    assert items[1]["name"] == "search"


def test_to_input_items_tool_output():
    msgs = [
        NormalizedMessage(
            role=Role.TOOL, content="result", tool_call_id="call-1",
        ),
    ]
    _, items = _to_input_items(msgs)
    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == "call-1"


def test_to_tools_functions():
    tools = [
        ToolDefinition(name="fn", description="d", json_schema={"type": "object"}),
    ]
    result = _to_tools(tools)
    assert result is not None
    assert result[0]["type"] == "function"
    assert result[0]["name"] == "fn"


def test_to_tools_with_built_in():
    built_in = [{"type": "web_search"}, {"type": "x_search"}]
    result = _to_tools(None, built_in_tools=built_in)
    assert result is not None
    assert len(result) == 2
    assert result[0]["type"] == "web_search"
    assert result[1]["type"] == "x_search"


def test_to_tools_with_mcp():
    mcp = [
        {
            "server_url": "https://mcp.example.com",
            "server_label": "test",
            "authorization": "Bearer tok",
        },
    ]
    result = _to_tools(None, mcp_servers=mcp)
    assert result is not None
    assert result[0]["type"] == "mcp"
    assert result[0]["server_url"] == "https://mcp.example.com"
    assert result[0]["authorization"] == "Bearer tok"


def test_to_tools_none():
    result = _to_tools(None)
    assert result is None


def test_to_tools_combined():
    tools = [
        ToolDefinition(name="fn", description="d", json_schema={"type": "object"}),
    ]
    built_in = [{"type": "x_search"}]
    mcp = [{"server_url": "https://mcp.example.com", "server_label": "ex"}]
    result = _to_tools(tools, built_in_tools=built_in, mcp_servers=mcp)
    assert result is not None
    assert len(result) == 3
    types = [r["type"] for r in result]
    assert "function" in types
    assert "x_search" in types
    assert "mcp" in types


def test_built_in_tool_types():
    assert "web_search" in BUILT_IN_TOOL_TYPES
    assert "x_search" in BUILT_IN_TOOL_TYPES
    assert "code_interpreter" in BUILT_IN_TOOL_TYPES
    assert "code_execution" in BUILT_IN_TOOL_TYPES
    assert "file_search" in BUILT_IN_TOOL_TYPES
    assert "collections_search" in BUILT_IN_TOOL_TYPES
    assert "attachment_search" in BUILT_IN_TOOL_TYPES
    assert "mcp" in BUILT_IN_TOOL_TYPES
