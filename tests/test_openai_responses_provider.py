from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition
from providers.openai_responses import _to_input_items, _to_tools


def test_to_input_items_basic():
    msgs = [
        NormalizedMessage(role=Role.SYSTEM, content="You are helpful"),
        NormalizedMessage(role=Role.USER, content="hello"),
    ]
    instructions, items = _to_input_items(msgs)
    assert instructions == "You are helpful"
    assert len(items) == 1
    assert items[0] == {"role": "user", "content": "hello"}


def test_to_input_items_no_system():
    msgs = [NormalizedMessage(role=Role.USER, content="hi")]
    instructions, items = _to_input_items(msgs)
    assert instructions is None
    assert len(items) == 1


def test_to_input_items_assistant_with_function_call():
    tc = ToolCall(id="call-1", name="search", arguments={"q": "test"})
    msgs = [
        NormalizedMessage(role=Role.ASSISTANT, content="searching", tool_calls=[tc]),
    ]
    _, items = _to_input_items(msgs)
    assert len(items) == 2
    assert items[0]["type"] == "message"
    assert items[0]["role"] == "assistant"
    assert items[0]["content"][0]["type"] == "output_text"
    assert items[0]["content"][0]["text"] == "searching"
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call-1"
    assert items[1]["name"] == "search"


def test_to_input_items_tool_output():
    msgs = [
        NormalizedMessage(role=Role.TOOL, content="result data", tool_call_id="call-1"),
    ]
    _, items = _to_input_items(msgs)
    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == "call-1"
    assert items[0]["output"] == "result data"


def test_to_tools_functions_only():
    tools = [ToolDefinition(name="fn", description="desc", json_schema={"type": "object"})]
    result = _to_tools(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["name"] == "fn"
    assert result[0]["strict"] is False


def test_to_tools_with_built_in():
    built_in = [{"type": "web_search"}]
    result = _to_tools(None, built_in_tools=built_in)
    assert result is not None
    assert result[0]["type"] == "web_search"


def test_to_tools_with_mcp_servers():
    mcp = [
        {
            "server_url": "https://mcp.example.com/sse",
            "server_label": "example",
            "server_description": "An MCP server",
            "require_approval": "never",
        }
    ]
    result = _to_tools(None, mcp_servers=mcp)
    assert result is not None
    assert result[0]["type"] == "mcp"
    assert result[0]["server_url"] == "https://mcp.example.com/sse"
    assert result[0]["server_label"] == "example"


def test_to_tools_with_mcp_connector_id():
    mcp = [
        {
            "server_label": "gmail",
            "connector_id": "connector_gmail",
            "authorization": "Bearer token-123",
        }
    ]
    result = _to_tools(None, mcp_servers=mcp)
    assert result is not None
    assert result[0]["type"] == "mcp"
    assert result[0]["connector_id"] == "connector_gmail"
    assert result[0]["authorization"] == "Bearer token-123"
    assert "server_url" not in result[0]


def test_to_tools_with_mcp_all_fields():
    mcp = [
        {
            "server_url": "https://mcp.example.com",
            "server_label": "full",
            "headers": {"X-Key": "val"},
            "authorization": "Bearer tok",
            "allowed_tools": ["tool_a", "tool_b"],
            "defer_loading": True,
            "server_description": "Full config",
            "require_approval": "always",
        }
    ]
    result = _to_tools(None, mcp_servers=mcp)
    assert result is not None
    entry = result[0]
    assert entry["server_url"] == "https://mcp.example.com"
    assert entry["headers"] == {"X-Key": "val"}
    assert entry["authorization"] == "Bearer tok"
    assert entry["allowed_tools"] == ["tool_a", "tool_b"]
    assert entry["defer_loading"] is True
    assert entry["server_description"] == "Full config"
    assert entry["require_approval"] == "always"


def test_to_tools_combined():
    tools = [ToolDefinition(name="fn", description="d", json_schema={"type": "object"})]
    built_in = [{"type": "code_interpreter"}]
    mcp = [{"server_url": "https://mcp.example.com", "server_label": "ex"}]
    result = _to_tools(tools, built_in_tools=built_in, mcp_servers=mcp)
    assert result is not None
    assert len(result) == 3
    types = [r["type"] for r in result]
    assert "function" in types
    assert "code_interpreter" in types
    assert "mcp" in types


def test_to_tools_returns_none_when_empty():
    result = _to_tools(None)
    assert result is None


def test_provider_init():
    from providers.openai_responses import OpenAIResponsesProvider

    p = OpenAIResponsesProvider(api_key="sk-test", model="gpt-4o")
    assert p.name == "openai_responses"
    assert p.api_key == "sk-test"
    assert p.model == "gpt-4o"
