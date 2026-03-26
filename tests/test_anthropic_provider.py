from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition
from providers.anthropic import _to_anthropic_messages, _to_tools


def test_to_anthropic_messages_user():
    msgs = [NormalizedMessage(role=Role.USER, content="hello")]
    result = _to_anthropic_messages(msgs)
    assert result == [{"role": "user", "content": "hello"}]


def test_to_anthropic_messages_assistant_with_tool_use():
    tc = ToolCall(id="tu-1", name="search", arguments={"q": "test"})
    msgs = [
        NormalizedMessage(role=Role.ASSISTANT, content="let me search", tool_calls=[tc]),
    ]
    result = _to_anthropic_messages(msgs)
    assert result[0]["role"] == "assistant"
    blocks = result[0]["content"]
    assert blocks[0] == {"type": "text", "text": "let me search"}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["id"] == "tu-1"
    assert blocks[1]["name"] == "search"
    assert blocks[1]["input"] == {"q": "test"}


def test_to_anthropic_messages_tool_result():
    msgs = [
        NormalizedMessage(role=Role.TOOL, content="found 3 results", tool_call_id="tu-1"),
    ]
    result = _to_anthropic_messages(msgs)
    assert result[0]["role"] == "user"
    content_blocks = result[0]["content"]
    assert content_blocks[0]["type"] == "tool_result"
    assert content_blocks[0]["tool_use_id"] == "tu-1"
    assert content_blocks[0]["content"] == "found 3 results"


def test_to_anthropic_messages_system_skipped():
    msgs = [
        NormalizedMessage(role=Role.SYSTEM, content="sys"),
        NormalizedMessage(role=Role.USER, content="hi"),
    ]
    result = _to_anthropic_messages(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_to_tools():
    tools = [ToolDefinition(name="fn", description="desc", json_schema={"type": "object"})]
    result = _to_tools(tools)
    assert result[0]["name"] == "fn"
    assert result[0]["input_schema"] == {"type": "object"}


def test_to_tools_none():
    assert _to_tools(None) is None
