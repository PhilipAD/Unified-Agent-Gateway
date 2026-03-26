from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition
from providers.anthropic import (
    _convert_user_content_parts,
    _to_anthropic_messages,
    _to_tools,
)


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


def test_to_anthropic_messages_assistant_with_thinking():
    msgs = [
        NormalizedMessage(
            role=Role.ASSISTANT,
            content="The answer is 42",
            thinking_content="Let me reason about this carefully",
        ),
    ]
    result = _to_anthropic_messages(msgs)
    blocks = result[0]["content"]
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "Let me reason about this carefully"
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"] == "The answer is 42"


def test_to_anthropic_messages_user_with_image_url():
    msgs = [
        NormalizedMessage(
            role=Role.USER,
            content=[
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        ),
    ]
    result = _to_anthropic_messages(msgs)
    blocks = result[0]["content"]
    assert len(blocks) == 2
    assert blocks[0] == {"type": "text", "text": "What is in this image?"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["type"] == "url"
    assert blocks[1]["source"]["url"] == "https://example.com/image.png"


def test_to_anthropic_messages_user_with_base64_image():
    b64_url = "data:image/png;base64,iVBORw0KGgo="
    msgs = [
        NormalizedMessage(
            role=Role.USER,
            content=[
                {"type": "image_url", "image_url": {"url": b64_url}},
            ],
        ),
    ]
    result = _to_anthropic_messages(msgs)
    blocks = result[0]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[0]["source"]["media_type"] == "image/png"
    assert blocks[0]["source"]["data"] == "iVBORw0KGgo="


def test_to_anthropic_messages_user_with_document():
    msgs = [
        NormalizedMessage(
            role=Role.USER,
            content=[
                {"type": "text", "text": "Summarize this document"},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "data": "JVBERi0...",
                        "media_type": "application/pdf",
                    },
                },
            ],
        ),
    ]
    result = _to_anthropic_messages(msgs)
    blocks = result[0]["content"]
    assert len(blocks) == 2
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "document"
    assert blocks[1]["source"]["type"] == "base64"


def test_to_anthropic_messages_user_with_search_result():
    msgs = [
        NormalizedMessage(
            role=Role.USER,
            content=[
                {
                    "type": "search_result",
                    "content": [{"type": "text", "text": "relevant passage"}],
                    "source": "https://example.com",
                    "title": "Example",
                },
            ],
        ),
    ]
    result = _to_anthropic_messages(msgs)
    blocks = result[0]["content"]
    assert blocks[0]["type"] == "search_result"


def test_convert_user_content_parts_string():
    result = _convert_user_content_parts(["hello world"])
    assert result == [{"type": "text", "text": "hello world"}]


def test_convert_user_content_parts_text_dict():
    result = _convert_user_content_parts([{"type": "text", "text": "hi"}])
    assert result == [{"type": "text", "text": "hi"}]


def test_convert_user_content_parts_text_with_cache_control():
    result = _convert_user_content_parts([
        {"type": "text", "text": "cached text", "cache_control": {"type": "ephemeral"}}
    ])
    assert result[0]["cache_control"] == {"type": "ephemeral"}


def test_convert_user_content_parts_native_image_passthrough():
    native = {"type": "image", "source": {"type": "url", "url": "https://x.com/i.png"}}
    result = _convert_user_content_parts([native])
    assert result == [native]


def test_to_tools_basic():
    tools = [ToolDefinition(name="fn", description="desc", json_schema={"type": "object"})]
    result = _to_tools(tools)
    assert result[0]["name"] == "fn"
    assert result[0]["input_schema"] == {"type": "object"}


def test_to_tools_none():
    assert _to_tools(None) is None


def test_to_tools_with_server_tools():
    server_tools = [
        {"type": "web_search_20250305", "name": "web_search"},
        {
            "type": "code_execution_20250825",
            "name": "code_execution",
        },
    ]
    result = _to_tools(None, server_tools=server_tools)
    assert result is not None
    assert len(result) == 2
    assert result[0]["type"] == "web_search_20250305"
    assert result[1]["type"] == "code_execution_20250825"


def test_to_tools_combined_user_and_server_tools():
    tools = [ToolDefinition(name="fn", description="d", json_schema={"type": "object"})]
    server_tools = [{"type": "web_search_20250305", "name": "web_search"}]
    result = _to_tools(tools, server_tools=server_tools)
    assert result is not None
    assert len(result) == 2
    assert result[0]["name"] == "fn"
    assert result[1]["type"] == "web_search_20250305"
