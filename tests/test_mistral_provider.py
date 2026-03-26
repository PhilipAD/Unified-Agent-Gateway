from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition
from providers.mistral import (
    AGENT_TOOL_TYPES,
    MistralProvider,
    _convert_content_parts,
    _to_mistral_messages,
    _to_tools,
)


def _provider(**kw):
    defaults = {"api_key": "test-key", "model": "mistral-large-latest"}
    defaults.update(kw)
    return MistralProvider(**defaults)


def test_provider_init():
    p = _provider()
    assert p.name == "mistral"
    assert p.model == "mistral-large-latest"


def test_to_mistral_messages_user():
    msgs = [NormalizedMessage(role=Role.USER, content="hello")]
    result = _to_mistral_messages(msgs)
    assert result == [{"role": "user", "content": "hello"}]


def test_to_mistral_messages_system():
    msgs = [NormalizedMessage(role=Role.SYSTEM, content="be helpful")]
    result = _to_mistral_messages(msgs)
    assert result == [{"role": "system", "content": "be helpful"}]


def test_to_mistral_messages_assistant_with_tool_calls():
    tc = ToolCall(id="tc-1", name="fn", arguments={"a": 1})
    msgs = [
        NormalizedMessage(
            role=Role.ASSISTANT, content="calling", tool_calls=[tc],
        ),
    ]
    result = _to_mistral_messages(msgs)
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "calling"
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["id"] == "tc-1"
    assert result[0]["tool_calls"][0]["function"]["name"] == "fn"


def test_to_mistral_messages_tool_result():
    msgs = [
        NormalizedMessage(
            role=Role.TOOL,
            content="result data",
            tool_call_id="tc-1",
            name="fn",
        ),
    ]
    result = _to_mistral_messages(msgs)
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "tc-1"
    assert result[0]["name"] == "fn"


def test_to_mistral_messages_multimodal_user():
    msgs = [
        NormalizedMessage(
            role=Role.USER,
            content=[
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ],
        ),
    ]
    result = _to_mistral_messages(msgs)
    blocks = result[0]["content"]
    assert len(blocks) == 2
    assert blocks[0] == {"type": "text", "text": "Describe this image"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"] == "https://example.com/img.png"


def test_convert_content_parts_string():
    result = _convert_content_parts(["hello"])
    assert result == [{"type": "text", "text": "hello"}]


def test_convert_content_parts_document():
    doc = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
    result = _convert_content_parts([doc])
    assert result == [doc]


def test_convert_content_parts_file():
    f = {"type": "file", "file_id": "file-123"}
    result = _convert_content_parts([f])
    assert result == [f]


def test_convert_content_parts_audio():
    audio = {"type": "input_audio", "input_audio": "base64data"}
    result = _convert_content_parts([audio])
    assert result == [audio]


def test_to_tools():
    tools = [
        ToolDefinition(name="fn", description="d", json_schema={"type": "object"}),
    ]
    result = _to_tools(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "fn"


def test_to_tools_none():
    assert _to_tools(None) is None


def test_agent_tool_types():
    assert "web_search" in AGENT_TOOL_TYPES
    assert "code_interpreter" in AGENT_TOOL_TYPES
    assert "image_generation" in AGENT_TOOL_TYPES
    assert "document_library" in AGENT_TOOL_TYPES
    assert "connector" in AGENT_TOOL_TYPES
    assert "web_search_premium" in AGENT_TOOL_TYPES
