from core.types import (
    NormalizedMessage,
    NormalizedResponse,
    Role,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)


def test_normalized_message_to_dict_basic():
    msg = NormalizedMessage(role=Role.USER, content="hello")
    d = msg.to_dict()
    assert d == {"role": "user", "content": "hello"}


def test_normalized_message_to_dict_with_tool_calls():
    tc = ToolCall(id="tc-1", name="lookup", arguments={"q": "test"})
    msg = NormalizedMessage(role=Role.ASSISTANT, content="", tool_calls=[tc])
    d = msg.to_dict()
    assert d["tool_calls"] == [{"id": "tc-1", "name": "lookup", "arguments": {"q": "test"}}]


def test_normalized_message_to_dict_tool_result():
    msg = NormalizedMessage(role=Role.TOOL, content="42", tool_call_id="tc-1", name="lookup")
    d = msg.to_dict()
    assert d["tool_call_id"] == "tc-1"
    assert d["name"] == "lookup"


def test_tool_definition_to_dict():
    td = ToolDefinition(name="fn", description="desc", json_schema={"type": "object"})
    assert td.to_dict()["name"] == "fn"


def test_stream_event_chunk():
    ev = StreamEvent(type="chunk", delta="hi")
    d = ev.to_dict()
    assert d == {"type": "chunk", "delta": "hi"}


def test_stream_event_tool_call():
    tc = ToolCall(id="tc-1", name="fn", arguments={})
    ev = StreamEvent(type="tool_call", tool_call=tc)
    d = ev.to_dict()
    assert d["tool_call"]["id"] == "tc-1"


def test_normalized_response_to_dict():
    msg = NormalizedMessage(role=Role.ASSISTANT, content="ok")
    resp = NormalizedResponse(messages=[msg], usage={"input_tokens": 5}, provider="test")
    d = resp.to_dict()
    assert d["provider"] == "test"
    assert d["usage"]["input_tokens"] == 5
