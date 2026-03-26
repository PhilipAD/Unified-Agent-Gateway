import json

from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition
from providers.openai_compatible import OpenAICompatibleProvider


def test_msg_to_api_user():
    p = OpenAICompatibleProvider(api_key="k", model="m")
    msg = NormalizedMessage(role=Role.USER, content="hi")
    api = p._msg_to_api(msg)
    assert api == {"role": "user", "content": "hi"}


def test_msg_to_api_assistant_with_tool_calls():
    p = OpenAICompatibleProvider(api_key="k", model="m")
    tc = ToolCall(id="tc-1", name="fn", arguments={"a": 1})
    msg = NormalizedMessage(role=Role.ASSISTANT, content="", tool_calls=[tc])
    api = p._msg_to_api(msg)
    assert api["tool_calls"][0]["id"] == "tc-1"
    assert api["tool_calls"][0]["function"]["name"] == "fn"
    assert json.loads(api["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


def test_msg_to_api_tool_result():
    p = OpenAICompatibleProvider(api_key="k", model="m")
    msg = NormalizedMessage(role=Role.TOOL, content="result", tool_call_id="tc-1", name="fn")
    api = p._msg_to_api(msg)
    assert api["role"] == "tool"
    assert api["tool_call_id"] == "tc-1"
    assert api["name"] == "fn"


def test_build_payload_with_tools():
    p = OpenAICompatibleProvider(api_key="k", model="test-model")
    msgs = [NormalizedMessage(role=Role.USER, content="q")]
    tools = [ToolDefinition(name="fn", description="d", json_schema={"type": "object"})]
    payload = p._build_payload(msgs, tools)
    assert payload["model"] == "test-model"
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["name"] == "fn"


def test_parse_tool_calls_string_arguments():
    raw = [
        {
            "id": "tc-1",
            "function": {"name": "fn", "arguments": '{"x": 1}'},
        }
    ]
    parsed = OpenAICompatibleProvider._parse_tool_calls(raw)
    assert parsed[0].arguments == {"x": 1}


def test_parse_tool_calls_empty_arguments():
    raw = [
        {
            "id": "tc-2",
            "function": {"name": "fn2", "arguments": ""},
        }
    ]
    parsed = OpenAICompatibleProvider._parse_tool_calls(raw)
    assert parsed[0].arguments == {}
