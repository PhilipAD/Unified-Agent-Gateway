import json

from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition
from providers.groq import COMPOUND_MODELS, VALID_BUILTIN_TOOLS, GroqProvider


def _provider(**kw):
    defaults = {"api_key": "gsk-test", "model": "llama-3.3-70b-versatile"}
    defaults.update(kw)
    return GroqProvider(**defaults)


def test_provider_init():
    p = _provider()
    assert p.name == "groq"
    assert p.api_key == "gsk-test"


def test_is_compound_true():
    p = _provider(model="compound-beta")
    assert p._is_compound() is True


def test_is_compound_mini_true():
    p = _provider(model="compound-beta-mini")
    assert p._is_compound() is True


def test_is_compound_false():
    p = _provider(model="llama-3.3-70b-versatile")
    assert p._is_compound() is False


def test_msg_to_api_user():
    p = _provider()
    msg = NormalizedMessage(role=Role.USER, content="hi")
    api = p._msg_to_api(msg)
    assert api == {"role": "user", "content": "hi"}


def test_msg_to_api_assistant_with_tool_calls():
    p = _provider()
    tc = ToolCall(id="tc-1", name="fn", arguments={"a": 1})
    msg = NormalizedMessage(role=Role.ASSISTANT, content="", tool_calls=[tc])
    api = p._msg_to_api(msg)
    assert api["tool_calls"][0]["id"] == "tc-1"
    assert api["tool_calls"][0]["function"]["name"] == "fn"
    assert json.loads(api["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


def test_msg_to_api_assistant_with_reasoning():
    p = _provider()
    msg = NormalizedMessage(
        role=Role.ASSISTANT,
        content="answer",
        thinking_content="step-by-step reasoning here",
    )
    api = p._msg_to_api(msg)
    assert api["reasoning"] == "step-by-step reasoning here"


def test_msg_to_api_tool_result():
    p = _provider()
    msg = NormalizedMessage(role=Role.TOOL, content="result", tool_call_id="tc-1", name="fn")
    api = p._msg_to_api(msg)
    assert api["role"] == "tool"
    assert api["tool_call_id"] == "tc-1"
    assert api["name"] == "fn"


def test_build_payload_with_tools():
    p = _provider()
    msgs = [NormalizedMessage(role=Role.USER, content="q")]
    tools = [ToolDefinition(name="fn", description="d", json_schema={"type": "object"})]
    payload = p._build_payload(msgs, tools)
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["name"] == "fn"


def test_build_payload_no_tools():
    p = _provider()
    msgs = [NormalizedMessage(role=Role.USER, content="q")]
    payload = p._build_payload(msgs, None)
    assert "tools" not in payload


def test_compound_models_known():
    assert "compound-beta" in COMPOUND_MODELS
    assert "compound-beta-mini" in COMPOUND_MODELS


def test_valid_builtin_tools_known():
    assert "web_search" in VALID_BUILTIN_TOOLS
    assert "code_interpreter" in VALID_BUILTIN_TOOLS
    assert "browser_search" in VALID_BUILTIN_TOOLS
