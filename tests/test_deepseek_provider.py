from core.types import NormalizedMessage, Role, ToolDefinition
from providers.deepseek import DEEPSEEK_BASE_URL, DeepSeekProvider


def _provider(**kw):
    defaults = {"api_key": "sk-ds-test", "model": "deepseek-chat"}
    defaults.update(kw)
    return DeepSeekProvider(**defaults)


def test_provider_init():
    p = _provider()
    assert p.name == "deepseek"
    assert p.api_key == "sk-ds-test"
    assert p.model == "deepseek-chat"


def test_effective_base_url_default():
    p = _provider()
    assert p._effective_base_url() == DEEPSEEK_BASE_URL


def test_effective_base_url_custom():
    p = _provider(base_url="https://custom.deepseek.com/v1/")
    assert p._effective_base_url() == "https://custom.deepseek.com/v1"


def test_inherits_openai_compatible_msg_to_api():
    p = _provider()
    msg = NormalizedMessage(role=Role.USER, content="hello")
    api = p._msg_to_api(msg)
    assert api == {"role": "user", "content": "hello"}


def test_msg_to_api_assistant_with_reasoning():
    p = _provider()
    msg = NormalizedMessage(
        role=Role.ASSISTANT,
        content="answer",
        thinking_content="step by step reasoning",
    )
    api = p._msg_to_api(msg)
    assert api["reasoning_content"] == "step by step reasoning"
    assert api["content"] == "answer"


def test_inherits_openai_compatible_build_payload():
    p = _provider()
    msgs = [NormalizedMessage(role=Role.USER, content="q")]
    tools = [ToolDefinition(name="fn", description="d", json_schema={"type": "object"})]
    payload = p._build_payload(msgs, tools)
    assert payload["model"] == "deepseek-chat"
    assert len(payload["tools"]) == 1


def test_inherits_parse_tool_calls():
    raw = [
        {
            "id": "tc-1",
            "function": {"name": "fn", "arguments": '{"x": 1}'},
        }
    ]
    from providers.openai_compatible import OpenAICompatibleProvider

    parsed = OpenAICompatibleProvider._parse_tool_calls(raw)
    assert parsed[0].arguments == {"x": 1}


def test_thinking_content_field_available():
    msg = NormalizedMessage(
        role=Role.ASSISTANT,
        content="answer",
        thinking_content="I need to reason about this",
    )
    d = msg.to_dict()
    assert d["thinking_content"] == "I need to reason about this"


def test_thinking_content_none_not_in_dict():
    msg = NormalizedMessage(role=Role.ASSISTANT, content="answer")
    d = msg.to_dict()
    assert "thinking_content" not in d


def test_normalize_thinking_param_bool_true():
    p = _provider()
    result = p._normalize_thinking_param(True)
    assert result == {"type": "enabled"}


def test_normalize_thinking_param_bool_false():
    p = _provider()
    result = p._normalize_thinking_param(False)
    assert result == {"type": "disabled"}


def test_normalize_thinking_param_dict():
    p = _provider()
    result = p._normalize_thinking_param({"type": "enabled"})
    assert result == {"type": "enabled"}


def test_normalize_thinking_param_none():
    p = _provider()
    result = p._normalize_thinking_param(None)
    assert result is None


def test_normalize_thinking_param_string():
    p = _provider()
    result = p._normalize_thinking_param("enabled")
    assert result == {"type": "enabled"}
