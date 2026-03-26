from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition
from providers.gemini import GeminiProvider


def _provider(**kw):
    defaults = {"api_key": "test-key", "model": "gemini-2.5-flash"}
    defaults.update(kw)
    return GeminiProvider(**defaults)


def test_provider_init():
    p = _provider()
    assert p.name == "gemini"
    assert p.model == "gemini-2.5-flash"


def test_build_contents_user():
    p = _provider()
    msgs = [NormalizedMessage(role=Role.USER, content="hello")]
    contents = p._build_contents(msgs)
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "hello"


def test_build_contents_system_skipped():
    p = _provider()
    msgs = [
        NormalizedMessage(role=Role.SYSTEM, content="system prompt"),
        NormalizedMessage(role=Role.USER, content="hi"),
    ]
    contents = p._build_contents(msgs)
    assert len(contents) == 1
    assert contents[0].role == "user"


def test_build_contents_assistant_with_tool_calls():
    p = _provider()
    tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
    msgs = [
        NormalizedMessage(role=Role.ASSISTANT, content="searching", tool_calls=[tc]),
    ]
    contents = p._build_contents(msgs)
    assert len(contents) == 1
    assert contents[0].role == "model"
    assert len(contents[0].parts) == 2
    assert contents[0].parts[0].text == "searching"
    assert contents[0].parts[1].function_call.name == "search"


def test_build_contents_tool_response():
    p = _provider()
    msgs = [
        NormalizedMessage(
            role=Role.TOOL, content="result", tool_call_id="tc-1", name="search"
        ),
    ]
    contents = p._build_contents(msgs)
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].function_response.name == "search"


def test_build_tools_function_declarations():
    p = _provider()
    tools = [ToolDefinition(name="fn", description="d", json_schema={"type": "object"})]
    result = p._build_tools(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0].function_declarations is not None
    assert result[0].function_declarations[0].name == "fn"


def test_build_tools_none():
    p = _provider()
    assert p._build_tools(None) is None


def test_build_tools_with_code_execution():
    p = _provider()
    result = p._build_tools(None, built_in_tools=["code_execution"])
    assert result is not None
    assert len(result) == 1
    assert result[0].code_execution is not None


def test_build_tools_with_google_search():
    p = _provider()
    result = p._build_tools(None, built_in_tools=["google_search"])
    assert result is not None
    assert len(result) == 1
    assert result[0].google_search is not None


def test_build_tools_with_url_context():
    p = _provider()
    result = p._build_tools(None, built_in_tools=["url_context"])
    assert result is not None
    assert len(result) == 1
    assert result[0].url_context is not None


def test_build_tools_with_google_maps():
    p = _provider()
    result = p._build_tools(None, built_in_tools=["google_maps"])
    assert result is not None
    assert len(result) == 1
    assert result[0].google_maps is not None


def test_build_tools_with_file_search():
    p = _provider()
    configs = [{"type": "file_search", "file_search_store_names": ["store-1"]}]
    result = p._build_tools(None, built_in_tools=["file_search"], built_in_tool_configs=configs)
    assert result is not None
    assert len(result) == 1
    assert result[0].file_search is not None


def test_build_tools_with_computer_use():
    p = _provider()
    result = p._build_tools(None, built_in_tools=["computer_use"])
    assert result is not None
    assert len(result) == 1
    assert result[0].computer_use is not None


def test_build_tools_with_mcp_servers():
    p = _provider()
    mcp = [{"server_url": "https://mcp.example.com", "server_label": "test"}]
    result = p._build_tools(None, mcp_servers=mcp)
    assert result is not None
    assert len(result) == 1
    assert result[0].mcp_servers is not None
    assert result[0].mcp_servers[0].name == "test"
    assert result[0].mcp_servers[0].streamable_http_transport.url == "https://mcp.example.com"


def test_build_tools_combined_function_and_builtin():
    p = _provider()
    tools = [ToolDefinition(name="fn", description="d", json_schema={"type": "object"})]
    result = p._build_tools(tools, built_in_tools=["code_execution", "google_search"])
    assert result is not None
    assert len(result) == 3
    assert result[0].function_declarations is not None
    assert result[1].code_execution is not None
    assert result[2].google_search is not None


def test_build_tools_ignores_unknown_builtins():
    p = _provider()
    result = p._build_tools(None, built_in_tools=["unknown_tool"])
    assert result is None
