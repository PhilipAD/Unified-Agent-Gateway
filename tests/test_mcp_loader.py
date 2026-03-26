import pytest

from tools.mcp_loader import load_mcp_tools_from_server
from tools.registry import ToolRegistry, ToolSource


class FakeMCPTool:
    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.schema = schema


class FakeMCPClient:
    def __init__(self, tools):
        self._tools = tools
        self.calls = []

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"status": "ok", "tool": name}


@pytest.mark.asyncio
async def test_load_mcp_tools_registers_namespaced():
    client = FakeMCPClient(
        [
            FakeMCPTool("get_events", "Get calendar events", {"type": "object"}),
            FakeMCPTool("create_event", "Create event", {"type": "object"}),
        ]
    )
    reg = ToolRegistry()
    count = await load_mcp_tools_from_server(reg, client, namespace="calendar")
    assert count == 2
    assert reg.has("calendar.get_events")
    assert reg.has("calendar.create_event")
    tool = reg.get("calendar.get_events")
    assert tool.source == ToolSource.MCP


@pytest.mark.asyncio
async def test_mcp_handler_delegates_to_client():
    client = FakeMCPClient(
        [
            FakeMCPTool("search", "Search code", {"type": "object"}),
        ]
    )
    reg = ToolRegistry()
    await load_mcp_tools_from_server(reg, client, namespace="code")
    tool = reg.get("code.search")
    result = await tool.handler(query="test")
    assert result == {"status": "ok", "tool": "search"}
    assert client.calls == [("search", {"query": "test"})]


@pytest.mark.asyncio
async def test_no_closure_bug_with_multiple_tools():
    """Ensure each tool handler captures the correct tool_name."""
    client = FakeMCPClient(
        [
            FakeMCPTool("a", "tool a", {}),
            FakeMCPTool("b", "tool b", {}),
            FakeMCPTool("c", "tool c", {}),
        ]
    )
    reg = ToolRegistry()
    await load_mcp_tools_from_server(reg, client, namespace="ns")

    await reg.get("ns.a").handler()
    await reg.get("ns.b").handler()
    await reg.get("ns.c").handler()

    called_names = [name for name, _ in client.calls]
    assert called_names == ["a", "b", "c"]
