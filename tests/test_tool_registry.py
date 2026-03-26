import pytest

from tools.registry import ToolRegistry, ToolSource


async def _dummy_handler(**kwargs):
    return {"echo": kwargs}


def test_register_and_get():
    reg = ToolRegistry()
    reg.register(
        name="echo",
        description="Echo back input",
        json_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        source=ToolSource.PYTHON,
        handler=_dummy_handler,
    )
    tool = reg.get("echo")
    assert tool.name == "echo"
    assert tool.source == ToolSource.PYTHON


def test_get_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="Unknown tool"):
        reg.get("nope")


def test_list_for_provider():
    reg = ToolRegistry()
    reg.register(
        name="a",
        description="tool a",
        json_schema={},
        source=ToolSource.PYTHON,
        handler=_dummy_handler,
    )
    reg.register(
        name="b",
        description="tool b",
        json_schema={},
        source=ToolSource.MCP,
        handler=_dummy_handler,
    )
    defs = reg.list_for_provider()
    assert len(defs) == 2
    names = {d.name for d in defs}
    assert names == {"a", "b"}


def test_has():
    reg = ToolRegistry()
    reg.register(
        name="x",
        description="x",
        json_schema={},
        source=ToolSource.PYTHON,
        handler=_dummy_handler,
    )
    assert reg.has("x")
    assert not reg.has("y")
