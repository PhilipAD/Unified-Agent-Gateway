import pytest

from core.agent_loop import AgentLoop
from core.types import NormalizedMessage, NormalizedResponse, Role, ToolCall
from tools.registry import ToolRegistry, ToolSource


class FakeProvider:
    """Provider that returns scripted responses."""

    name = "fake"

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_idx = 0

    async def run(self, messages, tools=None, **kwargs):
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp

    async def stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_simple_conversation_no_tools():
    provider = FakeProvider(
        [
            NormalizedResponse(
                messages=[NormalizedMessage(role=Role.ASSISTANT, content="Hello!")],
                usage={"input_tokens": 5, "output_tokens": 2},
                provider="fake",
                model="fake-1",
            ),
        ]
    )
    loop = AgentLoop(provider=provider)
    result = await loop.run_conversation([NormalizedMessage(role=Role.USER, content="Hi")])
    assert result.messages[-1].content == "Hello!"
    assert len(result.conversation) == 2  # user + assistant


@pytest.mark.asyncio
async def test_tool_calling_loop():
    async def add_handler(a: int = 0, b: int = 0, **kwargs):
        return a + b

    tools = ToolRegistry()
    tools.register(
        name="add",
        description="Add two numbers",
        json_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
        },
        source=ToolSource.PYTHON,
        handler=add_handler,
    )

    # First response: model calls the add tool
    # Second response: model gives final answer
    provider = FakeProvider(
        [
            NormalizedResponse(
                messages=[
                    NormalizedMessage(
                        role=Role.ASSISTANT,
                        content="",
                        tool_calls=[ToolCall(id="tc-1", name="add", arguments={"a": 2, "b": 3})],
                    )
                ],
            ),
            NormalizedResponse(
                messages=[NormalizedMessage(role=Role.ASSISTANT, content="The sum is 5.")],
            ),
        ]
    )

    loop = AgentLoop(provider=provider, tools=tools)
    result = await loop.run_conversation([NormalizedMessage(role=Role.USER, content="Add 2 and 3")])
    assert result.messages[-1].content == "The sum is 5."
    # conversation: user, assistant(tool_call), tool_result, assistant(final)
    assert len(result.conversation) == 4


@pytest.mark.asyncio
async def test_max_tool_hops_limit():
    """Ensure the loop stops after max_tool_hops even if model keeps calling tools."""

    async def noop_handler(**kwargs):
        return "ok"

    tools = ToolRegistry()
    tools.register(
        name="loop_tool",
        description="Always called",
        json_schema={"type": "object"},
        source=ToolSource.PYTHON,
        handler=noop_handler,
    )

    # Model always returns a tool call
    def make_tc_response():
        return NormalizedResponse(
            messages=[
                NormalizedMessage(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(id="tc-x", name="loop_tool", arguments={})],
                )
            ],
        )

    provider = FakeProvider([make_tc_response() for _ in range(10)])
    loop = AgentLoop(provider=provider, tools=tools, max_tool_hops=3)
    await loop.run_conversation([NormalizedMessage(role=Role.USER, content="go")])
    # Should have stopped after 3 hops (3 model calls)
    assert provider._call_idx == 3


@pytest.mark.asyncio
async def test_tool_error_is_captured():
    async def failing_handler(**kwargs):
        raise ValueError("tool broke")

    tools = ToolRegistry()
    tools.register(
        name="broken",
        description="Will fail",
        json_schema={"type": "object"},
        source=ToolSource.PYTHON,
        handler=failing_handler,
    )

    provider = FakeProvider(
        [
            NormalizedResponse(
                messages=[
                    NormalizedMessage(
                        role=Role.ASSISTANT,
                        content="",
                        tool_calls=[ToolCall(id="tc-1", name="broken", arguments={})],
                    )
                ],
            ),
            NormalizedResponse(
                messages=[NormalizedMessage(role=Role.ASSISTANT, content="Tool failed.")],
            ),
        ]
    )

    loop = AgentLoop(provider=provider, tools=tools)
    result = await loop.run_conversation([NormalizedMessage(role=Role.USER, content="do it")])
    # The tool error should have been passed as a tool result message
    tool_msg = result.conversation[2]  # user, assistant, tool_result
    assert tool_msg.role == Role.TOOL
    assert "Error:" in tool_msg.content
