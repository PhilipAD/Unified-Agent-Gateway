import pytest

from core.handoff import HANDOFF_TOOL_DEFINITION, execute_handoff
from core.types import NormalizedMessage, NormalizedResponse, Role


def test_handoff_tool_definition_shape():
    td = HANDOFF_TOOL_DEFINITION
    assert td.name == "call_agent"
    assert "input" in td.json_schema["properties"]
    assert "input" in td.json_schema["required"]


@pytest.mark.asyncio
async def test_execute_handoff_no_fn():
    result = await execute_handoff(agent_id="x", profile="y", input_text="do something")
    assert "not configured" in result


@pytest.mark.asyncio
async def test_execute_handoff_delegates():
    async def fake_run(messages, agent_id, profile):
        return NormalizedResponse(
            messages=[
                NormalizedMessage(
                    role=Role.ASSISTANT,
                    content=f"handled by {agent_id}/{profile}",
                )
            ]
        )

    result = await execute_handoff(
        agent_id="specialist",
        profile="deep",
        input_text="analyze this",
        run_conversation_fn=fake_run,
    )
    assert result == "handled by specialist/deep"
