"""Agent handoff abstraction for multi-agent orchestration.

Provides a minimal ``call_agent`` meta-tool pattern that routes to another
configured agent profile.  This does **not** implement a full graph/tree
orchestrator; it ensures the architecture supports specialist delegation
without restructuring.
"""

from __future__ import annotations

from typing import Any, List, Optional

from core.types import NormalizedMessage, NormalizedResponse, Role, ToolDefinition

HANDOFF_TOOL_NAME = "call_agent"

HANDOFF_TOOL_DEFINITION = ToolDefinition(
    name=HANDOFF_TOOL_NAME,
    description=(
        "Delegate a sub-task to another agent profile. "
        "Returns the delegated agent's final response."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "Target agent identifier",
            },
            "profile": {
                "type": "string",
                "description": "Target agent profile",
            },
            "input": {
                "type": "string",
                "description": "The instruction / query to delegate",
            },
        },
        "required": ["input"],
    },
)


async def execute_handoff(
    agent_id: str,
    profile: str,
    input_text: str,
    *,
    parent_context: Optional[List[NormalizedMessage]] = None,
    run_conversation_fn: Any = None,
) -> str:
    """Execute a handoff by running another agent loop.

    Parameters
    ----------
    agent_id / profile:
        Identify the target agent configuration.
    input_text:
        The delegated query.
    parent_context:
        Optional prior conversation context to pass through.
    run_conversation_fn:
        An async callable ``(messages, agent_id, profile) -> NormalizedResponse``.
        Injected by the caller (typically ``AgentLoop`` or the HTTP layer)
        to avoid circular imports.
    """
    if run_conversation_fn is None:
        return "Handoff not configured: no run_conversation_fn provided."

    messages = list(parent_context or [])
    messages.append(NormalizedMessage(role=Role.USER, content=input_text))

    result: NormalizedResponse = await run_conversation_fn(
        messages, agent_id=agent_id, profile=profile
    )
    return result.messages[-1].content if result.messages else ""
