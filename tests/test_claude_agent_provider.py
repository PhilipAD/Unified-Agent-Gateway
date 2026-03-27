"""Tests for Claude Agent SDK provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.types import GatewayError, NormalizedMessage, Role
from providers import claude_agent as ca


@pytest.mark.asyncio
async def test_claude_missing_sdk() -> None:
    with patch.object(ca, "_get_sdk", return_value=None):
        p = ca.ClaudeAgentProvider(api_key="k", model="m")
        with pytest.raises(GatewayError) as ei:
            await p.run([NormalizedMessage(role=Role.USER, content="hi")])
    assert ei.value.status_code == 503


@pytest.mark.asyncio
async def test_claude_with_mock_sdk() -> None:
    class Opts:
        def __init__(self, **kw):
            self.kw = kw

    class Msg:
        pass

    class Assistant(Msg):
        def __init__(self, c: str) -> None:
            self.content = c

    class Result(Msg):
        stop_reason = "end"
        session_id = "s1"
        usage = None

    async def query_fn(*, prompt, options):
        yield Assistant("partial")
        yield Result()

    fake_sdk = SimpleNamespace(
        ClaudeAgentOptions=Opts,
        query=query_fn,
    )

    with patch.object(ca, "_get_sdk", return_value=fake_sdk):
        p = ca.ClaudeAgentProvider(api_key="k", model="opus")
        r = await p.run([NormalizedMessage(role=Role.USER, content="go")])
    assert "partial" in r.messages[-1].content
