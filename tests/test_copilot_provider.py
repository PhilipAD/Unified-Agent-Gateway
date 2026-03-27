"""Tests for GitHub Copilot SDK provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.types import GatewayError, NormalizedMessage, Role
from providers import copilot_provider as cp


@pytest.mark.asyncio
async def test_copilot_missing_sdk() -> None:
    with patch.object(cp, "_load_copilot", return_value=None):
        p = cp.CopilotProvider(api_key="t", model="default")
        with pytest.raises(GatewayError) as ei:
            await p.run([NormalizedMessage(role=Role.USER, content="hi")])
    assert ei.value.status_code == 503


@pytest.mark.asyncio
async def test_copilot_mock_run() -> None:
    def run_sync(prompt, model=None):
        return SimpleNamespace(text="out")

    fake = SimpleNamespace(run=run_sync)
    with patch.object(cp, "_load_copilot", return_value=fake):
        p = cp.CopilotProvider(api_key="t", model="m")
        r = await p.run([NormalizedMessage(role=Role.USER, content="x")])
    assert r.messages[-1].content == "out"
