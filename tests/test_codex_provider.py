"""Tests for Codex CLI provider (subprocess)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.types import GatewayError, NormalizedMessage, Role
from providers.codex_provider import CodexProvider


@pytest.mark.asyncio
async def test_codex_subprocess_success() -> None:
    p = CodexProvider(api_key="sk", model="codex-mini-latest")

    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"hello codex", b""))

    with (
        patch(
            "providers.codex_provider.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ),
        patch("providers.codex_provider.shutil.which", return_value="/bin/codex"),
    ):
        r = await p.run([NormalizedMessage(role=Role.USER, content="fix")])
    assert r.messages[-1].content == "hello codex"


@pytest.mark.asyncio
async def test_codex_binary_missing() -> None:
    p = CodexProvider(api_key="sk", model="m")
    with patch("providers.codex_provider.shutil.which", return_value=None):
        with pytest.raises(GatewayError) as ei:
            await p.run([NormalizedMessage(role=Role.USER, content="x")])
    assert ei.value.status_code == 503
