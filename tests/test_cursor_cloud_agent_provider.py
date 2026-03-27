"""Tests for Cursor Cloud Agent provider."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from unittest.mock import patch

import httpx
import pytest

from core.types import GatewayError, NormalizedMessage, Role
from providers.cursor_cloud_agent import CursorCloudAgentProvider, verify_cursor_webhook_signature


class _FakeResponse:
    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.content = json.dumps(data).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("GET", "http://test")
            raise httpx.HTTPStatusError("x", request=req, response=self)  # type: ignore[arg-type]

    def json(self) -> dict:
        return self._data


class _FakeClient:
    def __init__(self) -> None:
        self._polls = 0

    async def get(self, url: str, headers: dict | None = None) -> _FakeResponse:
        if url.endswith("/conversation"):
            return _FakeResponse({"messages": [{"role": "assistant", "content": "done"}]})
        if "/artifacts" in url:
            return _FakeResponse([])
        self._polls += 1
        if self._polls < 2:
            return _FakeResponse({"status": "RUNNING"})
        return _FakeResponse({"status": "FINISHED", "prUrl": "https://example.com/pr"})

    async def post(
        self,
        url: str,
        headers: dict | None = None,
        json: dict | None = None,
    ) -> _FakeResponse:
        return _FakeResponse({"id": "agent-1"})


class _ClientCtx:
    def __init__(self, inner: _FakeClient) -> None:
        self._inner = inner

    async def __aenter__(self) -> _FakeClient:
        return self._inner

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_cursor_run_finishes() -> None:
    provider = CursorCloudAgentProvider(api_key="tok", model="default")
    fake = _FakeClient()

    def _factory(*a: object, **k: object) -> _ClientCtx:
        return _ClientCtx(fake)

    with patch("providers.cursor_cloud_agent.httpx.AsyncClient", side_effect=_factory):
        resp = await provider.run(
            [NormalizedMessage(role=Role.USER, content="hi")],
            repository="https://github.com/o/r",
            poll_interval_seconds=0.001,
            max_wait_seconds=2.0,
        )
    assert "done" in (resp.messages[-1].content or "")
    assert resp.usage.get("agent_id") == "agent-1"


def test_verify_cursor_webhook_signature() -> None:
    body = b'{"id":"x"}'
    secret = "x" * 32
    sig = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    assert verify_cursor_webhook_signature(body, sig, secret) is True
    assert verify_cursor_webhook_signature(body, "bad", secret) is False


@pytest.mark.asyncio
async def test_cursor_missing_repo() -> None:
    p = CursorCloudAgentProvider(api_key="k", model="m")
    with pytest.raises(GatewayError):
        await p.run([NormalizedMessage(role=Role.USER, content="x")])
