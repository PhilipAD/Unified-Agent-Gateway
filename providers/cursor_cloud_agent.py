"""Cursor Cloud Agents API provider (remote job orchestration)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from core.types import (
    GatewayError,
    NormalizedMessage,
    NormalizedResponse,
    Role,
    StreamEvent,
    ToolDefinition,
)
from providers.base import BaseProvider
from runtime.cursor_webhook import get_cursor_agent_event

logger = logging.getLogger(__name__)

BASE_URL = "https://api.cursor.com"
TERMINAL_STATUSES = frozenset({"FINISHED", "ERROR", "EXPIRED"})

_REPO_CACHE: Dict[str, Any] = {"ts": 0.0, "data": []}


def verify_cursor_webhook_signature(body: bytes, signature_b64: str, secret: str) -> bool:
    if not secret or len(secret) < 32:
        return False
    try:
        digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        expected = base64.b64encode(digest).decode()
    except Exception:
        return False
    return hmac.compare_digest(expected.strip(), signature_b64.strip())


class CursorCloudAgentProvider(BaseProvider):
    name = "cursor_cloud_agent"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key, model, base_url, **kwargs)
        self._api_base = (base_url or BASE_URL).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        token = self.api_key
        raw = f"{token}:".encode()
        b64 = base64.b64encode(raw).decode()
        return {
            "Authorization": f"Basic {b64}",
            "Content-Type": "application/json",
        }

    async def _get_json(self, client: httpx.AsyncClient, path: str) -> Any:
        r = await client.get(f"{self._api_base}{path}", headers=self._headers())
        if r.status_code == 409:
            raise GatewayError(r.text, provider=self.name, status_code=409)
        r.raise_for_status()
        return r.json()

    async def _post_json(self, client: httpx.AsyncClient, path: str, body: Dict[str, Any]) -> Any:
        r = await client.post(f"{self._api_base}{path}", headers=self._headers(), json=body)
        if r.status_code == 409:
            raise GatewayError(r.text, provider=self.name, status_code=409)
        r.raise_for_status()
        if not r.content:
            return {}
        return r.json()

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        if tools:
            logger.warning("Cursor Cloud Agents API does not accept UAG tools; ignoring.")

        prompt_text = self._messages_to_prompt(messages)
        extra = {**self.extra, **kwargs}
        poll_interval = float(extra.get("poll_interval_seconds", 15.0))
        max_wait = float(extra.get("max_wait_seconds", 600.0))
        repository = extra.get("repository") or extra.get("CURSOR_DEFAULT_REPOSITORY")
        ref = extra.get("ref", "main")
        webhook_url = extra.get("webhook_url")

        if not repository:
            raise GatewayError(
                "Cursor Cloud Agent requires repository URL in profile.extra or kwargs",
                provider=self.name,
                status_code=400,
            )

        launch_body: Dict[str, Any] = {
            "prompt": {"text": prompt_text},
            "source": {"repository": repository, "ref": ref},
            "model": self.model or "default",
        }
        if extra.get("target"):
            launch_body["target"] = extra["target"]
        if webhook_url:
            launch_body["webhook"] = {"url": webhook_url}

        async with httpx.AsyncClient(timeout=120.0) as client:
            created = await self._post_json(client, "/v0/agents", launch_body)
            agent_id = created.get("id") or created.get("agentId")
            if not agent_id:
                raise GatewayError(
                    f"Unexpected launch response: {created}",
                    provider=self.name,
                    status_code=502,
                )

            deadline = time.monotonic() + max_wait
            backoff = poll_interval
            status_data: Dict[str, Any] = {}
            while time.monotonic() < deadline:
                ev = get_cursor_agent_event(str(agent_id))
                try:
                    remain = max(1.0, deadline - time.monotonic())
                    await asyncio.wait_for(ev.wait(), timeout=min(backoff, remain))
                except asyncio.TimeoutError:
                    pass
                ev.clear()
                status_data = await self._get_json(client, f"/v0/agents/{agent_id}")
                st = status_data.get("status") or status_data.get("state")
                if st in TERMINAL_STATUSES:
                    break
                backoff = min(backoff * 1.5, 60.0)
            else:
                raise GatewayError(
                    "Cursor agent polling timed out",
                    provider=self.name,
                    status_code=504,
                )

            conv = await self._get_json(client, f"/v0/agents/{agent_id}/conversation")
            artifacts: List[Any] = []
            try:
                art = await self._get_json(client, f"/v0/agents/{agent_id}/artifacts")
                artifacts = art if isinstance(art, list) else art.get("artifacts", [])
            except httpx.HTTPStatusError:
                artifacts = []

        assistant_text = self._conversation_to_text(conv)
        usage = {
            "agent_id": agent_id,
            "status": status_data.get("status"),
            "pr_url": status_data.get("prUrl") or status_data.get("pr_url"),
            "summary": status_data.get("summary"),
            "artifacts": artifacts,
            "raw_status": status_data,
        }
        return NormalizedResponse(
            messages=[NormalizedMessage(role=Role.ASSISTANT, content=assistant_text)],
            conversation=self._normalize_conversation(conv),
            usage=usage,
            provider=self.name,
            model=self.model,
            raw={"status": status_data, "conversation": conv},
        )

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        try:
            resp = await self.run(messages, tools, **kwargs)
        except GatewayError as exc:
            yield StreamEvent(type="error", error=str(exc))
            return
        yield StreamEvent(type="metadata", metadata={"usage": resp.usage})
        text = ""
        if resp.messages:
            c = resp.messages[-1].content
            text = c if isinstance(c, str) else str(c)
        yield StreamEvent(type="chunk", delta=text)
        yield StreamEvent(type="done", response=resp)

    def _messages_to_prompt(self, messages: List[NormalizedMessage]) -> str:
        parts: List[str] = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            c = m.content
            if isinstance(c, str):
                parts.append(f"{role}: {c}")
            else:
                parts.append(f"{role}: {c!r}")
        return "\n\n".join(parts)

    def _conversation_to_text(self, conv: Any) -> str:
        if isinstance(conv, dict):
            msgs = conv.get("messages") or conv.get("conversation") or []
        elif isinstance(conv, list):
            msgs = conv
        else:
            return str(conv)
        out: List[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or m.get("type")
            content = m.get("content") or m.get("text") or ""
            if isinstance(content, list):
                content = " ".join(str(x) for x in content)
            if role == "assistant" or role == "ASSISTANT":
                out.append(str(content))
        return "\n".join(out) if out else str(conv)

    def _normalize_conversation(self, conv: Any) -> List[NormalizedMessage]:
        if isinstance(conv, dict):
            msgs = conv.get("messages") or conv.get("conversation") or []
        elif isinstance(conv, list):
            msgs = conv
        else:
            return []
        result: List[NormalizedMessage] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role_raw = (m.get("role") or m.get("type") or "user").lower()
            role = Role.USER
            if role_raw in ("assistant", "agent"):
                role = Role.ASSISTANT
            elif role_raw == "system":
                role = Role.SYSTEM
            content = m.get("content") or m.get("text") or ""
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            result.append(NormalizedMessage(role=role, content=str(content)))
        return result

    async def followup(
        self,
        agent_id: str,
        text: str,
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"prompt": {"text": text}}
        if images:
            body["prompt"]["images"] = images
        async with httpx.AsyncClient(timeout=120.0) as client:
            return await self._post_json(client, f"/v0/agents/{agent_id}/followup", body)

    async def stop_agent(self, agent_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await self._post_json(client, f"/v0/agents/{agent_id}/stop", {})

    async def delete_agent(self, agent_id: str) -> None:
        async with httpx.AsyncClient(timeout=60.0) as client:
            url = f"{self._api_base}/v0/agents/{agent_id}"
            r = await client.delete(url, headers=self._headers())
            if r.status_code == 409:
                raise GatewayError(r.text, provider=self.name, status_code=409)
            r.raise_for_status()

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await self._get_json(client, f"/v0/agents/{agent_id}")

    async def get_conversation(self, agent_id: str) -> Any:
        async with httpx.AsyncClient(timeout=120.0) as client:
            return await self._get_json(client, f"/v0/agents/{agent_id}/conversation")

    async def list_agent_artifacts(self, agent_id: str) -> Any:
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await self._get_json(client, f"/v0/agents/{agent_id}/artifacts")


__all__ = ["CursorCloudAgentProvider", "verify_cursor_webhook_signature"]
