"""ContextForge integration as a registered context source.

Fetches enrichment context from a ContextForge-compatible HTTP endpoint and
returns it as plain text for prompt injection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from context.registry import ContextRegistry, ContextSource, RegisteredContext

logger = logging.getLogger(__name__)


def register_contextforge(
    registry: ContextRegistry,
    base_url: str,
    api_key: str,
    required: bool = False,
    max_chars: Optional[int] = None,
    timeout: float = 10.0,
) -> None:
    """Register a ContextForge fetch function in *registry*."""

    async def fetch(**kwargs: Any) -> str:
        payload = {
            "input": kwargs.get("input", ""),
            "user_id": kwargs.get("user_id"),
            "session_id": kwargs.get("session_id"),
        }
        # Forward any extra domain-specific fields
        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v

        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/context",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        return data.get("context", "")

    registry.register(
        RegisteredContext(
            name="contextforge",
            source=ContextSource.CONTEXT_FORGE,
            fetch=fetch,
            required=required,
            max_chars=max_chars,
            metadata={"base_url": base_url},
        )
    )
