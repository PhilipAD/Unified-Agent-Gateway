from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

ContextFn = Callable[..., Awaitable[str]]


class ContextSource(str, Enum):
    CONTEXT_FORGE = "context_forge"
    RAG = "rag"
    STATIC = "static"
    KV = "kv"


class RegisteredContext:
    def __init__(
        self,
        name: str,
        source: ContextSource,
        fetch: ContextFn,
        required: bool = False,
        max_chars: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.source = source
        self.fetch = fetch
        self.required = required
        self.max_chars = max_chars
        self.metadata = metadata or {}


class ContextRegistry:
    def __init__(self) -> None:
        self._contexts: Dict[str, RegisteredContext] = {}

    def register(self, ctx: RegisteredContext) -> None:
        self._contexts[ctx.name] = ctx

    async def load_all(self, **kwargs: Any) -> Dict[str, str]:
        """Fetch all registered context sources. Returns name -> text mapping."""
        out: Dict[str, str] = {}
        for name, ctx in self._contexts.items():
            try:
                text = await ctx.fetch(**kwargs)
                if ctx.max_chars and len(text) > ctx.max_chars:
                    text = text[: ctx.max_chars] + "\n[truncated]"
                out[name] = text
            except Exception:
                if ctx.required:
                    raise
                logger.warning("Context source '%s' failed; skipping.", name, exc_info=True)
        return out

    def list_registered(self) -> Dict[str, RegisteredContext]:
        """Return name -> registered context mapping."""
        return dict(self._contexts)

    def copy(self) -> "ContextRegistry":
        """Shallow copy registry entries for per-request isolation."""
        new_registry = ContextRegistry()
        for ctx in self._contexts.values():
            new_registry.register(
                RegisteredContext(
                    name=ctx.name,
                    source=ctx.source,
                    fetch=ctx.fetch,
                    required=ctx.required,
                    max_chars=ctx.max_chars,
                    metadata=dict(ctx.metadata),
                )
            )
        return new_registry
