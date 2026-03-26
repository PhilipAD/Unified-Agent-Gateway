from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional

from core.types import NormalizedMessage, NormalizedResponse, StreamEvent, ToolDefinition


class BaseProvider(ABC):
    name: str

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.extra = kwargs

    @abstractmethod
    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        ...
        yield  # pragma: no cover  (makes this a valid async generator)
