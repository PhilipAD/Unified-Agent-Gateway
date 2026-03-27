"""In-memory bridge for Cursor Cloud Agent status webhooks."""

from __future__ import annotations

import asyncio
from typing import Dict

_events: Dict[str, asyncio.Event] = {}


def get_cursor_agent_event(agent_id: str) -> asyncio.Event:
    """Return (and create if needed) an :class:`asyncio.Event` for *agent_id*."""
    ev = _events.get(agent_id)
    if ev is None:
        ev = asyncio.Event()
        _events[agent_id] = ev
    return ev


def signal_cursor_agent_event(agent_id: str) -> None:
    ev = _events.get(agent_id)
    if ev is not None:
        ev.set()


def clear_cursor_agent_event(agent_id: str) -> None:
    ev = _events.get(agent_id)
    if ev is not None:
        ev.clear()
