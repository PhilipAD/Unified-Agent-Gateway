from __future__ import annotations

import json
from typing import Any, Dict


def format_sse(event_type: str, payload: Dict[str, Any]) -> str:
    """Format a single SSE frame: ``event: <type>\\ndata: <json>\\n\\n``."""
    data = json.dumps(payload, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"
