"""Windsurf Enterprise Analytics API client (service key in JSON body)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "https://server.codeium.com/api/v1"


async def post_cascade_analytics(
    service_key: str,
    *,
    start_timestamp: str,
    end_timestamp: str,
    query_requests: List[Dict[str, Any]],
    group_name: Optional[str] = None,
    emails: Optional[List[str]] = None,
    ide_types: Optional[List[str]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "service_key": service_key,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "query_requests": query_requests,
    }
    if group_name:
        body["group_name"] = group_name
    if emails:
        body["emails"] = emails
    if ide_types:
        body["ide_types"] = ide_types
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{BASE_URL}/CascadeAnalytics", json=body)
        r.raise_for_status()
        return r.json()
