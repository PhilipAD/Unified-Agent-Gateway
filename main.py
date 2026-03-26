"""Entry point: ``python main.py`` or ``uvicorn main:app``."""

from __future__ import annotations

import asyncio

import uvicorn

from api.http import app, lifespan  # noqa: F401  (re-export for uvicorn)
from runtime.bootstrap import bootstrap_and_configure_app


async def _startup() -> None:
    await bootstrap_and_configure_app()


# Run bootstrap before uvicorn takes over
asyncio.get_event_loop().run_until_complete(_startup())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
