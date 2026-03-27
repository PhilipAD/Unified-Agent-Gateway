"""Minimal JSON-RPC 2.0 client for ``codex app-server`` (stdio, newline-delimited)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


class CodexAppServerClient:
    """Runs ``codex app-server`` and exchanges JSON-RPC messages."""

    def __init__(
        self,
        *,
        command: str = "codex",
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._env = env
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._next_id = 1
        self._reader_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._buf = ""

    async def start(self) -> None:
        if self._proc is not None:
            return
        env = dict(os.environ)
        if self._env:
            env.update(self._env)
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        logger.info("Started codex app-server subprocess")

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                logger.debug("Non-JSON line from codex app-server: %s", text[:200])
                continue
            if "id" in msg and msg["id"] in self._pending:
                fut = self._pending.pop(msg["id"])
                if "error" in msg:
                    fut.set_exception(RuntimeError(str(msg["error"])))
                else:
                    fut.set_result(msg.get("result"))

    async def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        await self.start()
        assert self._proc and self._proc.stdin
        req_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        line = json.dumps(payload) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()
        return await asyncio.wait_for(fut, timeout=600.0)

    async def stream_notifications(self) -> AsyncIterator[Dict[str, Any]]:
        """Best-effort: not fully implemented (use request/response only)."""
        if False:
            yield {}
        return
