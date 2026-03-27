"""OpenAI Codex CLI provider (subprocess ``codex -q`` or app-server JSON-RPC)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any, AsyncIterator, Dict, List, Optional

from core.types import (
    GatewayError,
    NormalizedMessage,
    NormalizedResponse,
    Role,
    StreamEvent,
    ToolDefinition,
)
from providers.base import BaseProvider
from runtime.codex_app_server import CodexAppServerClient

logger = logging.getLogger(__name__)


class CodexProvider(BaseProvider):
    name = "codex"

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        if tools:
            logger.warning("Codex CLI manages its own tools; UAG tool list is not forwarded.")
        extra = {**self.extra, **kwargs}
        if extra.get("use_app_server") or os.environ.get("CODEX_USE_APP_SERVER") == "1":
            return await self._run_app_server(messages, extra)
        return await self._run_subprocess(messages, extra)

    async def _run_subprocess(
        self,
        messages: List[NormalizedMessage],
        extra: Dict[str, Any],
    ) -> NormalizedResponse:
        binary = str(extra.get("codex_binary") or extra.get("CODEX_BINARY") or "codex")
        if not shutil.which(binary.split()[0]):
            raise GatewayError(
                f"Codex binary not found in PATH: {binary}",
                provider=self.name,
                status_code=503,
            )
        prompt = self._flatten_messages(messages)
        args: List[str] = [binary, "-q", "--model", self.model or "codex-mini-latest"]
        prov = extra.get("provider") or extra.get("CODEX_PROVIDER")
        if prov:
            args.extend(["--provider", str(prov)])
        reasoning = extra.get("reasoning") or extra.get("CODEX_REASONING_EFFORT")
        if reasoning:
            args.extend(["--reasoning", str(reasoning)])
        if extra.get("no_project_doc"):
            args.append("--no-project-doc")
        pd = extra.get("project_doc") or extra.get("project_doc_path")
        if pd:
            args.extend(["--project-doc", str(pd)])
        if extra.get("full_context"):
            args.append("--full-context")
        env = dict(os.environ)
        key = self.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
        if key:
            env["OPENAI_API_KEY"] = key
        cwd = extra.get("cwd") or os.getcwd()
        full_args = list(args) + [prompt]
        proc = await asyncio.create_subprocess_exec(
            *full_args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        out_b, err_b = await proc.communicate()
        out = out_b.decode("utf-8", errors="replace")
        err = err_b.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise GatewayError(
                err or out or f"codex exited {proc.returncode}",
                provider=self.name,
                status_code=500,
            )
        return NormalizedResponse(
            messages=[NormalizedMessage(role=Role.ASSISTANT, content=out.strip())],
            usage={"stderr": err} if err.strip() else {},
            provider=self.name,
            model=self.model,
        )

    async def _run_app_server(
        self,
        messages: List[NormalizedMessage],
        extra: Dict[str, Any],
    ) -> NormalizedResponse:
        client = CodexAppServerClient(
            command=str(extra.get("codex_binary") or "codex"),
            cwd=extra.get("cwd"),
            env=self._env_with_key(),
        )
        try:
            prompt = self._flatten_messages(messages)
            # Methods are illustrative; real app-server API may differ.
            await client.request(
                "thread/start",
                {"sandbox": {"type": extra.get("sandbox_mode", "workspaceWrite")}},
            )
            result = await client.request(
                "turn/start",
                {"prompt": prompt, "model": self.model},
            )
            text = json.dumps(result) if not isinstance(result, str) else result
            return NormalizedResponse(
                messages=[NormalizedMessage(role=Role.ASSISTANT, content=text)],
                usage={},
                provider=self.name,
                model=self.model,
                raw=result,
            )
        finally:
            await client.close()

    def _env_with_key(self) -> Dict[str, str]:
        env = dict(os.environ)
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            env["OPENAI_API_KEY"] = key
        return env

    def _flatten_messages(self, messages: List[NormalizedMessage]) -> str:
        parts: List[str] = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            c = m.content
            parts.append(f"{role}: {c}" if isinstance(c, str) else f"{role}: {c!r}")
        return "\n\n".join(parts)

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        resp = await self.run(messages, tools, **kwargs)
        if resp.messages:
            c = resp.messages[-1].content
            text = c if isinstance(c, str) else str(c)
            yield StreamEvent(type="chunk", delta=text)
        yield StreamEvent(type="done", response=resp)
