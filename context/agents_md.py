"""Cross-tool ``AGENTS.md`` context (Codex, Gemini CLI alias, Windsurf, Claude Code)."""

from __future__ import annotations

from typing import Any

from context.md_hierarchy import collect_md_hierarchy


def load_agents_md(cwd: str = ".") -> str:
    """Load ``AGENTS.md`` from *cwd* and ancestors up to git root."""
    return collect_md_hierarchy(
        cwd,
        ("AGENTS.md",),
        system_dirs=(),
        user_dirs=(),
        project_walk=True,
        stop_at_git_root=True,
        resolve_imports=True,
    )


async def fetch_agents_md(**kwargs: Any) -> str:
    """ContextRegistry fetch: uses kwargs ``cwd`` or :class:`AgentHarnessSettings`."""
    from config.settings import AgentHarnessSettings

    h = AgentHarnessSettings()
    cwd = str(kwargs.get("cwd") or kwargs.get("agents_md_cwd") or h.AGENTS_MD_CWD)
    return load_agents_md(cwd)
