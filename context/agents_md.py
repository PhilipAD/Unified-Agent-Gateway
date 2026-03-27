"""Cross-tool ``AGENTS.md`` context (Codex, Gemini CLI alias, Windsurf, Claude Code)."""

from __future__ import annotations

from typing import Any, Sequence

from context.md_hierarchy import collect_md_hierarchy

_DEFAULT_FILENAMES = ("AGENTS.md",)


def load_agents_md(
    cwd: str = ".",
    *,
    filenames: Sequence[str] = _DEFAULT_FILENAMES,
    system_dirs: Sequence[str] = (),
    user_dirs: Sequence[str] = (),
) -> str:
    """Load AGENTS.md (or custom *filenames*) from *cwd* and ancestors up to git root."""
    return collect_md_hierarchy(
        cwd,
        filenames,
        system_dirs=system_dirs,
        user_dirs=user_dirs,
        project_walk=True,
        stop_at_git_root=True,
        resolve_imports=True,
    )


async def fetch_agents_md(**kwargs: Any) -> str:
    """ContextRegistry fetch.

    Respects per-request kwargs:
      ``cwd`` / ``agents_md_cwd``        — working directory
      ``agents_md_filenames``             — override filenames list
      ``agents_md_system_dirs``           — extra system-level directories
      ``agents_md_user_dirs``             — extra user-level directories
    """
    from config.settings import AgentHarnessSettings

    h = AgentHarnessSettings()
    cwd = str(kwargs.get("cwd") or kwargs.get("agents_md_cwd") or h.AGENTS_MD_CWD)
    filenames = kwargs.get("agents_md_filenames") or _DEFAULT_FILENAMES
    system_dirs = kwargs.get("agents_md_system_dirs") or ()
    user_dirs = kwargs.get("agents_md_user_dirs") or ()
    return load_agents_md(cwd, filenames=filenames, system_dirs=system_dirs, user_dirs=user_dirs)
