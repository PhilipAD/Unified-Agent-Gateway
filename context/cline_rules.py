"""Cline ``.clinerules`` context bridge."""

from __future__ import annotations

from typing import Any

from context.md_hierarchy import collect_md_hierarchy


def load_cline_rules(workspace_dir: str = ".") -> str:
    return collect_md_hierarchy(
        workspace_dir,
        (".clinerules", ".clinerules.md"),
        system_dirs=(),
        user_dirs=(),
        project_walk=True,
        stop_at_git_root=True,
        resolve_imports=True,
    )


async def fetch_cline_rules(**kwargs: Any) -> str:
    from config.settings import AgentHarnessSettings

    h = AgentHarnessSettings()
    cwd = str(kwargs.get("cwd") or kwargs.get("workspace_dir") or h.CLINE_RULES_WORKSPACE_DIR)
    return load_cline_rules(cwd)
