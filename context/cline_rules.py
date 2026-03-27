"""Cline ``.clinerules`` context bridge."""

from __future__ import annotations

from typing import Any, Sequence

from context.md_hierarchy import collect_md_hierarchy

_DEFAULT_FILENAMES = (".clinerules", ".clinerules.md")


def load_cline_rules(
    workspace_dir: str = ".",
    *,
    filenames: Sequence[str] = _DEFAULT_FILENAMES,
    system_dirs: Sequence[str] = (),
    user_dirs: Sequence[str] = (),
) -> str:
    """Load Cline rules from *workspace_dir* ancestors.

    *filenames* defaults to ``(".clinerules", ".clinerules.md")`` but can be
    overridden per request via the ``cline_filenames`` context kwarg.
    """
    return collect_md_hierarchy(
        workspace_dir,
        filenames,
        system_dirs=system_dirs,
        user_dirs=user_dirs,
        project_walk=True,
        stop_at_git_root=True,
        resolve_imports=True,
    )


async def fetch_cline_rules(**kwargs: Any) -> str:
    """ContextRegistry fetch.

    Respects per-request kwargs:
      ``cwd`` / ``workspace_dir``      — working directory
      ``cline_filenames``              — override filenames list
      ``cline_system_dirs``            — extra system-level directories
      ``cline_user_dirs``              — extra user-level directories
    """
    from config.settings import AgentHarnessSettings

    h = AgentHarnessSettings()
    cwd = str(kwargs.get("cwd") or kwargs.get("workspace_dir") or h.CLINE_RULES_WORKSPACE_DIR)
    filenames = kwargs.get("cline_filenames") or _DEFAULT_FILENAMES
    system_dirs = kwargs.get("cline_system_dirs") or ()
    user_dirs = kwargs.get("cline_user_dirs") or ()
    return load_cline_rules(cwd, filenames=filenames, system_dirs=system_dirs, user_dirs=user_dirs)
