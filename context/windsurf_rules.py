"""Windsurf Cascade rules: global file, workspace ``.windsurf/rules``, ``AGENTS.md``."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from context.md_hierarchy import collect_glob_files_in_dirs, collect_md_hierarchy, find_git_root

logger = logging.getLogger(__name__)

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_trigger(md_text: str) -> Tuple[Optional[str], Optional[str]]:
    m = _FRONTMATTER.match(md_text)
    if not m:
        return None, None
    block = m.group(1)
    trigger = None
    globs = None
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k == "trigger":
                trigger = v
            if k == "globs":
                globs = v
    return trigger, globs


def _strip_frontmatter(md_text: str) -> str:
    m = _FRONTMATTER.match(md_text)
    if m:
        return md_text[m.end() :].lstrip()
    return md_text


def _system_rules_dirs() -> List[Path]:
    out: List[Path] = []
    if sys.platform == "darwin":
        out.append(Path("/Library/Application Support/Windsurf/rules"))
    elif sys.platform.startswith("linux"):
        out.append(Path("/etc/windsurf/rules"))
    elif sys.platform == "win32":
        out.append(Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Windsurf" / "rules")
    return out


def load_windsurf_rules(workspace_dir: str = ".") -> str:
    """Load Windsurf rules and compatible ``AGENTS.md`` snippets."""
    parts: List[str] = []
    ws = Path(workspace_dir).resolve()

    # System tier
    for d in _system_rules_dirs():
        blob = collect_glob_files_in_dirs([d], "*.md")
        if blob.strip():
            parts.append("## Windsurf system rules\n" + blob)

    # Global rules file
    global_rules = Path.home() / ".codeium" / "windsurf" / "memories" / "global_rules.md"
    if global_rules.is_file():
        try:
            parts.append("## Windsurf global rules\n" + global_rules.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("Cannot read global rules: %s", exc)

    # Workspace .windsurf/rules — include always_on and model_decision; skip glob/manual
    rule_dirs: List[Path] = []
    cur = ws
    git_root = find_git_root(ws)
    for _ in range(64):
        rd = cur / ".windsurf" / "rules"
        if rd.is_dir():
            rule_dirs.append(rd)
        if git_root is not None and cur == git_root:
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    for rd in rule_dirs:
        for path in sorted(rd.glob("*.md")):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            trigger, _globs = _parse_trigger(raw)
            if trigger in (None, "", "always_on", "model_decision"):
                body = _strip_frontmatter(raw) if trigger else raw
                parts.append(f"## Rule {path.name}\n{body.strip()}\n")

    # AGENTS.md along workspace ancestors (shared with other tools)
    agents = collect_md_hierarchy(
        str(ws),
        ("AGENTS.md",),
        system_dirs=(),
        user_dirs=(),
        project_walk=True,
        stop_at_git_root=True,
        resolve_imports=True,
    )
    if agents.strip():
        parts.append("## AGENTS.md (Windsurf-compatible)\n" + agents)

    return "\n\n".join(parts).strip()


async def fetch_windsurf_rules(**kwargs: Any) -> str:
    """ContextRegistry fetch.

    Respects per-request kwargs:
      ``cwd`` / ``workspace_dir``          — workspace directory
      ``windsurf_extra_rule_dirs``         — extra directories of ``*.md`` rule files
      ``windsurf_agents_md_filenames``     — override AGENTS.md filenames searched
    """
    from config.settings import AgentHarnessSettings

    h = AgentHarnessSettings()
    cwd = str(kwargs.get("cwd") or kwargs.get("workspace_dir") or h.WINDSURF_RULES_WORKSPACE_DIR)
    text = load_windsurf_rules(cwd)

    extra_dirs = kwargs.get("windsurf_extra_rule_dirs") or []
    if extra_dirs:
        from pathlib import Path

        from context.md_hierarchy import collect_glob_files_in_dirs

        extra_blob = collect_glob_files_in_dirs(
            [Path(d).expanduser().resolve() for d in extra_dirs], "*.md"
        )
        if extra_blob.strip():
            text = (text + "\n\n## Extra windsurf rules\n" + extra_blob).strip()

    return text
