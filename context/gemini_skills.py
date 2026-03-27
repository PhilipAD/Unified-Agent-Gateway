"""Discover Gemini CLI-style skills from ``SKILL.md`` files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    name: str
    description: str
    location: str
    disabled: bool = False
    argument_hint: Optional[str] = None
    disable_model_invocation: bool = False
    allowed_tools: List[str] = field(default_factory=list)
    raw_frontmatter: Dict[str, Any] = field(default_factory=dict)


def _parse_simple_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML-like ``---`` frontmatter (minimal key: value lines)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: Dict[str, Any] = {}
    i = 1
    while i < len(lines):
        line = lines[i]
        if line.strip() == "---":
            body = "\n".join(lines[i + 1 :])
            break
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            meta[key] = val
        i += 1
    else:
        return {}, text
    return meta, body


def _skill_from_file(path: Path) -> Optional[SkillMetadata]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read skill file %s: %s", path, exc)
        return None
    fm, body = _parse_simple_frontmatter(raw)
    name = str(fm.get("name") or path.parent.name or path.stem)
    desc = str(fm.get("description") or body.strip()[:500] or name)
    disabled = str(fm.get("disabled", "")).lower() in ("true", "1", "yes")
    dmi = str(fm.get("disable-model-invocation", "")).lower() in ("true", "1", "yes")
    arg_hint = fm.get("argument-hint") or fm.get("argument_hint")
    allowed = fm.get("allowed-tools") or fm.get("allowed_tools") or ""
    allowed_list = [x.strip() for x in str(allowed).split(",") if x.strip()]
    return SkillMetadata(
        name=name,
        description=desc,
        location=str(path.resolve()),
        disabled=disabled,
        argument_hint=str(arg_hint) if arg_hint else None,
        disable_model_invocation=dmi,
        allowed_tools=allowed_list,
        raw_frontmatter=fm,
    )


def discover_skills(
    workspace_dir: str = ".",
    *,
    user_skill_roots: Sequence[str] = (
        "~/.gemini/skills",
        "~/.agents/skills",
    ),
    project_skill_roots: Sequence[str] = (
        ".gemini/skills",
        ".agents/skills",
    ),
) -> List[SkillMetadata]:
    """Discover skills; project names override user names on collision."""
    ws = Path(workspace_dir).resolve()
    by_name: Dict[str, SkillMetadata] = {}

    def add_from_root(root: Path, tier: str) -> None:
        if not root.is_dir():
            return
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            md = skill_dir / "SKILL.md"
            if not md.is_file():
                continue
            meta = _skill_from_file(md)
            if meta is None or meta.disabled:
                continue
            key = meta.name
            if tier == "user":
                by_name.setdefault(key, meta)
            else:
                by_name[key] = meta

    for rel in user_skill_roots:
        add_from_root(Path(rel).expanduser(), "user")
    for rel in project_skill_roots:
        add_from_root((ws / rel).resolve(), "project")
        # Walk ancestors for project-local skills
        cur = ws
        for _ in range(32):
            for sub in (".gemini/skills", ".agents/skills"):
                add_from_root((cur / sub).resolve(), "project")
            if cur.parent == cur:
                break
            cur = cur.parent

    return list(by_name.values())


def format_skills_catalog(skills: Sequence[SkillMetadata]) -> str:
    lines: List[str] = ["# Gemini CLI skills (discovered)", ""]
    for s in skills:
        lines.append(f"## {s.name}")
        lines.append(s.description)
        if s.argument_hint:
            lines.append(f"Argument hint: {s.argument_hint}")
        if s.allowed_tools:
            lines.append("Allowed tools: " + ", ".join(s.allowed_tools))
        lines.append(f"Path: {s.location}")
        lines.append("")
    return "\n".join(lines).strip()


async def fetch_gemini_skills_catalog(**kwargs: Any) -> str:
    from config.settings import AgentHarnessSettings

    h = AgentHarnessSettings()
    cwd = str(kwargs.get("cwd") or kwargs.get("workspace_dir") or h.GEMINI_CLI_SKILLS_WORKSPACE_DIR)
    skills = discover_skills(cwd)
    return format_skills_catalog(skills)
