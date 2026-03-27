"""Gemini CLI-style hierarchical memory (GEMINI.md / AGENTS.md / CLAUDE.md)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from context.md_hierarchy import collect_md_hierarchy

logger = logging.getLogger(__name__)

_MEMORY_SECTION = re.compile(
    r"^##\s+Gemini Added Memories\s*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class HierarchicalMemory:
    global_mem: str = ""
    extension_mem: str = ""
    project_mem: str = ""


def _default_system_gemini_dir() -> Optional[str]:
    import sys

    if sys.platform == "darwin":
        mac = "/Library/Application Support/GeminiCli"
        if Path(mac).is_dir():
            return mac
    return None


def _read_json_settings(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Gemini settings not read from %s: %s", path, exc)
        return {}


def gemini_context_filenames_from_settings(workspace_dir: str) -> List[str]:
    """Resolve ``context.fileName`` (or list) from workspace ``settings.json``."""
    p = Path(workspace_dir).resolve() / ".gemini" / "settings.json"
    data = _read_json_settings(p)
    ctx = data.get("context") or {}
    fn = ctx.get("fileName") or ctx.get("filename")
    if isinstance(fn, list):
        return [str(x) for x in fn]
    if isinstance(fn, str) and fn.strip():
        return [fn.strip()]
    return []


def strip_gemini_auto_memory_section(text: str) -> str:
    """Remove ``## Gemini Added Memories`` and following content."""
    m = _MEMORY_SECTION.search(text)
    if not m:
        return text
    return text[: m.start()].rstrip()


def flatten_memory(mem: HierarchicalMemory) -> str:
    """Format similar to Gemini CLI ``flattenMemory``."""
    parts: List[str] = []
    if mem.global_mem.strip():
        parts.append("--- Global ---\n" + mem.global_mem.strip())
    if mem.extension_mem.strip():
        parts.append("--- Extension ---\n" + mem.extension_mem.strip())
    if mem.project_mem.strip():
        parts.append("--- Project ---\n" + mem.project_mem.strip())
    return "\n\n".join(parts).strip()


def load_gemini_md_hierarchy(
    cwd: str = ".",
    filenames: Optional[Sequence[str]] = None,
    *,
    system_config_dir: Optional[str] = None,
    strip_auto_memory: bool = False,
) -> HierarchicalMemory:
    """Load tiered markdown; splits rough global vs project using user vs project walk."""
    names = list(filenames) if filenames else ["GEMINI.md", "AGENTS.md", "CLAUDE.md"]
    sys_dir = system_config_dir or _default_system_gemini_dir()
    system_dirs: List[str] = [sys_dir] if sys_dir else []
    user_dirs = [str(Path.home() / ".gemini")]

    # Global tier: system + user (concatenate via collect — we split heuristically below)
    global_blob = collect_md_hierarchy(
        cwd,
        tuple(names),
        system_dirs=tuple(system_dirs),
        user_dirs=tuple(user_dirs),
        project_walk=False,
        stop_at_git_root=False,
        resolve_imports=True,
        section_header_template="",
    )
    project_blob = collect_md_hierarchy(
        cwd,
        tuple(names),
        system_dirs=(),
        user_dirs=(),
        project_walk=True,
        stop_at_git_root=True,
        resolve_imports=True,
        section_header_template="",
    )
    if strip_auto_memory:
        global_blob = strip_gemini_auto_memory_section(global_blob)
        project_blob = strip_gemini_auto_memory_section(project_blob)

    return HierarchicalMemory(
        global_mem=global_blob,
        extension_mem="",
        project_mem=project_blob,
    )


def load_gemini_md_text(
    cwd: str = ".",
    filenames: Optional[Sequence[str]] = None,
    *,
    system_config_dir: Optional[str] = None,
    strip_auto_memory: bool = False,
    use_flatten_headers: bool = True,
) -> str:
    mem = load_gemini_md_hierarchy(
        cwd,
        filenames,
        system_config_dir=system_config_dir,
        strip_auto_memory=strip_auto_memory,
    )
    if use_flatten_headers:
        return flatten_memory(mem)
    return "\n\n".join(
        x for x in (mem.global_mem, mem.extension_mem, mem.project_mem) if x.strip()
    ).strip()


async def fetch_gemini_md(**kwargs: Any) -> str:
    from config.settings import AgentHarnessSettings

    h = AgentHarnessSettings()
    cwd = str(kwargs.get("cwd") or kwargs.get("gemini_cwd") or h.GEMINI_CLI_MD_CWD)
    names = list(h.GEMINI_CLI_MD_FILENAMES)
    extra = gemini_context_filenames_from_settings(cwd)
    if extra:
        names = extra + [n for n in names if n not in extra]
    text = load_gemini_md_text(
        cwd,
        names,
        system_config_dir=h.GEMINI_CLI_SYSTEM_CONFIG_DIR,
        strip_auto_memory=h.GEMINI_CLI_MD_STRIP_AUTO_MEMORY,
    )
    if h.GEMINI_CLI_MD_MAX_CHARS and len(text) > h.GEMINI_CLI_MD_MAX_CHARS:
        text = text[: h.GEMINI_CLI_MD_MAX_CHARS] + "\n[truncated]"
    return text
