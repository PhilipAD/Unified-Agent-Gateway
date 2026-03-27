"""Shared hierarchical markdown collection for agent harness context.

Walks system/user/project directories, optionally up to a git root, resolves
``@relative/path.md`` imports (single pass, relative to the source file), and
concatenates discovered files with section headers.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

_IMPORT_LINE = re.compile(r"^@([^\s#][^\s#]*)$", re.MULTILINE)
_MAX_IMPORT_DEPTH = 20


def _expand_path(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path))).resolve()


def find_git_root(start: Path, max_steps: int = 64) -> Optional[Path]:
    """Return the directory containing ``.git`` when walking parents from *start*."""
    cur = start.resolve()
    for _ in range(max_steps):
        if (cur / ".git").exists():
            return cur
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent
    return None


def _ancestor_chain(cwd: Path, stop_at_git_root: bool, max_depth: int) -> List[Path]:
    """Return *cwd* then each parent, optionally capped at git root."""
    chain: List[Path] = []
    cur = cwd.resolve()
    git_root = find_git_root(cur) if stop_at_git_root else None
    for _ in range(max_depth):
        chain.append(cur)
        if git_root is not None and cur == git_root:
            break
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return chain


def _read_file_safe(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _resolve_imports(
    content: str,
    base_dir: Path,
    seen: Set[Path],
    depth: int,
) -> str:
    """Inline ``@file.md`` lines relative to *base_dir* (Gemini-style)."""
    if depth > _MAX_IMPORT_DEPTH:
        return content

    def repl(match: re.Match[str]) -> str:
        rel = match.group(1).strip()
        if not rel or rel.startswith("#"):
            return match.group(0)
        base_resolved = base_dir.resolve()
        target = (base_resolved / rel).resolve()
        try:
            target.relative_to(base_resolved)
        except ValueError:
            return match.group(0)
        if target in seen:
            return ""
        if not target.is_file():
            logger.debug("Import not found: %s", target)
            return ""
        seen.add(target)
        inner = _read_file_safe(target) or ""
        inner = _resolve_imports(inner, target.parent, seen, depth + 1)
        return f"\n<!-- imported: {rel} -->\n{inner}\n"

    return _IMPORT_LINE.sub(repl, content)


def collect_md_hierarchy(
    cwd: str,
    filenames: Sequence[str],
    *,
    system_dirs: Sequence[str] = (),
    user_dirs: Sequence[str] = (),
    project_walk: bool = True,
    stop_at_git_root: bool = True,
    resolve_imports: bool = True,
    max_depth: int = 20,
    section_header_template: str = "--- {label}: {path} ---\n",
) -> str:
    """Collect markdown files from configured tiers and concatenate with headers.

    *system_dirs* and *user_dirs* are scanned for each *filename* (first match per dir).
    When *project_walk* is true, each ancestor of *cwd* (optionally stopping at git root)
    is scanned for each *filename*.
    """
    root_cwd = _expand_path(cwd)
    parts: List[str] = []
    emitted: Set[Tuple[str, str]] = set()

    def emit(label: str, path: Path, text: str) -> None:
        key = (label, str(path))
        if key in emitted:
            return
        emitted.add(key)
        body = text
        if resolve_imports:
            body = _resolve_imports(body, path.parent, {path}, 0)
        parts.append(section_header_template.format(label=label, path=path))
        parts.append(body.rstrip())
        parts.append("")

    # System tier
    for d in system_dirs:
        base = Path(os.path.expanduser(os.path.expandvars(d)))
        if not base.is_dir():
            continue
        for name in filenames:
            p = base / name
            if p.is_file():
                raw = _read_file_safe(p)
                if raw is not None:
                    emit("system", p, raw)

    # User tier
    for d in user_dirs:
        base = Path(os.path.expanduser(os.path.expandvars(d)))
        if not base.is_dir():
            continue
        for name in filenames:
            p = base / name
            if p.is_file():
                raw = _read_file_safe(p)
                if raw is not None:
                    emit("user", p, raw)

    # Project tier (walk up)
    if project_walk:
        for ancestor in _ancestor_chain(root_cwd, stop_at_git_root, max_depth):
            for name in filenames:
                p = ancestor / name
                if p.is_file():
                    raw = _read_file_safe(p)
                    if raw is not None:
                        emit("project", p, raw)

    return "\n".join(parts).strip()


def collect_glob_files_in_dirs(
    directories: Iterable[Path],
    glob_pattern: str,
    *,
    resolve_imports: bool = True,
    section_header_template: str = "--- {label}: {path} ---\n",
) -> str:
    """Collect all files matching *glob_pattern* under each directory in *directories*."""
    parts: List[str] = []
    seen_paths: Set[Path] = set()
    for base in directories:
        if not base.is_dir():
            continue
        for path in sorted(base.glob(glob_pattern)):
            if not path.is_file() or path in seen_paths:
                continue
            seen_paths.add(path)
            raw = _read_file_safe(path)
            if raw is None:
                continue
            body = raw
            if resolve_imports:
                body = _resolve_imports(body, path.parent, {path}, 0)
            parts.append(section_header_template.format(label="file", path=path))
            parts.append(body.rstrip())
            parts.append("")
    return "\n".join(parts).strip()
