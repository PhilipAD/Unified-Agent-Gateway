"""Tests for Gemini-style hierarchical markdown loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from context.gemini_md import (
    gemini_context_filenames_from_settings,
    load_gemini_md_text,
    strip_gemini_auto_memory_section,
)
from context.md_hierarchy import collect_md_hierarchy


def test_collect_md_hierarchy_project_walk(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("root agents", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("pkg agents", encoding="utf-8")
    text = collect_md_hierarchy(
        str(sub),
        ("AGENTS.md",),
        project_walk=True,
        stop_at_git_root=False,
    )
    assert "pkg agents" in text
    assert "root agents" in text


def test_strip_gemini_auto_memory() -> None:
    raw = "Hello\n\n## Gemini Added Memories\n\n- foo"
    assert "foo" not in strip_gemini_auto_memory_section(raw)
    assert "Hello" in strip_gemini_auto_memory_section(raw)


def test_gemini_settings_context_filename(tmp_path: Path) -> None:
    gem = tmp_path / ".gemini"
    gem.mkdir()
    (gem / "settings.json").write_text(
        '{"context": {"fileName": "CUSTOM.md"}}',
        encoding="utf-8",
    )
    names = gemini_context_filenames_from_settings(str(tmp_path))
    assert names == ["CUSTOM.md"]


def test_load_gemini_md_text_minimal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    (fake_home / ".gemini").mkdir()
    (tmp_path / "GEMINI.md").write_text("# Project gemini", encoding="utf-8")
    text = load_gemini_md_text(str(tmp_path), ("GEMINI.md",), use_flatten_headers=True)
    assert "Project gemini" in text
    assert "--- Project ---" in text
