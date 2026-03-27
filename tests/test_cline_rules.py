"""Tests for Cline rules loader."""

from __future__ import annotations

from pathlib import Path

from context.cline_rules import load_cline_rules


def test_cline_rules_hierarchy(tmp_path: Path) -> None:
    (tmp_path / ".clinerules").write_text("use ruff", encoding="utf-8")
    sub = tmp_path / "app"
    sub.mkdir()
    (sub / ".clinerules.md").write_text("app rules", encoding="utf-8")
    text = load_cline_rules(str(sub))
    assert "app rules" in text
    assert "use ruff" in text
