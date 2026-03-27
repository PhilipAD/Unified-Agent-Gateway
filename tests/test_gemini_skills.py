"""Tests for Gemini skill discovery."""

from __future__ import annotations

from pathlib import Path

from context.gemini_skills import discover_skills, format_skills_catalog


def test_discover_skills_workspace(tmp_path: Path) -> None:
    skills = tmp_path / ".gemini" / "skills" / "demo"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A demo\n---\nBody",
        encoding="utf-8",
    )
    found = discover_skills(
        str(tmp_path),
        user_skill_roots=(),
        project_skill_roots=(".gemini/skills",),
    )
    assert len(found) == 1
    assert found[0].name == "demo-skill"
    cat = format_skills_catalog(found)
    assert "demo-skill" in cat
