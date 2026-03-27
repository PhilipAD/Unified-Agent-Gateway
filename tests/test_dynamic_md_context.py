"""Tests for DynamicContext md_hierarchy / md_files / md_glob modes and
per-request filenames overrides on harness fetch functions."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# md_hierarchy mode via DynamicContext / _compose_registries integration
# ---------------------------------------------------------------------------


class TestMdHierarchyMode:
    def test_collect_md_hierarchy_custom_filenames(self, tmp_path: Path) -> None:
        """md_hierarchy mode reads caller-specified filenames from cwd ancestors."""
        from context.md_hierarchy import collect_md_hierarchy

        proj = tmp_path / "project"
        proj.mkdir()
        rules = proj / "MY_RULES.md"
        rules.write_text("# Team rules\nNo magic numbers.", encoding="utf-8")

        result = collect_md_hierarchy(
            cwd=str(proj),
            filenames=["MY_RULES.md"],
            stop_at_git_root=False,
        )
        assert "Team rules" in result
        assert "No magic numbers" in result

    def test_collect_md_hierarchy_multiple_filenames(self, tmp_path: Path) -> None:
        """Multiple filenames are each searched in the ancestor tree."""
        from context.md_hierarchy import collect_md_hierarchy

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "RULES.md").write_text("Rule A", encoding="utf-8")
        (ws / "STANDARDS.md").write_text("Standard B", encoding="utf-8")

        result = collect_md_hierarchy(
            cwd=str(ws),
            filenames=["RULES.md", "STANDARDS.md"],
            stop_at_git_root=False,
        )
        assert "Rule A" in result
        assert "Standard B" in result

    def test_collect_md_hierarchy_system_and_user_dirs(self, tmp_path: Path) -> None:
        """system_dirs and user_dirs are scanned before the cwd walk."""
        from context.md_hierarchy import collect_md_hierarchy

        sys_dir = tmp_path / "system"
        sys_dir.mkdir()
        (sys_dir / "GLOBAL.md").write_text("Global policy", encoding="utf-8")

        user_dir = tmp_path / "user"
        user_dir.mkdir()
        (user_dir / "GLOBAL.md").write_text("User override", encoding="utf-8")

        ws = tmp_path / "ws"
        ws.mkdir()

        result = collect_md_hierarchy(
            cwd=str(ws),
            filenames=["GLOBAL.md"],
            system_dirs=[str(sys_dir)],
            user_dirs=[str(user_dir)],
            project_walk=False,
            stop_at_git_root=False,
        )
        assert "Global policy" in result
        assert "User override" in result

    def test_collect_md_hierarchy_missing_file_silent(self, tmp_path: Path) -> None:
        """Missing filenames are silently skipped, returning empty string."""
        from context.md_hierarchy import collect_md_hierarchy

        result = collect_md_hierarchy(
            cwd=str(tmp_path),
            filenames=["NONEXISTENT.md"],
            stop_at_git_root=False,
        )
        assert result == ""


# ---------------------------------------------------------------------------
# md_files mode
# ---------------------------------------------------------------------------


class TestMdFilesMode:
    @pytest.mark.asyncio
    async def test_md_files_explicit_paths(self, tmp_path: Path) -> None:
        """md_files fetcher reads explicit absolute paths."""
        f1 = tmp_path / "A.md"
        f1.write_text("Content A", encoding="utf-8")
        f2 = tmp_path / "B.md"
        f2.write_text("Content B", encoding="utf-8")

        # Simulate the fetch closure created in _compose_registries
        import os

        from context.md_hierarchy import _read_file_safe  # noqa: PLC2701

        paths = [str(f1), str(f2)]

        async def fetch_md_files(**kwargs):
            base = Path(str(kwargs.get("cwd") or ".")).resolve()
            parts = []
            for raw_path in paths:
                p = Path(os.path.expandvars(os.path.expanduser(raw_path)))
                if not p.is_absolute():
                    p = base / p
                p = p.resolve()
                text = _read_file_safe(p)
                if text is not None:
                    parts.append(f"--- file: {p} ---\n{text.rstrip()}")
            return "\n\n".join(parts)

        result = await fetch_md_files(cwd=str(tmp_path))
        assert "Content A" in result
        assert "Content B" in result

    @pytest.mark.asyncio
    async def test_md_files_relative_paths(self, tmp_path: Path) -> None:
        """md_files relative paths are resolved from cwd kwarg."""
        (tmp_path / "rel.md").write_text("Relative content", encoding="utf-8")

        import os

        from context.md_hierarchy import _read_file_safe  # noqa: PLC2701

        paths = ["rel.md"]

        async def fetch_md_files(**kwargs):
            base = Path(str(kwargs.get("cwd") or ".")).resolve()
            parts = []
            for raw_path in paths:
                p = Path(os.path.expandvars(os.path.expanduser(raw_path)))
                if not p.is_absolute():
                    p = base / p
                p = p.resolve()
                text = _read_file_safe(p)
                if text is not None:
                    parts.append(f"--- file: {p} ---\n{text.rstrip()}")
            return "\n\n".join(parts)

        result = await fetch_md_files(cwd=str(tmp_path))
        assert "Relative content" in result

    @pytest.mark.asyncio
    async def test_md_files_missing_path_skipped(self, tmp_path: Path) -> None:
        """Missing paths are silently skipped."""
        import os

        from context.md_hierarchy import _read_file_safe  # noqa: PLC2701

        paths = [str(tmp_path / "does_not_exist.md")]

        async def fetch_md_files(**kwargs):
            base = Path(str(kwargs.get("cwd") or ".")).resolve()
            parts = []
            for raw_path in paths:
                p = Path(os.path.expandvars(os.path.expanduser(raw_path)))
                if not p.is_absolute():
                    p = base / p
                p = p.resolve()
                text = _read_file_safe(p)
                if text is not None:
                    parts.append(f"--- file: {p} ---\n{text.rstrip()}")
            return "\n\n".join(parts)

        result = await fetch_md_files(cwd=str(tmp_path))
        assert result == ""


# ---------------------------------------------------------------------------
# md_glob mode
# ---------------------------------------------------------------------------


class TestMdGlobMode:
    def test_collect_glob_files_in_dirs(self, tmp_path: Path) -> None:
        """collect_glob_files_in_dirs returns all matching files."""
        from context.md_hierarchy import collect_glob_files_in_dirs

        rules = tmp_path / "rules"
        rules.mkdir()
        (rules / "first.md").write_text("Rule one", encoding="utf-8")
        (rules / "second.md").write_text("Rule two", encoding="utf-8")
        (rules / "skip.txt").write_text("Not included", encoding="utf-8")

        result = collect_glob_files_in_dirs([rules], "*.md", resolve_imports=False)
        assert "Rule one" in result
        assert "Rule two" in result
        assert "Not included" not in result

    def test_collect_glob_files_missing_dir(self, tmp_path: Path) -> None:
        """Missing glob_dirs are silently skipped."""
        from context.md_hierarchy import collect_glob_files_in_dirs

        result = collect_glob_files_in_dirs([tmp_path / "nonexistent"], "*.md")
        assert result == ""


# ---------------------------------------------------------------------------
# Per-request filenames override on harness fetchers
# ---------------------------------------------------------------------------


class TestHarnessFetcherKwargs:
    @pytest.mark.asyncio
    async def test_agents_md_custom_filenames(self, tmp_path: Path) -> None:
        """fetch_agents_md respects agents_md_filenames kwarg."""
        from context.agents_md import fetch_agents_md

        (tmp_path / "CUSTOM.md").write_text("Custom rules content", encoding="utf-8")

        result = await fetch_agents_md(
            cwd=str(tmp_path),
            agents_md_filenames=["CUSTOM.md"],
        )
        assert "Custom rules content" in result

    @pytest.mark.asyncio
    async def test_agents_md_default_filename_not_present(self, tmp_path: Path) -> None:
        """fetch_agents_md returns empty string when AGENTS.md not present."""
        from context.agents_md import fetch_agents_md

        result = await fetch_agents_md(cwd=str(tmp_path))
        assert result == ""

    @pytest.mark.asyncio
    async def test_cline_custom_filenames(self, tmp_path: Path) -> None:
        """fetch_cline_rules respects cline_filenames kwarg."""
        from context.cline_rules import fetch_cline_rules

        (tmp_path / "MYRULES").write_text("Cline custom rule", encoding="utf-8")

        result = await fetch_cline_rules(
            cwd=str(tmp_path),
            cline_filenames=["MYRULES"],
        )
        assert "Cline custom rule" in result

    @pytest.mark.asyncio
    async def test_gemini_md_custom_filenames(self, tmp_path: Path) -> None:
        """fetch_gemini_md respects gemini_filenames kwarg."""
        from context.gemini_md import fetch_gemini_md

        (tmp_path / "TEAM.md").write_text("Team memory", encoding="utf-8")

        result = await fetch_gemini_md(
            cwd=str(tmp_path),
            gemini_filenames=["TEAM.md"],
        )
        assert "Team memory" in result

    @pytest.mark.asyncio
    async def test_gemini_md_extra_filenames_merged(self, tmp_path: Path) -> None:
        """fetch_gemini_md gemini_extra_filenames are prepended to default list."""
        from context.gemini_md import fetch_gemini_md

        (tmp_path / "EXTRA.md").write_text("Extra memory", encoding="utf-8")

        result = await fetch_gemini_md(
            cwd=str(tmp_path),
            gemini_extra_filenames=["EXTRA.md"],
        )
        assert "Extra memory" in result

    @pytest.mark.asyncio
    async def test_gemini_skills_custom_roots(self, tmp_path: Path) -> None:
        """fetch_gemini_skills_catalog respects skills_project_roots kwarg."""
        from context.gemini_skills import fetch_gemini_skills_catalog

        skills_dir = tmp_path / "my_skills" / "demo"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: demo\ndescription: A demo skill\n---\n",
            encoding="utf-8",
        )

        result = await fetch_gemini_skills_catalog(
            cwd=str(tmp_path),
            skills_project_roots=["my_skills"],
            skills_user_roots=(),
        )
        assert "demo" in result

    @pytest.mark.asyncio
    async def test_windsurf_extra_rule_dirs(self, tmp_path: Path) -> None:
        """fetch_windsurf_rules appends extra_rule_dirs content."""
        from context.windsurf_rules import fetch_windsurf_rules

        extra = tmp_path / "extra_rules"
        extra.mkdir()
        (extra / "custom.md").write_text("Custom windsurf rule", encoding="utf-8")

        result = await fetch_windsurf_rules(
            cwd=str(tmp_path),
            windsurf_extra_rule_dirs=[str(extra)],
        )
        assert "Custom windsurf rule" in result


# ---------------------------------------------------------------------------
# CUSTOM_MD_FILENAMES env-based global context
# ---------------------------------------------------------------------------


class TestCustomMdSettings:
    def test_agents_md_load_custom_filename(self, tmp_path: Path) -> None:
        """load_agents_md with custom filenames picks up any named file."""
        from context.agents_md import load_agents_md

        (tmp_path / "PROJECT.md").write_text("Project standards", encoding="utf-8")

        result = load_agents_md(str(tmp_path), filenames=["PROJECT.md"])
        assert "Project standards" in result

    def test_cline_rules_load_custom_filename(self, tmp_path: Path) -> None:
        """load_cline_rules with custom filenames picks up any named file."""
        from context.cline_rules import load_cline_rules

        (tmp_path / "RULES").write_text("Custom cline rules", encoding="utf-8")

        result = load_cline_rules(str(tmp_path), filenames=["RULES"])
        assert "Custom cline rules" in result

    def test_collect_md_hierarchy_user_dir_override(self, tmp_path: Path) -> None:
        """user_dirs are scanned and their content appears in result."""
        from context.md_hierarchy import collect_md_hierarchy

        user_dir = tmp_path / "user_home"
        user_dir.mkdir()
        (user_dir / "SHARED.md").write_text("Shared user doc", encoding="utf-8")

        ws = tmp_path / "ws"
        ws.mkdir()

        result = collect_md_hierarchy(
            cwd=str(ws),
            filenames=["SHARED.md"],
            user_dirs=[str(user_dir)],
            stop_at_git_root=False,
        )
        assert "Shared user doc" in result
