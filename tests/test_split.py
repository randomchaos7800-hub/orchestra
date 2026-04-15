"""Tests for tools/split.py — project file splitting by year."""

import pytest
from pathlib import Path

# Adjust sys.path so we can import the split module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _make_project_file(projects_dir: Path, name: str, entries: list[tuple[str, str]]) -> Path:
    """Create a project file with entries. Each entry is (date, title)."""
    f = projects_dir / f"{name}.md"
    lines = [f"# {name}\n\n"]
    for date, title in entries:
        lines.append(f"## {date} - {title}\n")
        lines.append(f"**Source:** Test conversation\n\n")
        lines.append("Content for this entry. " * 20 + "\n\n")
        lines.append("---\n\n")
    f.write_text("".join(lines), encoding="utf-8")
    return f


class TestParseEntries:
    def test_entries_detected(self, tmp_path):
        from split import _parse_entries
        content = "# PROJECTS\n\n## 2025-06-01 - First\nContent\n\n---\n\n## 2026-01-15 - Second\nContent\n"
        entries = _parse_entries(content)
        assert len(entries) == 2
        assert entries[0]["year"] == "2025"
        assert entries[1]["year"] == "2026"

    def test_no_entries(self, tmp_path):
        from split import _parse_entries
        entries = _parse_entries("# Empty file\n\nNo entries here.\n")
        assert entries == []

    def test_single_year(self, tmp_path):
        from split import _parse_entries
        content = "## 2026-01-01 - A\nX\n## 2026-02-01 - B\nY\n"
        entries = _parse_entries(content)
        assert len(entries) == 2
        assert all(e["year"] == "2026" for e in entries)


class TestGetPreamble:
    def test_preamble_before_entries(self):
        from split import _get_preamble
        content = "# PROJECTS\n\nSome description.\n\n## 2026-01-01 - Entry\nContent\n"
        preamble = _get_preamble(content)
        assert "# PROJECTS" in preamble
        assert "## 2026" not in preamble

    def test_no_entries_returns_all(self):
        from split import _get_preamble
        content = "# PROJECTS\n\nJust a header.\n"
        assert _get_preamble(content) == content


class TestSplitFile:
    def test_below_threshold_skips(self, tmp_path):
        from split import split_file
        projects = tmp_path / "projects"
        projects.mkdir()
        f = _make_project_file(projects, "SMALL", [("2026-01-01", "Entry")])
        # File is small, threshold is huge
        stats = split_file(f, threshold_kb=99999)
        assert not stats["split"]

    def test_single_year_skips(self, tmp_path):
        from split import split_file
        projects = tmp_path / "projects"
        projects.mkdir()
        # Many entries but all same year
        entries = [(f"2026-{m:02d}-01", f"Entry {m}") for m in range(1, 13)]
        f = _make_project_file(projects, "SAMEYEAR", entries)
        stats = split_file(f, threshold_kb=0)  # threshold 0 = always try
        assert not stats["split"]

    def test_multi_year_splits(self, tmp_path, monkeypatch):
        from split import split_file
        import split as split_mod
        monkeypatch.setattr(split_mod, "PROJECTS_DIR", tmp_path / "projects")

        projects = tmp_path / "projects"
        projects.mkdir()
        entries = [
            ("2024-06-15", "Old entry 1"),
            ("2024-09-01", "Old entry 2"),
            ("2025-03-10", "Mid entry"),
            ("2026-01-20", "Current entry"),
            ("2026-04-01", "Recent entry"),
        ]
        f = _make_project_file(projects, "TEST", entries)
        stats = split_file(f, threshold_kb=0)

        assert stats["split"]
        # 2024 and 2025 should be archived
        assert (projects / "archive" / "2024" / "TEST.md").exists()
        assert (projects / "archive" / "2025" / "TEST.md").exists()
        # Main file should only have 2026 entries
        remaining = f.read_text(encoding="utf-8")
        assert "2026-01-20" in remaining
        assert "2026-04-01" in remaining
        assert "2024-06-15" not in remaining
        assert "2025-03-10" not in remaining

    def test_dry_run_no_writes(self, tmp_path, monkeypatch):
        from split import split_file
        import split as split_mod
        monkeypatch.setattr(split_mod, "PROJECTS_DIR", tmp_path / "projects")

        projects = tmp_path / "projects"
        projects.mkdir()
        entries = [("2024-01-01", "Old"), ("2026-01-01", "New")]
        f = _make_project_file(projects, "DRYTEST", entries)
        original = f.read_text()

        split_file(f, threshold_kb=0, dry_run=True)

        # File should be unchanged
        assert f.read_text() == original
        # No archive created
        assert not (projects / "archive").exists()
