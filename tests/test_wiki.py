"""
Tests for Orchestra wiki structure, frontmatter, wikilinks, backlink
consistency, index sync, and source tracking.

All tests use tmp_path fixtures to create isolated test wikis.
No real LLM calls.
"""

import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
TYPED_WIKILINK_RE = re.compile(r"\[\[(\w+):([^\]]+)\]\]")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)

VALID_LINK_TYPES = {"references", "depends_on", "extends", "contradicts", "related"}


def parse_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter from article text."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    result = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",") if x.strip()]
        result[k] = v
    return result


def extract_wikilinks(text: str) -> list[str]:
    """Extract all wikilink targets from text (both bare and typed)."""
    links = []
    for match in WIKILINK_RE.finditer(text):
        target = match.group(1)
        # Strip type prefix if present
        if ":" in target:
            target = target.split(":", 1)[1]
        links.append(target)
    return links


def create_article(wiki_dir: Path, section: str, slug: str,
                   title: str = "", tags: list[str] | None = None,
                   content: str = "", connections: list[str] | None = None) -> Path:
    """Create a wiki article file and return its path."""
    section_dir = wiki_dir / section
    section_dir.mkdir(parents=True, exist_ok=True)
    if not title:
        title = slug.replace("-", " ").title()
    if tags is None:
        tags = ["test"]
    tag_str = ", ".join(tags)
    conn_lines = ""
    if connections:
        conn_lines = "\n".join(f"- [[{c}]]" for c in connections)

    article = f"""---
title: {title}
tags: [{tag_str}]
updated: 2026-04-09
sources: [raw/manual/test.md]
---

# {title}

**Test article for {slug}.**

## Overview

{content or f"This is the overview for {slug}."}

## Key Claims

- Claim one.

## Connections

{conn_lines or "- No connections yet."}

## Sources

- [2026-04-09 -- Test](../../raw/manual/test.md)
"""
    filepath = section_dir / f"{slug}.md"
    filepath.write_text(article, encoding="utf-8")
    return filepath


def create_wiki(tmp_path: Path) -> Path:
    """Create a minimal valid wiki structure and return the wiki dir."""
    wiki_dir = tmp_path / "wiki"
    for section in ["concepts", "entities", "events", "research", "meta"]:
        (wiki_dir / section).mkdir(parents=True, exist_ok=True)

    # Create _index.md
    (wiki_dir / "_index.md").write_text(
        "# Wiki Index\n\nArticle count: 0\n", encoding="utf-8"
    )

    # Create _sources.json
    (wiki_dir / "_sources.json").write_text(
        json.dumps({"processed": {}}, indent=2), encoding="utf-8"
    )

    return wiki_dir


# ---------------------------------------------------------------------------
# Directory structure validation
# ---------------------------------------------------------------------------

class TestDirectoryStructure:
    def test_required_sections_exist(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        required = ["concepts", "entities", "events", "research"]
        for section in required:
            assert (wiki_dir / section).is_dir(), f"Missing section: {section}"

    def test_meta_dir_exists(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        assert (wiki_dir / "meta").is_dir()

    def test_index_file_exists(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        assert (wiki_dir / "_index.md").is_file()

    def test_sources_file_exists(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        assert (wiki_dir / "_sources.json").is_file()


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

class TestFrontmatter:
    def test_valid_frontmatter(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        path = create_article(wiki_dir, "concepts", "test-concept",
                              tags=["ai", "agents"])
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        assert fm["title"] == "Test Concept"
        assert "ai" in fm["tags"]
        assert "agents" in fm["tags"]
        assert fm["updated"] == "2026-04-09"
        assert isinstance(fm["sources"], list)

    def test_missing_frontmatter(self):
        text = "# No Frontmatter\n\nJust content."
        fm = parse_frontmatter(text)
        assert fm == {}

    def test_incomplete_frontmatter(self):
        text = "---\ntitle: Incomplete\n"
        fm = parse_frontmatter(text)
        assert fm == {}

    def test_required_fields_present(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        path = create_article(wiki_dir, "concepts", "full-article")
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        required = {"title", "tags", "updated", "sources"}
        assert required.issubset(set(fm.keys())), (
            f"Missing fields: {required - set(fm.keys())}"
        )


# ---------------------------------------------------------------------------
# _sources.json structure
# ---------------------------------------------------------------------------

class TestSourcesJson:
    def test_valid_structure(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        sources = json.loads((wiki_dir / "_sources.json").read_text(encoding="utf-8"))
        assert "processed" in sources
        assert isinstance(sources["processed"], dict)

    def test_entry_has_required_fields(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        sources = {
            "processed": {
                "raw/manual/test.md": {
                    "processed_at": "2026-04-09T12:00:00Z",
                    "articles": ["concepts/test-concept.md"],
                }
            }
        }
        (wiki_dir / "_sources.json").write_text(
            json.dumps(sources, indent=2), encoding="utf-8"
        )

        data = json.loads((wiki_dir / "_sources.json").read_text(encoding="utf-8"))
        for key, entry in data["processed"].items():
            assert "processed_at" in entry
            assert "articles" in entry
            assert isinstance(entry["articles"], list)

    def test_empty_processed_is_valid(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        data = json.loads((wiki_dir / "_sources.json").read_text(encoding="utf-8"))
        assert data["processed"] == {}


# ---------------------------------------------------------------------------
# Wikilink extraction (typed and bare)
# ---------------------------------------------------------------------------

class TestWikilinkExtraction:
    def test_bare_wikilink(self):
        text = "See [[llm-agents]] for details."
        links = extract_wikilinks(text)
        assert links == ["llm-agents"]

    def test_typed_wikilink(self):
        text = "This [[depends_on:llm-agents]] concept."
        links = extract_wikilinks(text)
        assert links == ["llm-agents"]

    def test_multiple_wikilinks(self):
        text = "See [[foo]], [[extends:bar]], and [[baz]]."
        links = extract_wikilinks(text)
        assert links == ["foo", "bar", "baz"]

    def test_no_wikilinks(self):
        text = "No links here, just plain text."
        links = extract_wikilinks(text)
        assert links == []

    def test_typed_link_types_valid(self):
        text = "[[depends_on:a]] [[extends:b]] [[contradicts:c]] [[related:d]] [[references:e]]"
        for match in TYPED_WIKILINK_RE.finditer(text):
            link_type = match.group(1)
            assert link_type in VALID_LINK_TYPES, f"Invalid type: {link_type}"

    def test_wikilinks_in_article(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        path = create_article(wiki_dir, "concepts", "agent-memory",
                              connections=["llm-agents", "extends:rag"])
        text = path.read_text(encoding="utf-8")
        links = extract_wikilinks(text)
        assert "llm-agents" in links
        assert "rag" in links


# ---------------------------------------------------------------------------
# Backlink consistency
# ---------------------------------------------------------------------------

class TestBacklinkConsistency:
    def test_all_wikilinks_resolve(self, tmp_path: Path):
        """Every wikilink target should correspond to an existing article."""
        wiki_dir = create_wiki(tmp_path)
        create_article(wiki_dir, "concepts", "alpha", connections=["beta"])
        create_article(wiki_dir, "concepts", "beta", connections=["alpha"])

        # Gather all slugs
        slugs = set()
        for md in wiki_dir.rglob("*.md"):
            if not md.name.startswith("_") and "meta" not in md.parts:
                slugs.add(md.stem)

        # Check all links resolve
        broken = []
        for md in wiki_dir.rglob("*.md"):
            if md.name.startswith("_") or "meta" in md.parts:
                continue
            text = md.read_text(encoding="utf-8")
            for target in extract_wikilinks(text):
                if target not in slugs:
                    broken.append((md.stem, target))

        assert broken == [], f"Broken links: {broken}"

    def test_broken_link_detected(self, tmp_path: Path):
        """A link to a nonexistent article should be caught."""
        wiki_dir = create_wiki(tmp_path)
        create_article(wiki_dir, "concepts", "alpha", connections=["nonexistent"])

        slugs = set()
        for md in wiki_dir.rglob("*.md"):
            if not md.name.startswith("_") and "meta" not in md.parts:
                slugs.add(md.stem)

        broken = []
        for md in wiki_dir.rglob("*.md"):
            if md.name.startswith("_") or "meta" in md.parts:
                continue
            for target in extract_wikilinks(md.read_text(encoding="utf-8")):
                if target not in slugs:
                    broken.append((md.stem, target))

        assert len(broken) == 1
        assert broken[0] == ("alpha", "nonexistent")


# ---------------------------------------------------------------------------
# Index-to-disk sync
# ---------------------------------------------------------------------------

class TestIndexSync:
    def test_index_refs_match_disk(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        create_article(wiki_dir, "concepts", "alpha")
        create_article(wiki_dir, "entities", "beta")

        # Write matching index
        index_text = (
            "# Wiki Index\n\n"
            "Article count: 2\n\n"
            "## Concepts\n\n"
            "**[[alpha]]** -- Test article.\n\n"
            "## Entities\n\n"
            "**[[beta]]** -- Test article.\n"
        )
        (wiki_dir / "_index.md").write_text(index_text, encoding="utf-8")

        index_refs = set(WIKILINK_RE.findall(
            (wiki_dir / "_index.md").read_text(encoding="utf-8")
        ))
        disk_slugs = set()
        for md in wiki_dir.rglob("*.md"):
            if not md.name.startswith("_") and "meta" not in md.parts:
                disk_slugs.add(md.stem)

        assert index_refs == disk_slugs

    def test_index_missing_article(self, tmp_path: Path):
        """Detect when an article exists on disk but not in the index."""
        wiki_dir = create_wiki(tmp_path)
        create_article(wiki_dir, "concepts", "alpha")
        create_article(wiki_dir, "concepts", "beta")

        index_text = "# Wiki Index\n\n**[[alpha]]** -- Test.\n"
        (wiki_dir / "_index.md").write_text(index_text, encoding="utf-8")

        index_refs = set(WIKILINK_RE.findall(
            (wiki_dir / "_index.md").read_text(encoding="utf-8")
        ))
        disk_slugs = set()
        for md in wiki_dir.rglob("*.md"):
            if not md.name.startswith("_") and "meta" not in md.parts:
                disk_slugs.add(md.stem)

        missing_from_index = disk_slugs - index_refs
        assert "beta" in missing_from_index

    def test_index_phantom_ref(self, tmp_path: Path):
        """Detect when the index references an article that does not exist."""
        wiki_dir = create_wiki(tmp_path)
        create_article(wiki_dir, "concepts", "alpha")

        index_text = "# Wiki Index\n\n**[[alpha]]** -- Test.\n**[[ghost]]** -- Gone.\n"
        (wiki_dir / "_index.md").write_text(index_text, encoding="utf-8")

        index_refs = set(WIKILINK_RE.findall(
            (wiki_dir / "_index.md").read_text(encoding="utf-8")
        ))
        disk_slugs = set()
        for md in wiki_dir.rglob("*.md"):
            if not md.name.startswith("_") and "meta" not in md.parts:
                disk_slugs.add(md.stem)

        phantom = index_refs - disk_slugs
        assert "ghost" in phantom


# ---------------------------------------------------------------------------
# No duplicate slugs
# ---------------------------------------------------------------------------

class TestNoDuplicateSlugs:
    def test_unique_slugs(self, tmp_path: Path):
        wiki_dir = create_wiki(tmp_path)
        create_article(wiki_dir, "concepts", "alpha")
        create_article(wiki_dir, "entities", "beta")

        slug_map: dict[str, list[Path]] = {}
        for md in wiki_dir.rglob("*.md"):
            if md.name.startswith("_") or "meta" in md.parts:
                continue
            slug_map.setdefault(md.stem, []).append(md)

        duplicates = {s: ps for s, ps in slug_map.items() if len(ps) > 1}
        assert duplicates == {}, f"Duplicate slugs: {duplicates}"

    def test_duplicate_detected(self, tmp_path: Path):
        """Same slug in two sections should be flagged."""
        wiki_dir = create_wiki(tmp_path)
        create_article(wiki_dir, "concepts", "overlap")
        create_article(wiki_dir, "research", "overlap")

        slug_map: dict[str, list[Path]] = {}
        for md in wiki_dir.rglob("*.md"):
            if md.name.startswith("_") or "meta" in md.parts:
                continue
            slug_map.setdefault(md.stem, []).append(md)

        duplicates = {s: ps for s, ps in slug_map.items() if len(ps) > 1}
        assert "overlap" in duplicates
        assert len(duplicates["overlap"]) == 2
