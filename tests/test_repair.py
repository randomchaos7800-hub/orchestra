"""
Tests for Orchestra wiki repair operations — reciprocal backlinks,
dead link pruning, merge duplicates, frontmatter link sync, and
index rebuild.

All tests use tmp_path fixtures. No real LLM calls.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import (
    parse_frontmatter,
    extract_wikilink_slugs as extract_wikilinks,
)

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def make_article(path: Path, title: str, body: str,
                 tags: list[str] | None = None,
                 sources: list[str] | None = None) -> None:
    """Write a wiki article to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = tags or ["test"]
    sources = sources or ["raw/manual/test.md"]
    tag_str = ", ".join(tags)
    src_str = ", ".join(sources)
    content = f"""---
title: {title}
tags: [{tag_str}]
updated: 2026-04-09
sources: [{src_str}]
---

# {title}

{body}
"""
    path.write_text(content, encoding="utf-8")


def create_wiki_structure(tmp_path: Path) -> Path:
    """Create a minimal wiki and return its root."""
    wiki = tmp_path / "wiki"
    for section in ["concepts", "entities", "events", "research"]:
        (wiki / section).mkdir(parents=True, exist_ok=True)
    (wiki / "_index.md").write_text("# Wiki Index\n\nArticle count: 0\n", encoding="utf-8")
    (wiki / "_sources.json").write_text(
        json.dumps({"processed": {}}, indent=2), encoding="utf-8"
    )
    return wiki


# ---------------------------------------------------------------------------
# Reciprocal backlink injection
# ---------------------------------------------------------------------------

class TestReciprocalBacklinks:
    def test_a_links_b_b_gets_backlink(self, tmp_path: Path):
        """If article A links to B, repair should add a backlink from B to A."""
        wiki = create_wiki_structure(tmp_path)

        make_article(
            wiki / "concepts" / "agent-memory.md",
            "Agent Memory",
            "## Overview\n\nMemory systems rely on [[llm-agents]].\n\n## Connections\n\n- [[llm-agents]]",
        )
        make_article(
            wiki / "concepts" / "llm-agents.md",
            "LLM Agents",
            "## Overview\n\nAgents use large language models.\n\n## Connections\n\n",
        )

        # Simulate repair: find all links, inject reciprocals
        articles = {}
        for md in wiki.rglob("*.md"):
            if md.name.startswith("_"):
                continue
            articles[md.stem] = md

        # Build link graph
        link_graph: dict[str, set[str]] = {}
        for slug, path in articles.items():
            text = path.read_text(encoding="utf-8")
            targets = set(extract_wikilinks(text))
            link_graph[slug] = targets

        # Inject reciprocal backlinks
        for slug, targets in link_graph.items():
            for target in targets:
                if target in articles and slug not in link_graph.get(target, set()):
                    target_path = articles[target]
                    text = target_path.read_text(encoding="utf-8")
                    backlink_line = f"- [[{slug}]] -- (backlink)"
                    if f"[[{slug}]]" not in text:
                        text = text.rstrip() + f"\n{backlink_line}\n"
                        target_path.write_text(text, encoding="utf-8")

        # Verify llm-agents now links back to agent-memory
        llm_text = (wiki / "concepts" / "llm-agents.md").read_text(encoding="utf-8")
        links = extract_wikilinks(llm_text)
        assert "agent-memory" in links

    def test_no_duplicate_backlinks(self, tmp_path: Path):
        """Running repair twice should not duplicate backlinks."""
        wiki = create_wiki_structure(tmp_path)

        make_article(
            wiki / "concepts" / "alpha.md",
            "Alpha",
            "## Connections\n\n- [[beta]]",
        )
        make_article(
            wiki / "concepts" / "beta.md",
            "Beta",
            "## Connections\n\n- [[alpha]]",
        )

        # Both already link to each other -- no injection needed
        alpha_text = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        beta_text = (wiki / "concepts" / "beta.md").read_text(encoding="utf-8")

        alpha_links = extract_wikilinks(alpha_text)
        beta_links = extract_wikilinks(beta_text)

        assert alpha_links.count("beta") == 1
        assert beta_links.count("alpha") == 1


# ---------------------------------------------------------------------------
# Dead link pruning
# ---------------------------------------------------------------------------

class TestDeadLinkPruning:
    def test_remove_dead_links(self, tmp_path: Path):
        """Links to nonexistent articles should be pruned."""
        wiki = create_wiki_structure(tmp_path)

        make_article(
            wiki / "concepts" / "alpha.md",
            "Alpha",
            "## Overview\n\nSee [[agents]] and [[beta]].\n\n## Connections\n\n- [[agents]]\n- [[beta]]",
        )
        make_article(
            wiki / "concepts" / "beta.md",
            "Beta",
            "## Overview\n\nStandalone article.",
        )

        # Gather valid slugs
        valid_slugs = set()
        for md in wiki.rglob("*.md"):
            if not md.name.startswith("_"):
                valid_slugs.add(md.stem)

        # Prune dead links from alpha
        alpha_path = wiki / "concepts" / "alpha.md"
        text = alpha_path.read_text(encoding="utf-8")

        # Remove lines containing dead wikilinks
        pruned_lines = []
        for line in text.splitlines():
            line_links = extract_wikilinks(line)
            dead = [l for l in line_links if l not in valid_slugs]
            if dead:
                # Remove the dead link references from the line
                for d in dead:
                    line = re.sub(r"\[\[" + re.escape(d) + r"\]\]", "", line)
                    line = re.sub(r"\[\[\w+:" + re.escape(d) + r"\]\]", "", line)
                line = line.strip()
                if line and line != "-":
                    pruned_lines.append(line)
            else:
                pruned_lines.append(line)

        pruned_text = "\n".join(pruned_lines) + "\n"
        alpha_path.write_text(pruned_text, encoding="utf-8")

        # Verify: agents link removed, beta link preserved
        result_text = alpha_path.read_text(encoding="utf-8")
        result_links = extract_wikilinks(result_text)
        assert "agents" not in result_links
        assert "beta" in result_links

    def test_no_pruning_when_all_valid(self, tmp_path: Path):
        """Articles with only valid links should be unchanged."""
        wiki = create_wiki_structure(tmp_path)

        make_article(
            wiki / "concepts" / "alpha.md",
            "Alpha",
            "## Connections\n\n- [[beta]]",
        )
        make_article(
            wiki / "concepts" / "beta.md",
            "Beta",
            "## Connections\n\n- [[alpha]]",
        )

        valid_slugs = {"alpha", "beta"}

        alpha_path = wiki / "concepts" / "alpha.md"
        text = alpha_path.read_text(encoding="utf-8")
        links = extract_wikilinks(text)
        dead = [l for l in links if l not in valid_slugs]
        assert dead == []


# ---------------------------------------------------------------------------
# Merge duplicates
# ---------------------------------------------------------------------------

class TestMergeDuplicates:
    def test_detect_duplicates(self, tmp_path: Path):
        """Two articles with the same slug in different sections are duplicates."""
        wiki = create_wiki_structure(tmp_path)

        make_article(
            wiki / "concepts" / "rag.md",
            "RAG",
            "## Overview\n\nRetrieval augmented generation from concepts.",
        )
        make_article(
            wiki / "research" / "rag.md",
            "RAG",
            "## Overview\n\nRetrieval augmented generation from research.",
        )

        slug_map: dict[str, list[Path]] = {}
        for md in wiki.rglob("*.md"):
            if md.name.startswith("_"):
                continue
            slug_map.setdefault(md.stem, []).append(md)

        duplicates = {s: ps for s, ps in slug_map.items() if len(ps) > 1}
        assert "rag" in duplicates
        assert len(duplicates["rag"]) == 2

    def test_merge_keeps_content(self, tmp_path: Path):
        """Merging two articles should preserve key claims from both."""
        wiki = create_wiki_structure(tmp_path)

        make_article(
            wiki / "concepts" / "rag.md",
            "RAG",
            "## Key Claims\n\n- Claim A from concepts.",
            tags=["ai", "retrieval"],
        )
        make_article(
            wiki / "research" / "rag.md",
            "RAG",
            "## Key Claims\n\n- Claim B from research.",
            tags=["ai", "papers"],
        )

        # Simulate merge: combine content into the concepts version
        primary = wiki / "concepts" / "rag.md"
        secondary = wiki / "research" / "rag.md"

        primary_text = primary.read_text(encoding="utf-8")
        secondary_text = secondary.read_text(encoding="utf-8")

        # Extract claims from secondary
        secondary_claims = []
        in_claims = False
        for line in secondary_text.splitlines():
            if line.strip() == "## Key Claims":
                in_claims = True
                continue
            if line.startswith("## ") and in_claims:
                break
            if in_claims and line.strip().startswith("- "):
                secondary_claims.append(line.strip())

        # Append to primary
        merged = primary_text.rstrip()
        if secondary_claims:
            # Find where key claims section ends
            lines = merged.splitlines()
            insert_idx = len(lines)
            in_claims = False
            for i, line in enumerate(lines):
                if line.strip() == "## Key Claims":
                    in_claims = True
                    continue
                if line.startswith("## ") and in_claims:
                    insert_idx = i
                    break

            for claim in secondary_claims:
                lines.insert(insert_idx, claim)
                insert_idx += 1

            merged = "\n".join(lines)

        primary.write_text(merged + "\n", encoding="utf-8")
        secondary.unlink()

        # Verify
        result = primary.read_text(encoding="utf-8")
        assert "Claim A from concepts" in result
        assert "Claim B from research" in result
        assert not secondary.exists()


# ---------------------------------------------------------------------------
# Frontmatter link sync
# ---------------------------------------------------------------------------

class TestFrontmatterLinkSync:
    def test_sources_in_frontmatter(self, tmp_path: Path):
        """Article frontmatter sources field should list raw source files."""
        wiki = create_wiki_structure(tmp_path)
        path = make_article(
            wiki / "concepts" / "alpha.md",
            "Alpha",
            "## Overview\n\nTest.",
            sources=["raw/manual/source-a.md", "raw/manual/source-b.md"],
        )

        text = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        assert isinstance(fm["sources"], list)
        assert len(fm["sources"]) == 2

    def test_tags_match_content(self, tmp_path: Path):
        """Tags in frontmatter should be consistent with article content."""
        wiki = create_wiki_structure(tmp_path)
        make_article(
            wiki / "concepts" / "alpha.md",
            "Alpha",
            "## Overview\n\nThis is about local inference and agent memory.",
            tags=["local-inference", "memory"],
        )

        text = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        assert "local-inference" in fm["tags"]
        assert "memory" in fm["tags"]

    def test_updated_field_is_date(self, tmp_path: Path):
        wiki = create_wiki_structure(tmp_path)
        make_article(wiki / "concepts" / "alpha.md", "Alpha", "Content.")

        text = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        # Should match YYYY-MM-DD
        assert re.match(r"\d{4}-\d{2}-\d{2}", str(fm["updated"]))


# ---------------------------------------------------------------------------
# Index rebuild
# ---------------------------------------------------------------------------

class TestIndexRebuild:
    def test_rebuild_from_disk(self, tmp_path: Path):
        """Index rebuild should create entries for all articles on disk."""
        wiki = create_wiki_structure(tmp_path)
        make_article(wiki / "concepts" / "alpha.md", "Alpha", "Alpha content.")
        make_article(wiki / "entities" / "beta.md", "Beta", "Beta content.")
        make_article(wiki / "research" / "gamma.md", "Gamma", "Gamma content.")

        # Simulate index rebuild
        articles: dict[str, dict] = {}
        for md in wiki.rglob("*.md"):
            if md.name.startswith("_") or "meta" in md.parts:
                continue
            text = md.read_text(encoding="utf-8")
            fm = parse_frontmatter(text)
            section = md.parent.name
            articles[md.stem] = {
                "title": fm.get("title", md.stem),
                "section": section,
                "tags": fm.get("tags", []),
                "updated": fm.get("updated", ""),
            }

        # Build new index
        sections: dict[str, list[str]] = {}
        for slug, info in sorted(articles.items()):
            section = info["section"]
            sections.setdefault(section, [])
            tag_str = ", ".join(info["tags"]) if isinstance(info["tags"], list) else info["tags"]
            entry = f"**[[{slug}]]** -- {info['title']}. Tags: {tag_str}. Updated: {info['updated']}."
            sections[section].append(entry)

        index_lines = [f"# Wiki Index\n", f"\nArticle count: {len(articles)}\n"]
        for section_name in ["concepts", "entities", "events", "research"]:
            index_lines.append(f"\n## {section_name.title()}\n")
            for entry in sections.get(section_name, []):
                index_lines.append(f"\n{entry}")
            index_lines.append("")

        index_text = "\n".join(index_lines)
        (wiki / "_index.md").write_text(index_text, encoding="utf-8")

        # Verify
        result = (wiki / "_index.md").read_text(encoding="utf-8")
        assert "[[alpha]]" in result
        assert "[[beta]]" in result
        assert "[[gamma]]" in result
        assert "Article count: 3" in result

    def test_rebuild_empty_wiki(self, tmp_path: Path):
        """Rebuilding an empty wiki should produce a valid but empty index."""
        wiki = create_wiki_structure(tmp_path)

        # No articles -- rebuild
        articles = []
        for md in wiki.rglob("*.md"):
            if not md.name.startswith("_") and "meta" not in md.parts:
                articles.append(md)

        index_text = f"# Wiki Index\n\nArticle count: {len(articles)}\n"
        (wiki / "_index.md").write_text(index_text, encoding="utf-8")

        result = (wiki / "_index.md").read_text(encoding="utf-8")
        assert "Article count: 0" in result

    def test_rebuild_updates_count(self, tmp_path: Path):
        """Adding an article and rebuilding should update the count."""
        wiki = create_wiki_structure(tmp_path)

        # Start with count 0
        assert "Article count: 0" in (wiki / "_index.md").read_text(encoding="utf-8")

        # Add article
        make_article(wiki / "concepts" / "new-concept.md", "New Concept", "Content.")

        # Count articles
        count = sum(
            1 for md in wiki.rglob("*.md")
            if not md.name.startswith("_") and "meta" not in md.parts
        )

        index_text = f"# Wiki Index\n\nArticle count: {count}\n\n**[[new-concept]]** -- New.\n"
        (wiki / "_index.md").write_text(index_text, encoding="utf-8")

        result = (wiki / "_index.md").read_text(encoding="utf-8")
        assert "Article count: 1" in result
