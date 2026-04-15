#!/usr/bin/env python3
"""
Wiki Health Report -- tools/health.py

Read-only audit of the knowledge base. Never modifies files.

Usage:
  python3 tools/health.py              # report to stdout
  python3 tools/health.py --write      # stdout + write wiki/meta/stale.md, orphans.md
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

KB_ROOT = Path(__file__).parent.parent
WIKI_DIR = KB_ROOT / "wiki"
RAW_DIR = KB_ROOT / "raw"
SOURCES_FILE = WIKI_DIR / "_sources.json"

LINK_TYPES = ["references", "depends_on", "extends", "contradicts", "related"]
WIKI_SECTIONS_DEFAULT = ["concepts", "entities", "events", "research"]


# -- Helpers (minimal copies -- health.py is standalone) -----------------------

def _get_wiki_sections() -> list[str]:
    sections = set(WIKI_SECTIONS_DEFAULT)
    if WIKI_DIR.exists():
        for d in WIKI_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name != "meta":
                if any(d.rglob("*.md")):
                    sections.add(d.name)
    return sorted(sections)


def _parse_frontmatter_yaml(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    try:
        result = yaml.safe_load(fm_text)
        return result if isinstance(result, dict) else {}
    except yaml.YAMLError:
        return {}


def _extract_wikilinks(content: str) -> list[str]:
    """Extract all slug targets from [[type:slug]] and [[slug]] links."""
    slugs = []
    # Typed
    for m in re.finditer(r"\[\[[a-z_]+:([^\]]+)\]\]", content):
        slugs.append(Path(m.group(1).strip()).stem)
    # Bare
    for m in re.finditer(r"\[\[([^\]:]+)\]\]", content):
        slugs.append(m.group(1).strip())
    return slugs


def _load_sources() -> dict:
    if SOURCES_FILE.exists():
        return json.loads(SOURCES_FILE.read_text())
    return {"processed": {}}


# -- Checks --------------------------------------------------------------------

def check_stale(sources: dict) -> list[dict]:
    """Articles where last_compiled < newest contributing source processed_at."""
    article_sources: dict[str, list[dict]] = {}
    for source_rel, entry in sources.get("processed", {}).items():
        for article_path in entry.get("articles", []):
            article_sources.setdefault(article_path, []).append({
                "source": source_rel,
                "processed_at": entry.get("processed_at", ""),
            })

    stale = []
    for article_rel, contributing in article_sources.items():
        article_path = WIKI_DIR / article_rel
        if not article_path.exists():
            continue
        try:
            fm = _parse_frontmatter_yaml(article_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        last_compiled = fm.get("last_compiled", "")
        if not last_compiled:
            continue
        newest = max(contributing, key=lambda x: x["processed_at"])
        newest_date = newest["processed_at"][:10]
        if newest_date > str(last_compiled):
            try:
                stale_days = (
                    datetime.strptime(newest_date, "%Y-%m-%d")
                    - datetime.strptime(str(last_compiled), "%Y-%m-%d")
                ).days
            except ValueError:
                stale_days = -1
            stale.append({
                "article": article_rel,
                "last_compiled": str(last_compiled),
                "newest_source_date": newest_date,
                "newest_source": newest["source"],
                "stale_days": stale_days,
            })
    return sorted(stale, key=lambda x: x["stale_days"], reverse=True)


def check_orphan_briefs(sources: dict) -> list[str]:
    """Raw files processed but produced no articles."""
    return [
        rel for rel, entry in sources.get("processed", {}).items()
        if not entry.get("articles")
    ]


def check_orphan_articles(sources: dict) -> list[str]:
    """Wiki articles with zero inbound links."""
    referenced: set[str] = set()
    all_articles: list[Path] = []

    for section in _get_wiki_sections():
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            continue
        for f in section_dir.rglob("*.md"):
            all_articles.append(f)
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            fm = _parse_frontmatter_yaml(text)
            links_fm = fm.get("links", [])
            if links_fm and isinstance(links_fm, list):
                for link in links_fm:
                    if isinstance(link, dict):
                        t = link.get("target", "")
                        if t:
                            referenced.add(Path(t).stem)
            else:
                for slug in _extract_wikilinks(text):
                    referenced.add(slug)

    orphans = []
    for f in all_articles:
        if f.stem not in referenced:
            rel = str(f.relative_to(WIKI_DIR))
            orphans.append(rel)
    return sorted(orphans)


def check_dead_links() -> list[dict]:
    """Cross-references pointing to articles that don't exist."""
    existing_slugs: set[str] = set()
    for section in _get_wiki_sections():
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            continue
        for f in section_dir.rglob("*.md"):
            existing_slugs.add(f.stem)

    dead = []
    for section in _get_wiki_sections():
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            continue
        for f in section_dir.rglob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            for slug in _extract_wikilinks(text):
                if slug not in existing_slugs:
                    source = str(f.relative_to(WIKI_DIR))
                    entry = {"source": source, "target": slug}
                    if entry not in dead:
                        dead.append(entry)
    return dead


def check_link_type_distribution() -> dict[str, int]:
    """Count each typed link type across all wiki articles."""
    counts = {t: 0 for t in LINK_TYPES}
    counts["bare"] = 0

    for section in _get_wiki_sections():
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            continue
        for f in section_dir.rglob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            for m in re.finditer(r"\[\[([a-z_]+):([^\]]+)\]\]", text):
                lt = m.group(1)
                if lt in counts:
                    counts[lt] += 1
                else:
                    counts["bare"] += 1
            for _ in re.finditer(r"\[\[([^\]:]+)\]\]", text):
                counts["bare"] += 1

    return counts


def check_last_compile(sources: dict) -> str:
    """Max processed_at across all sources."""
    dates = [
        entry.get("processed_at", "")
        for entry in sources.get("processed", {}).values()
    ]
    if not dates:
        return "unknown"
    return max(d for d in dates if d)[:19].replace("T", " ")


# -- Report formatting ---------------------------------------------------------

def format_report(
    stale: list[dict],
    orphan_briefs: list[str],
    orphan_articles: list[str],
    dead_links: list[dict],
    link_types: dict[str, int],
    last_compile: str,
    total_articles: int,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"WIKI HEALTH REPORT -- {now}",
        "=" * 50,
        "",
        f"Stale articles:      {len(stale)}",
    ]
    for s in stale[:5]:
        lines.append(
            f"  {s['article']} "
            f"(compiled {s['last_compiled']}, source {s['newest_source_date']}, {s['stale_days']}d)"
        )
    if len(stale) > 5:
        lines.append(f"  ... and {len(stale) - 5} more")

    lines += [
        f"Orphan briefs:       {len(orphan_briefs)}",
        f"Orphan articles:     {len(orphan_articles)}",
    ]
    for o in orphan_articles[:5]:
        lines.append(f"  {o}")
    if len(orphan_articles) > 5:
        lines.append(f"  ... and {len(orphan_articles) - 5} more")

    lines.append(f"Dead links:          {len(dead_links)}")
    for d in dead_links[:5]:
        lines.append(f"  {d['source']} -> [[{d['target']}]] [missing]")
    if len(dead_links) > 5:
        lines.append(f"  ... and {len(dead_links) - 5} more")

    lines += [
        f"Total articles:      {total_articles}",
        f"Last compile:        {last_compile}",
        "",
        "Link types:          "
        + "  ".join(f"{k}={v}" for k, v in link_types.items() if v > 0),
    ]

    lines.append("")
    issues = []
    if stale:
        issues.append(f"{len(stale)} STALE")
    if dead_links:
        issues.append(f"{len(dead_links)} DEAD LINK{'S' if len(dead_links) > 1 else ''}")
    if orphan_articles:
        issues.append(f"{len(orphan_articles)} ORPHAN ARTICLE{'S' if len(orphan_articles) > 1 else ''}")

    if issues:
        lines.append(f"Status: {', '.join(issues)} -- recompile recommended")
    else:
        lines.append("Status: OK")

    return "\n".join(lines)


# -- Write meta files ----------------------------------------------------------

def write_meta_stale(stale: list[dict]) -> None:
    meta_dir = WIKI_DIR / "meta"
    meta_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Stale Articles", f"_Last updated: {now}_", ""]
    if not stale:
        lines.append("*(none -- all compiled articles are current)*")
    else:
        lines.append(f"{len(stale)} article(s) with newer source material:\n")
        for s in stale:
            lines.append(
                f"- **{s['article']}** -- compiled {s['last_compiled']}, "
                f"source {s['newest_source_date']} ({s['stale_days']}d) "
                f"via `{s['newest_source']}`"
            )
    (meta_dir / "stale.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_meta_orphans(orphan_articles: list[str]) -> None:
    meta_dir = WIKI_DIR / "meta"
    meta_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Orphaned Articles", f"_Last updated: {now}_", ""]
    if not orphan_articles:
        lines.append("*(none)*")
    else:
        for o in orphan_articles:
            lines.append(f"- [{o}]({o})")
    (meta_dir / "orphans.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- Count all wiki articles ---------------------------------------------------

def count_articles() -> int:
    total = 0
    for section in _get_wiki_sections():
        section_dir = WIKI_DIR / section
        if section_dir.exists():
            total += len(list(section_dir.rglob("*.md")))
    return total


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wiki health report")
    parser.add_argument("--write", action="store_true", help="Write results to wiki/meta/*.md")
    args = parser.parse_args()

    sources = _load_sources()

    stale = check_stale(sources)
    orphan_briefs = check_orphan_briefs(sources)
    orphan_articles = check_orphan_articles(sources)
    dead_links = check_dead_links()
    link_types = check_link_type_distribution()
    last_compile = check_last_compile(sources)
    total_articles = count_articles()

    report = format_report(
        stale=stale,
        orphan_briefs=orphan_briefs,
        orphan_articles=orphan_articles,
        dead_links=dead_links,
        link_types=link_types,
        last_compile=last_compile,
        total_articles=total_articles,
    )

    print(report)

    if args.write:
        write_meta_stale(stale)
        write_meta_orphans(orphan_articles)
        print(f"\nWrote wiki/meta/stale.md and wiki/meta/orphans.md")


if __name__ == "__main__":
    main()
