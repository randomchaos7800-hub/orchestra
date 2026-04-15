#!/usr/bin/env python3
"""
Wiki Health Report -- tools/health.py

Read-only audit of the knowledge base. Never modifies files (except wiki/meta/ reports).

Checks:
  - Stale articles (source newer than compilation)
  - Orphan briefs (raw files producing no articles)
  - Orphan articles (zero inbound links)
  - Dead/broken links (references to non-existent articles)
  - Unwritten concepts (frequently mentioned but no article)
  - Link type distribution

Usage:
  python3 tools/health.py                   # report to stdout
  python3 tools/health.py --write           # stdout + write wiki/meta/*.md
  python3 tools/health.py --stale-days 60   # custom staleness threshold
  python3 tools/health.py --suggest         # LLM-assisted gap analysis
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import (
    WIKI_DIR, LINK_TYPES,
    _TYPED_LINK_RE, _BARE_LINK_RE,
    get_wiki_sections, all_articles, count_articles,
    parse_frontmatter, extract_wikilink_slugs,
    load_sources, make_llm_client, INDEX_FILE,
    staleness_check as check_stale,
)


# -- Single-pass wiki scan -----------------------------------------------------

def scan_wiki(stale_days: int = 30) -> dict:
    """Single pass over all wiki articles. Returns all health metrics at once.

    Keys: articles, orphan_articles, dead_links, link_types, unwritten, stale_by_date
    """
    articles = all_articles()
    existing_slugs = set(articles.keys())
    cutoff = datetime.now() - timedelta(days=stale_days)

    referenced: set[str] = set()
    dead: dict[tuple[str, str], None] = {}
    link_counts = {t: 0 for t in LINK_TYPES}
    link_counts["bare"] = 0
    unwritten_counts: dict[str, int] = {}
    stale_by_date: list[tuple[str, str]] = []

    for slug, path in articles.items():
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        fm = parse_frontmatter(text)
        source = str(path.relative_to(WIKI_DIR))

        # Orphan detection: track what's referenced
        links_fm = fm.get("links", [])
        if links_fm and isinstance(links_fm, list):
            for link in links_fm:
                if isinstance(link, dict):
                    t = link.get("target", "")
                    if t:
                        referenced.add(Path(t).stem)
        else:
            for s in extract_wikilink_slugs(text):
                referenced.add(s)

        # Dead links + unwritten concepts
        for s in extract_wikilink_slugs(text):
            if s not in existing_slugs:
                dead[(source, s)] = None
                unwritten_counts[s] = unwritten_counts.get(s, 0) + 1

        # Link type distribution
        for m in _TYPED_LINK_RE.finditer(text):
            lt = m.group(1)
            if lt in link_counts:
                link_counts[lt] += 1
            else:
                link_counts["bare"] += 1
        for _ in _BARE_LINK_RE.finditer(text):
            link_counts["bare"] += 1

        # Stale by date
        updated = fm.get("updated", "")
        if updated:
            try:
                if datetime.strptime(str(updated)[:10], "%Y-%m-%d") < cutoff:
                    stale_by_date.append((source, str(updated)[:10]))
            except ValueError:
                pass
        else:
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime < cutoff:
                    stale_by_date.append((source, mtime.strftime("%Y-%m-%d")))
            except Exception:
                pass

    return {
        "articles": articles,
        "orphan_articles": sorted(
            str(p.relative_to(WIKI_DIR)) for s, p in articles.items() if s not in referenced
        ),
        "dead_links": [{"source": s, "target": t} for s, t in dead],
        "link_types": link_counts,
        "unwritten": sorted(unwritten_counts.items(), key=lambda x: -x[1]),
        "stale_by_date": sorted(stale_by_date, key=lambda x: x[1]),
    }


# -- Individual checks (thin wrappers for repair.py compatibility) -------------

def check_orphan_briefs(sources: dict) -> list[str]:
    """Raw files processed but produced no articles."""
    return [
        rel for rel, entry in sources.get("processed", {}).items()
        if not entry.get("articles")
    ]


def check_orphan_articles(sources: dict) -> list[str]:
    """Wiki articles with zero inbound links."""
    return scan_wiki()["orphan_articles"]


def check_dead_links() -> list[dict]:
    """Cross-references pointing to non-existent articles."""
    return scan_wiki()["dead_links"]


def check_link_type_distribution() -> dict[str, int]:
    """Count each typed link type."""
    return scan_wiki()["link_types"]


def check_last_compile(sources: dict) -> str:
    """Max processed_at across all sources."""
    dates = [entry.get("processed_at", "") for entry in sources.get("processed", {}).values()]
    if not dates:
        return "unknown"
    return max(d for d in dates if d)[:19].replace("T", " ")


# -- LLM suggestions ----------------------------------------------------------

def _llm_suggestions(articles: dict[str, Path], orphans: list[str], unwritten: list[tuple[str, int]]) -> str:
    """Ask LLM to suggest new articles and connections."""
    try:
        client, model, _ = make_llm_client()
    except SystemExit:
        return "*(LLM unavailable -- skipping suggestions)*"

    index_text = INDEX_FILE.read_text(encoding="utf-8")[:3000] if INDEX_FILE.exists() else "(empty)"
    orphan_list = "\n".join(f"- {o}" for o in orphans[:10]) or "(none)"
    unwritten_list = "\n".join(f"- [[{s}]] ({c} mentions)" for s, c in unwritten[:10]) or "(none)"

    prompt = f"""You are reviewing a knowledge base wiki for structural completeness.

Current wiki index:
{index_text}

Orphaned articles (no backlinks):
{orphan_list}

Mentioned but not written:
{unwritten_list}

Suggest:
1. 3-5 new articles worth creating (based on gaps and orphan context)
2. 3-5 connection pairs that should have backlinks but don't

Return as clean markdown, no preamble."""

    try:
        from lib.common import llm_call
        return llm_call(client, model, "You are a wiki structural analyst.", prompt, max_tokens=1500)
    except Exception as e:
        return f"*(LLM suggestion failed: {e})*"


# -- Report formatting ---------------------------------------------------------

def format_report(stale, orphan_briefs, orphan_articles, dead_links, link_types,
                  last_compile, total_articles, stale_by_date=None, unwritten=None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"WIKI HEALTH REPORT -- {now}",
        "=" * 50,
        "",
        f"Stale articles:      {len(stale)}",
    ]
    for s in stale[:5]:
        lines.append(f"  {s['article']} (compiled {s['last_compiled']}, source {s['newest_source_date']}, {s['stale_days']}d)")
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

    if stale_by_date:
        lines.append(f"Stale by date:       {len(stale_by_date)}")

    if unwritten:
        lines.append(f"Unwritten concepts:  {len(unwritten)}")
        for slug, count in unwritten[:5]:
            lines.append(f"  [[{slug}]] -- {count} mention(s)")

    lines += [
        f"Total articles:      {total_articles}",
        f"Last compile:        {last_compile}",
        "",
        "Link types:          "
        + "  ".join(f"{k}={v}" for k, v in link_types.items() if v > 0),
        "",
    ]

    issues = []
    if stale:
        issues.append(f"{len(stale)} STALE")
    if dead_links:
        issues.append(f"{len(dead_links)} DEAD LINK{'S' if len(dead_links) > 1 else ''}")
    if orphan_articles:
        issues.append(f"{len(orphan_articles)} ORPHAN{'S' if len(orphan_articles) > 1 else ''}")

    lines.append(f"Status: {', '.join(issues) + ' -- recompile recommended' if issues else 'OK'}")
    return "\n".join(lines)


# -- Write meta files ----------------------------------------------------------

def _write_meta(path: Path, title: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(f"# {title}\n_Last updated: {now}_\n\n{content}\n", encoding="utf-8")


def write_meta_files(stale, orphan_articles, stale_by_date=None, unwritten=None, suggestions=None):
    """Write all meta report files."""
    meta_dir = WIKI_DIR / "meta"

    # Stale
    if stale:
        content = "\n".join(
            f"- **{s['article']}** -- compiled {s['last_compiled']}, source {s['newest_source_date']} ({s['stale_days']}d)"
            for s in stale
        )
    else:
        content = "*(none -- all compiled articles are current)*"
    _write_meta(meta_dir / "stale.md", "Stale Articles", content)

    # Orphans
    orphan_content = "\n".join(f"- [{o}]({o})" for o in orphan_articles) or "*(none)*"
    _write_meta(meta_dir / "orphans.md", "Orphaned Articles", orphan_content)

    # Suggestions
    if suggestions:
        _write_meta(meta_dir / "suggestions.md", "Suggested Articles and Connections", suggestions)
    elif unwritten:
        unwritten_content = "\n".join(f"- `[[{s}]]` -- {c} mention(s)" for s, c in unwritten[:20]) or "*(none)*"
        _write_meta(meta_dir / "suggestions.md", "Mentioned But Not Written",
                    "Run `health.py --suggest` for LLM-assisted gap analysis.\n\n## Frequently Mentioned\n\n" + unwritten_content)


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wiki health report")
    parser.add_argument("--write", action="store_true", help="Write results to wiki/meta/*.md")
    parser.add_argument("--stale-days", type=int, default=30, help="Days before article considered stale by date")
    parser.add_argument("--suggest", action="store_true", help="LLM-assisted gap analysis")
    args = parser.parse_args()

    articles = all_articles()
    if not articles:
        print("Wiki is empty. Run compile.py first.")
        return

    # Single scan for all metrics
    scan = scan_wiki(stale_days=args.stale_days)
    sources = load_sources()
    stale = check_stale(sources)
    orphan_briefs = check_orphan_briefs(sources)
    orphan_articles = scan["orphan_articles"]
    dead_links = scan["dead_links"]
    link_types = scan["link_types"]
    last_compile = check_last_compile(sources)
    total = len(articles)
    stale_by_date = scan["stale_by_date"]
    unwritten = scan["unwritten"]

    report = format_report(
        stale=stale, orphan_briefs=orphan_briefs, orphan_articles=orphan_articles,
        dead_links=dead_links, link_types=link_types, last_compile=last_compile,
        total_articles=total, stale_by_date=stale_by_date, unwritten=unwritten,
    )
    print(report)

    suggestions = None
    if args.suggest:
        print("\nRunning LLM gap analysis...")
        suggestions = _llm_suggestions(articles, orphan_articles, unwritten)
        print(suggestions)

    if args.write or args.suggest:
        write_meta_files(stale, orphan_articles, stale_by_date, unwritten, suggestions)
        print(f"\nResults written to wiki/meta/")


if __name__ == "__main__":
    main()
