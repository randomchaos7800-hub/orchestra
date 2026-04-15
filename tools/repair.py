#!/usr/bin/env python3
"""
Wiki Graph Repair -- tools/repair.py

One-shot repair tool that fixes structural problems in the wiki graph.
Safe to run repeatedly -- all operations are idempotent.

Operations:
  1. Inject reciprocal backlinks (A links to B -> B gets backlink to A)
  2. Prune dead wikilinks to generic/tag terms that aren't articles
  3. Merge near-duplicate articles (keeps the larger, redirects the smaller)
  4. Normalize frontmatter link blocks from content wikilinks
  5. Rebuild _index.md

Usage:
  python3 tools/repair.py                # full repair
  python3 tools/repair.py --dry-run      # show what would change, no writes
  python3 tools/repair.py --backlinks    # only inject reciprocal backlinks
  python3 tools/repair.py --prune        # only prune dead generic links
  python3 tools/repair.py --merge        # only merge duplicates
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
INDEX_FILE = WIKI_DIR / "_index.md"

LINK_TYPES = ["references", "depends_on", "extends", "contradicts", "related"]
INVERSE_TYPES = {
    "depends_on": "depended_on_by",
    "extends": "extended_by",
    "contradicts": "contradicts",      # symmetric
    "references": "referenced_by",
    "related": "related",              # symmetric
}
# For backlink injection, use canonical type names
INVERSE_CANONICAL = {
    "depends_on": "referenced_by",
    "extends": "referenced_by",
    "contradicts": "related",
    "references": "referenced_by",
    "related": "related",
}

WIKI_SECTIONS = ["concepts", "entities", "events", "research", "tools"]

# Generic terms that appear as dead links but aren't articles -- they're tags/categories.
# Add project-specific generic terms as needed.
GENERIC_TERMS = {
    "agents", "ai", "research", "math", "training", "memory", "architecture",
    "economics", "tools", "hardware", "models", "events", "entities",
    "concepts", "security", "ethics", "finance",
    "meta", "patterns", "frameworks",
}

# Known merge pairs: (keep, absorb).
# Populate with project-specific duplicates as they are discovered.
MERGE_PAIRS: list[tuple[str, str]] = []

# Dead link aliases: map non-existent slugs to existing article slugs.
# Populate with project-specific aliases as needed.
LINK_ALIASES: dict[str, str] = {}


# -- Helpers -------------------------------------------------------------------

def _get_sections() -> list[str]:
    sections = set(WIKI_SECTIONS)
    if WIKI_DIR.exists():
        for d in WIKI_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name != "meta":
                if any(d.rglob("*.md")):
                    sections.add(d.name)
    return sorted(sections)


def _all_articles() -> dict[str, Path]:
    """Return {slug: Path} for every wiki article."""
    articles = {}
    for section in _get_sections():
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            continue
        for f in section_dir.rglob("*.md"):
            articles[f.stem] = f
    return articles


def _parse_fm(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    try:
        result = yaml.safe_load(text[3:end].strip())
        return result if isinstance(result, dict) else {}
    except yaml.YAMLError:
        return {}


def _split_fm_body(text: str) -> tuple[dict, str]:
    """Split text into (frontmatter_dict, body_str)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    try:
        fm = yaml.safe_load(text[3:end].strip()) or {}
    except yaml.YAMLError:
        fm = {}
    body = text[end + 3:].lstrip("\n")
    return fm, body


def _write_article(path: Path, fm: dict, body: str) -> None:
    """Write article with YAML frontmatter."""
    class _Dumper(yaml.Dumper):
        pass

    def _list_representer(dumper, data):
        if data and isinstance(data[0], dict):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    _Dumper.add_representer(list, _list_representer)
    fm_out = yaml.dump(fm, Dumper=_Dumper, default_flow_style=False, allow_unicode=True).strip()
    path.write_text(f"---\n{fm_out}\n---\n\n{body}", encoding="utf-8")


def _extract_typed_links(content: str) -> list[dict]:
    """Extract all [[type:slug]] and [[slug]] wikilinks from content body."""
    seen = {}
    for m in re.finditer(r"\[\[(" + "|".join(LINK_TYPES) + r"):([^\]]+)\]\]", content):
        link_type, target = m.group(1), m.group(2).strip()
        slug = Path(target).stem if "/" not in target else target.rsplit("/", 1)[-1].replace(".md", "")
        if slug not in seen:
            seen[slug] = {"target": slug, "type": link_type}
    for m in re.finditer(r"\[\[([^\]:]+)\]\]", content):
        target = m.group(1).strip()
        slug = Path(target).stem
        if slug not in seen:
            seen[slug] = {"target": slug, "type": "references"}
    return list(seen.values())


def _fm_link_set(fm: dict) -> set[tuple[str, str]]:
    """Return set of (target, type) from frontmatter links."""
    links = fm.get("links", [])
    if not isinstance(links, list):
        return set()
    return {(l.get("target", ""), l.get("type", "")) for l in links if isinstance(l, dict)}


# -- Operation 1: Reciprocal Backlinks ----------------------------------------

def inject_reciprocal_backlinks(dry_run: bool = False) -> dict:
    """For every link A->B, ensure B has a backlink to A."""
    articles = _all_articles()
    stats = {"links_checked": 0, "backlinks_added": 0, "articles_modified": set()}

    outbound: dict[str, list[dict]] = {}
    for slug, path in articles.items():
        text = path.read_text(encoding="utf-8")
        fm = _parse_fm(text)
        links = fm.get("links", [])
        if isinstance(links, list):
            outbound[slug] = [l for l in links if isinstance(l, dict) and l.get("target")]
        else:
            outbound[slug] = _extract_typed_links(text)

    pending_additions: dict[str, list[dict]] = {}

    for source_slug, links in outbound.items():
        for link in links:
            target_slug = link.get("target", "")
            link_type = link.get("type", "references")
            stats["links_checked"] += 1

            if target_slug not in articles:
                continue

            inverse_type = INVERSE_CANONICAL.get(link_type, "related")

            target_links = outbound.get(target_slug, [])
            already_linked = any(
                l.get("target") == source_slug for l in target_links
            )
            if already_linked:
                continue

            pending = pending_additions.get(target_slug, [])
            already_pending = any(p["target"] == source_slug for p in pending)
            if already_pending:
                continue

            pending_additions.setdefault(target_slug, []).append({
                "target": source_slug,
                "type": inverse_type,
            })

    for target_slug, new_links in pending_additions.items():
        if target_slug not in articles:
            continue
        path = articles[target_slug]
        text = path.read_text(encoding="utf-8")
        fm, body = _split_fm_body(text)

        existing_links = fm.get("links", [])
        if not isinstance(existing_links, list):
            existing_links = []

        existing_targets = {l.get("target") for l in existing_links if isinstance(l, dict)}

        added = 0
        for nl in new_links:
            if nl["target"] not in existing_targets:
                existing_links.append(nl)
                existing_targets.add(nl["target"])
                added += 1

        if added > 0:
            fm["links"] = existing_links
            stats["backlinks_added"] += added
            stats["articles_modified"].add(target_slug)
            if not dry_run:
                _write_article(path, fm, body)

    stats["articles_modified"] = len(stats["articles_modified"])
    return stats


# -- Operation 2: Prune Dead Generic Links ------------------------------------

def _find_case_match(slug: str, existing: set[str]) -> str | None:
    """Find case-insensitive match for a slug."""
    slug_lower = slug.lower()
    for s in existing:
        if s.lower() == slug_lower:
            return s
    return None


def prune_dead_links(dry_run: bool = False) -> dict:
    """Remove [[slug]] wikilinks from content where slug doesn't exist and is noise."""
    articles = _all_articles()
    existing_slugs = set(articles.keys())
    stats = {"links_pruned": 0, "links_fixed": 0, "articles_modified": set()}

    dead_ref_counts: dict[str, int] = {}
    for slug, path in articles.items():
        text = path.read_text(encoding="utf-8")
        seen_in_file = set()
        for m in re.finditer(r"\[\[(?:" + "|".join(LINK_TYPES) + r"):([^\]]+)\]\]", text):
            t = Path(m.group(1).strip()).stem
            if t not in existing_slugs and t not in seen_in_file:
                dead_ref_counts[t] = dead_ref_counts.get(t, 0) + 1
                seen_in_file.add(t)
        for m in re.finditer(r"\[\[([^\]:]+)\]\]", text):
            t = m.group(1).strip()
            if t not in existing_slugs and t not in seen_in_file:
                dead_ref_counts[t] = dead_ref_counts.get(t, 0) + 1
                seen_in_file.add(t)

    prune_targets = set()
    fix_targets = {}

    for target in dead_ref_counts:
        if target in LINK_ALIASES and LINK_ALIASES[target] in existing_slugs:
            fix_targets[target] = LINK_ALIASES[target]

    for target, count in dead_ref_counts.items():
        if target in fix_targets:
            continue
        case_match = _find_case_match(target, existing_slugs)
        if case_match and case_match != target:
            fix_targets[target] = case_match
        elif target.lower() in GENERIC_TERMS:
            prune_targets.add(target)
        elif count == 1:
            prune_targets.add(target)

    for slug, path in articles.items():
        text = path.read_text(encoding="utf-8")
        fm, body = _split_fm_body(text)
        original_body = body

        def _replace_typed(m):
            link_type = m.group(1)
            target = m.group(2).strip()
            target_slug = Path(target).stem
            if target_slug in fix_targets:
                stats["links_fixed"] += 1
                return f"[[{link_type}:{fix_targets[target_slug]}]]"
            if target_slug in prune_targets:
                stats["links_pruned"] += 1
                return target_slug.replace("-", " ")
            return m.group(0)

        body = re.sub(
            r"\[\[(" + "|".join(LINK_TYPES) + r"):([^\]]+)\]\]",
            _replace_typed, body
        )

        def _replace_bare(m):
            target = m.group(1).strip()
            if target in fix_targets:
                stats["links_fixed"] += 1
                return f"[[{fix_targets[target]}]]"
            if target in prune_targets:
                stats["links_pruned"] += 1
                return target.replace("-", " ")
            return m.group(0)

        body = re.sub(r"\[\[([^\]:]+)\]\]", _replace_bare, body)

        fm_links = fm.get("links", [])
        if isinstance(fm_links, list):
            cleaned = []
            for l in fm_links:
                if isinstance(l, dict):
                    t = l.get("target", "")
                    if t in fix_targets:
                        l["target"] = fix_targets[t]
                        stats["links_fixed"] += 1
                    elif t in prune_targets:
                        stats["links_pruned"] += 1
                        continue
                cleaned.append(l)
            if len(cleaned) != len(fm_links):
                fm["links"] = cleaned

        if body != original_body or len(fm.get("links", [])) != len(fm_links):
            stats["articles_modified"].add(slug)
            if not dry_run:
                _write_article(path, fm, body)

    stats["articles_modified"] = len(stats["articles_modified"])
    return stats


# -- Operation 3: Merge Duplicates --------------------------------------------

def merge_duplicates(dry_run: bool = False) -> dict:
    """Merge near-duplicate articles. Keeps the larger one, absorbs the smaller."""
    articles = _all_articles()
    stats = {"merged": 0, "redirects_updated": 0}

    for keep_rel, absorb_rel in MERGE_PAIRS:
        keep_path = WIKI_DIR / keep_rel
        absorb_path = WIKI_DIR / absorb_rel

        if not keep_path.exists() or not absorb_path.exists():
            continue

        keep_slug = keep_path.stem
        absorb_slug = absorb_path.stem

        print(f"  Merge: {absorb_rel} -> {keep_rel}")

        if not dry_run:
            keep_text = keep_path.read_text(encoding="utf-8")
            absorb_text = absorb_path.read_text(encoding="utf-8")
            absorb_fm = _parse_fm(absorb_text)

            keep_fm, keep_body = _split_fm_body(keep_text)
            absorb_sources = absorb_fm.get("sources", [])
            keep_sources = keep_fm.get("sources", [])
            if isinstance(absorb_sources, list) and isinstance(keep_sources, list):
                for s in absorb_sources:
                    if s not in keep_sources:
                        keep_sources.append(s)
                keep_fm["sources"] = keep_sources

            keep_tags = set(keep_fm.get("tags", []))
            absorb_tags = set(absorb_fm.get("tags", []))
            keep_fm["tags"] = sorted(keep_tags | absorb_tags)

            _write_article(keep_path, keep_fm, keep_body)

            absorb_path.unlink()

            for slug, path in articles.items():
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8")
                if absorb_slug in text:
                    new_text = text.replace(absorb_slug, keep_slug)
                    if new_text != text:
                        path.write_text(new_text, encoding="utf-8")
                        stats["redirects_updated"] += 1

        stats["merged"] += 1

    return stats


# -- Operation 4: Sync Frontmatter Links from Content -------------------------

def sync_frontmatter_links(dry_run: bool = False) -> dict:
    """Ensure frontmatter links: block matches content wikilinks."""
    articles = _all_articles()
    stats = {"articles_synced": 0, "links_added": 0}

    for slug, path in articles.items():
        text = path.read_text(encoding="utf-8")
        fm, body = _split_fm_body(text)

        content_links = _extract_typed_links(body)
        if not content_links:
            continue

        fm_links = fm.get("links", [])
        if not isinstance(fm_links, list):
            fm_links = []

        existing_targets = {l.get("target") for l in fm_links if isinstance(l, dict)}

        added = 0
        for cl in content_links:
            if cl["target"] not in existing_targets:
                fm_links.append(cl)
                existing_targets.add(cl["target"])
                added += 1

        if added > 0:
            fm["links"] = fm_links
            stats["articles_synced"] += 1
            stats["links_added"] += added
            if not dry_run:
                _write_article(path, fm, body)

    return stats


# -- Index Rebuild -------------------------------------------------------------

def rebuild_index() -> int:
    """Rewrite _index.md from current wiki state. Returns article count."""
    articles = _all_articles()
    total = len(articles)
    now = datetime.now().strftime("%Y-%m-%d")

    summaries = {}
    if INDEX_FILE.exists():
        for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
            m = re.match(
                r"\*\*\[\[([^\]]+)\]\]\*\* -- ([^.]+(?:\.[^T][^a][^g][^s])*?)(?:\s+Tags:|\s+Updated:|$)",
                line,
            )
            if m:
                summaries[m.group(1)] = m.group(2).strip().rstrip(".")

    lines = [
        f"# Wiki Index",
        f"_Last updated: {now} | {total} article{'s' if total != 1 else ''}_",
        "", "---", "",
    ]

    for section in _get_sections():
        section_articles = {s: p for s, p in articles.items()
                          if str(p.relative_to(WIKI_DIR)).startswith(section + "/")}
        lines.append(f"## {section}/ ({len(section_articles)})")
        lines.append("")
        if not section_articles:
            lines.append("*(empty)*")
            lines.append("")
            continue
        for slug in sorted(section_articles):
            path = section_articles[slug]
            try:
                fm = _parse_fm(path.read_text(encoding="utf-8"))
            except Exception:
                fm = {}
            summary = summaries.get(slug) or fm.get("title", slug)
            tags = fm.get("tags", [])
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            updated = fm.get("updated", "")
            entry = f"**[[{slug}]]** -- {summary}"
            if tags_str:
                entry += f" Tags: {tags_str}."
            if updated:
                entry += f" Updated: {updated}."
            lines.append(entry)
        lines.append("")

    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")
    return total


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wiki graph repair")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backlinks", action="store_true", help="Only inject reciprocal backlinks")
    parser.add_argument("--prune", action="store_true", help="Only prune dead generic links")
    parser.add_argument("--merge", action="store_true", help="Only merge duplicates")
    args = parser.parse_args()

    run_all = not (args.backlinks or args.prune or args.merge)
    dry_label = " [DRY RUN]" if args.dry_run else ""
    print(f"Wiki repair{dry_label}")
    print("=" * 50)

    # Pre-repair snapshot
    from health import (check_orphan_articles, check_dead_links, check_link_type_distribution,
                        count_articles, _load_sources)
    sources = _load_sources()
    pre_orphans = len(check_orphan_articles(sources))
    pre_dead = len(check_dead_links())
    pre_articles = count_articles()
    pre_links = check_link_type_distribution()

    # Step 1: Merge duplicates first (changes the article set)
    if run_all or args.merge:
        print("\n1. Merging duplicates...")
        merge_stats = merge_duplicates(args.dry_run)
        print(f"   Merged: {merge_stats['merged']}  Redirects updated: {merge_stats['redirects_updated']}")

    # Step 2: Sync frontmatter links from content
    if run_all:
        print("\n2. Syncing frontmatter links from content...")
        sync_stats = sync_frontmatter_links(args.dry_run)
        print(f"   Articles synced: {sync_stats['articles_synced']}  Links added: {sync_stats['links_added']}")

    # Step 3: Prune dead generic links
    if run_all or args.prune:
        print("\n3. Pruning dead generic links...")
        prune_stats = prune_dead_links(args.dry_run)
        print(f"   Links pruned: {prune_stats['links_pruned']}  Articles modified: {prune_stats['articles_modified']}")

    # Step 4: Inject reciprocal backlinks
    if run_all or args.backlinks:
        print("\n4. Injecting reciprocal backlinks...")
        bl_stats = inject_reciprocal_backlinks(args.dry_run)
        print(f"   Links checked: {bl_stats['links_checked']}  Backlinks added: {bl_stats['backlinks_added']}  Articles modified: {bl_stats['articles_modified']}")

    # Step 5: Rebuild index
    if not args.dry_run:
        print("\n5. Rebuilding index...")
        total = rebuild_index()
        print(f"   Index rebuilt: {total} articles")

    # Post-repair snapshot
    if not args.dry_run:
        sources = _load_sources()
        post_orphans = len(check_orphan_articles(sources))
        post_dead = len(check_dead_links())
        post_articles = count_articles()
        post_links = check_link_type_distribution()

        print("\n" + "=" * 50)
        print("BEFORE -> AFTER")
        print(f"  Articles:    {pre_articles} -> {post_articles}")
        print(f"  Orphans:     {pre_orphans} -> {post_orphans}  ({pre_orphans - post_orphans:+d})")
        print(f"  Dead links:  {pre_dead} -> {post_dead}  ({pre_dead - post_dead:+d})")
        print(f"  Bare links:  {pre_links.get('bare', 0)} -> {post_links.get('bare', 0)}")
        print(f"  Typed links: {sum(v for k,v in pre_links.items() if k != 'bare')} -> {sum(v for k,v in post_links.items() if k != 'bare')}")


if __name__ == "__main__":
    main()
