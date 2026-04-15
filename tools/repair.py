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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import (
    WIKI_DIR, INDEX_FILE,
    LINK_TYPES, INVERSE_LINK_TYPE,
    _TYPED_LINK_RE, _BARE_LINK_RE,
    all_articles, count_articles,
    parse_frontmatter, split_frontmatter, write_article,
    extract_typed_links, load_sources,
    rebuild_index,
)

# Generic terms that appear as dead links but aren't articles -- they're tags/categories.
GENERIC_TERMS = {
    "agents", "ai", "research", "math", "training", "memory", "architecture",
    "economics", "tools", "hardware", "models", "events", "entities",
    "concepts", "security", "ethics", "finance",
    "meta", "patterns", "frameworks",
}

# Known merge pairs: (keep, absorb). Populate with project-specific duplicates.
MERGE_PAIRS: list[tuple[str, str]] = []

# Dead link aliases: map non-existent slugs to existing article slugs.
LINK_ALIASES: dict[str, str] = {}


# -- Operation 1: Reciprocal Backlinks ----------------------------------------

def inject_reciprocal_backlinks(dry_run: bool = False) -> dict:
    """For every link A->B, ensure B has a backlink to A."""
    articles = all_articles()
    stats = {"links_checked": 0, "backlinks_added": 0, "articles_modified": set()}

    # Collect outbound links
    outbound: dict[str, list[dict]] = {}
    for slug, path in articles.items():
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        links = fm.get("links", [])
        if isinstance(links, list):
            outbound[slug] = [l for l in links if isinstance(l, dict) and l.get("target")]
        else:
            outbound[slug] = extract_typed_links(text)

    # Find missing backlinks
    pending: dict[str, list[dict]] = {}
    for source_slug, links in outbound.items():
        for link in links:
            target_slug = link.get("target", "")
            link_type = link.get("type", "references")
            stats["links_checked"] += 1

            if target_slug not in articles:
                continue

            inverse_type = INVERSE_LINK_TYPE.get(link_type, "related")
            target_links = outbound.get(target_slug, [])
            if any(l.get("target") == source_slug for l in target_links):
                continue
            if any(p["target"] == source_slug for p in pending.get(target_slug, [])):
                continue

            pending.setdefault(target_slug, []).append({"target": source_slug, "type": inverse_type})

    # Apply
    for target_slug, new_links in pending.items():
        if target_slug not in articles:
            continue
        path = articles[target_slug]
        fm, body = split_frontmatter(path.read_text(encoding="utf-8"))

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
                write_article(path, fm, body)

    stats["articles_modified"] = len(stats["articles_modified"])
    return stats


# -- Operation 2: Prune Dead Generic Links ------------------------------------

def prune_dead_links(dry_run: bool = False) -> dict:
    """Remove wikilinks where target doesn't exist and is noise."""
    articles = all_articles()
    existing_slugs = set(articles.keys())
    stats = {"links_pruned": 0, "links_fixed": 0, "articles_modified": set()}

    # Count dead references
    dead_ref_counts: dict[str, int] = {}
    for slug, path in articles.items():
        text = path.read_text(encoding="utf-8")
        seen_in_file: set[str] = set()
        for m in _TYPED_LINK_RE.finditer(text):
            t = Path(m.group(2).strip()).stem
            if t not in existing_slugs and t not in seen_in_file:
                dead_ref_counts[t] = dead_ref_counts.get(t, 0) + 1
                seen_in_file.add(t)
        for m in _BARE_LINK_RE.finditer(text):
            t = m.group(1).strip()
            if t not in existing_slugs and t not in seen_in_file:
                dead_ref_counts[t] = dead_ref_counts.get(t, 0) + 1
                seen_in_file.add(t)

    # Classify: fix (alias/case) or prune (generic/singleton)
    fix_targets: dict[str, str] = {}
    prune_targets: set[str] = set()

    for target in dead_ref_counts:
        if target in LINK_ALIASES and LINK_ALIASES[target] in existing_slugs:
            fix_targets[target] = LINK_ALIASES[target]

    for target, count in dead_ref_counts.items():
        if target in fix_targets:
            continue
        # Case-insensitive match
        case_match = next((s for s in existing_slugs if s.lower() == target.lower()), None)
        if case_match and case_match != target:
            fix_targets[target] = case_match
        elif target.lower() in GENERIC_TERMS:
            prune_targets.add(target)
        elif count == 1:
            prune_targets.add(target)

    # Apply fixes and prunes
    for slug, path in articles.items():
        fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
        original_body = body

        def _replace_typed(m):
            lt, tgt = m.group(1), m.group(2).strip()
            ts = Path(tgt).stem
            if ts in fix_targets:
                stats["links_fixed"] += 1
                return f"[[{lt}:{fix_targets[ts]}]]"
            if ts in prune_targets:
                stats["links_pruned"] += 1
                return ts.replace("-", " ")
            return m.group(0)

        body = _TYPED_LINK_RE.sub(_replace_typed, body)

        def _replace_bare(m):
            target = m.group(1).strip()
            if target in fix_targets:
                stats["links_fixed"] += 1
                return f"[[{fix_targets[target]}]]"
            if target in prune_targets:
                stats["links_pruned"] += 1
                return target.replace("-", " ")
            return m.group(0)

        body = _BARE_LINK_RE.sub(_replace_bare, body)

        # Clean frontmatter links too
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
                write_article(path, fm, body)

    stats["articles_modified"] = len(stats["articles_modified"])
    return stats


# -- Operation 3: Merge Duplicates --------------------------------------------

def merge_duplicates(dry_run: bool = False) -> dict:
    """Merge near-duplicate articles. Keeps the larger, absorbs the smaller."""
    articles = all_articles()
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
            keep_fm, keep_body = split_frontmatter(keep_path.read_text(encoding="utf-8"))
            absorb_fm = parse_frontmatter(absorb_path.read_text(encoding="utf-8"))

            # Merge sources and tags
            for s in absorb_fm.get("sources", []):
                if s not in keep_fm.get("sources", []):
                    keep_fm.setdefault("sources", []).append(s)
            keep_fm["tags"] = sorted(set(keep_fm.get("tags", [])) | set(absorb_fm.get("tags", [])))
            write_article(keep_path, keep_fm, keep_body)

            absorb_path.unlink()

            # Update references
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
    """Ensure frontmatter links block matches content wikilinks."""
    articles = all_articles()
    stats = {"articles_synced": 0, "links_added": 0}

    for slug, path in articles.items():
        fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
        content_links = extract_typed_links(body)
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
                write_article(path, fm, body)

    return stats


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
    from health import (check_orphan_articles, check_dead_links, check_link_type_distribution)
    sources = load_sources()
    pre_orphans = len(check_orphan_articles(sources))
    pre_dead = len(check_dead_links())
    pre_articles = count_articles()
    pre_links = check_link_type_distribution()

    if run_all or args.merge:
        print("\n1. Merging duplicates...")
        s = merge_duplicates(args.dry_run)
        print(f"   Merged: {s['merged']}  Redirects: {s['redirects_updated']}")

    if run_all:
        print("\n2. Syncing frontmatter links...")
        s = sync_frontmatter_links(args.dry_run)
        print(f"   Synced: {s['articles_synced']}  Links added: {s['links_added']}")

    if run_all or args.prune:
        print("\n3. Pruning dead links...")
        s = prune_dead_links(args.dry_run)
        print(f"   Pruned: {s['links_pruned']}  Modified: {s['articles_modified']}")

    if run_all or args.backlinks:
        print("\n4. Injecting reciprocal backlinks...")
        s = inject_reciprocal_backlinks(args.dry_run)
        print(f"   Checked: {s['links_checked']}  Added: {s['backlinks_added']}  Modified: {s['articles_modified']}")

    if not args.dry_run:
        print("\n5. Rebuilding index...")
        total = rebuild_index()
        print(f"   Index: {total} articles")

        # Post-repair snapshot
        sources = load_sources()
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
        typed_pre = sum(v for k, v in pre_links.items() if k != 'bare')
        typed_post = sum(v for k, v in post_links.items() if k != 'bare')
        print(f"  Typed links: {typed_pre} -> {typed_post}")


if __name__ == "__main__":
    main()
