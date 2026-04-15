#!/usr/bin/env python3
"""
Wiki Search -- tools/search.py

Full-text search over wiki/ articles.

Usage:
  python3 tools/search.py "theory of mind"
  python3 tools/search.py --index-only "emergence"   # search only _index.md (fast)
  python3 tools/search.py --tag agents                # filter by tag
  python3 tools/search.py --section concepts "memory" # search within a section
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import WIKI_DIR, INDEX_FILE, parse_frontmatter

CONTEXT_LINES = 2


def search_index(query: str) -> list[dict]:
    """Search only _index.md (fast)."""
    if not INDEX_FILE.exists():
        return []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return [
        {"file": "_index.md", "line": line.strip(), "context": []}
        for line in INDEX_FILE.read_text(encoding="utf-8").splitlines()
        if pattern.search(line)
    ]


def search_wiki(query: str, section: str = None, tag: str = None) -> list[dict]:
    """Full-text search with optional section and tag filtering."""
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results = []

    if section:
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            print(f"Section '{section}' not found.", file=sys.stderr)
            return []
        search_paths = list(section_dir.rglob("*.md"))
    else:
        search_paths = [f for f in WIKI_DIR.rglob("*.md") if not f.name.startswith("_")]

    for md_file in sorted(search_paths):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        if tag:
            fm = parse_frontmatter(text)
            file_tags = fm.get("tags", [])
            if not isinstance(file_tags, list) or tag not in file_tags:
                continue

        lines = text.splitlines()
        file_results = []
        for i, line in enumerate(lines):
            if pattern.search(line):
                start = max(0, i - CONTEXT_LINES)
                end = min(len(lines), i + CONTEXT_LINES + 1)
                file_results.append({
                    "line_num": i + 1, "line": line.strip(),
                    "context": lines[start:end],
                })

        if file_results:
            results.append({
                "file": str(md_file.relative_to(WIKI_DIR)),
                "match_count": len(file_results),
                "matches": file_results[:3],
            })

    results.sort(key=lambda x: x["match_count"], reverse=True)
    return results


def print_results(results: list[dict], query: str, index_only: bool = False) -> None:
    if not results:
        print(f"No results for '{query}'")
        return

    total = sum(r.get("match_count", 1) for r in results)
    print(f"\n{total} match(es) in {len(results)} file(s) for '{query}'\n")
    print("-" * 60)

    for r in results:
        print(f"\n  {r['file']}")
        if "match_count" in r:
            print(f"   {r['match_count']} match(es)")
        if index_only:
            print(f"   {r['line']}")
        else:
            for m in r.get("matches", []):
                print(f"\n   Line {m['line_num']}:")
                for ctx_line in m["context"]:
                    print(f"   {ctx_line}")

    print("\n" + "-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Search the wiki")
    parser.add_argument("query", help="Search query (case-insensitive)")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--section", type=str)
    args = parser.parse_args()

    if args.index_only:
        results = search_index(args.query)
        print_results(results, args.query, index_only=True)
    else:
        results = search_wiki(args.query, section=args.section, tag=args.tag)
        print_results(results, args.query)


if __name__ == "__main__":
    main()
