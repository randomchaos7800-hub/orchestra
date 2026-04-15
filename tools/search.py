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

KB_ROOT = Path(__file__).parent.parent
WIKI_DIR = KB_ROOT / "wiki"
INDEX_FILE = WIKI_DIR / "_index.md"

CONTEXT_LINES = 2  # lines of context to show around each match


def search_index(query: str) -> list[dict]:
    """Search only _index.md (fast, for quick lookups)."""
    if not INDEX_FILE.exists():
        return []

    results = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
        if pattern.search(line):
            results.append({"file": "_index.md", "line": line.strip(), "context": []})
    return results


def search_wiki(query: str, section: str = None, tag: str = None) -> list[dict]:
    """Full-text search over wiki/ articles. Returns ranked results with context."""
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results = []

    search_paths = []
    if section:
        section_dir = WIKI_DIR / section
        if section_dir.exists():
            search_paths = list(section_dir.rglob("*.md"))
        else:
            print(f"Section '{section}' not found.", file=sys.stderr)
            return []
    else:
        search_paths = [f for f in WIKI_DIR.rglob("*.md") if not f.name.startswith("_")]

    for md_file in sorted(search_paths):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Tag filter
        if tag:
            fm_match = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
            if fm_match:
                tags_line = re.search(r"tags:\s*\[([^\]]*)\]", fm_match.group(1))
                if tags_line:
                    file_tags = [t.strip() for t in tags_line.group(1).split(",")]
                    if tag not in file_tags:
                        continue
                else:
                    continue
            else:
                continue

        lines = text.splitlines()
        match_count = 0
        file_results = []

        for i, line in enumerate(lines):
            if pattern.search(line):
                match_count += 1
                start = max(0, i - CONTEXT_LINES)
                end = min(len(lines), i + CONTEXT_LINES + 1)
                context = lines[start:end]
                file_results.append({
                    "line_num": i + 1,
                    "line": line.strip(),
                    "context": context,
                })

        if file_results:
            rel = md_file.relative_to(WIKI_DIR)
            results.append({
                "file": str(rel),
                "match_count": match_count,
                "matches": file_results[:3],  # top 3 matches per file
            })

    # Sort by match count descending
    results.sort(key=lambda x: x["match_count"], reverse=True)
    return results


def print_results(results: list[dict], query: str, index_only: bool = False) -> None:
    if not results:
        print(f"No results for '{query}'")
        return

    total_matches = sum(r.get("match_count", 1) for r in results)
    print(f"\n{total_matches} match(es) in {len(results)} file(s) for '{query}'\n")
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
    parser.add_argument("--index-only", action="store_true", help="Search only _index.md (fast)")
    parser.add_argument("--tag", type=str, help="Filter by tag")
    parser.add_argument("--section", type=str,
                        help="Search within a specific section (e.g., concepts, entities, events, research)")
    args = parser.parse_args()

    if args.index_only:
        results = search_index(args.query)
        print_results(results, args.query, index_only=True)
    else:
        results = search_wiki(args.query, section=args.section, tag=args.tag)
        print_results(results, args.query)


if __name__ == "__main__":
    main()
