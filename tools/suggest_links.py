#!/usr/bin/env python3
"""
Orchestra Link Suggester -- tools/suggest_links.py

For each wiki article, finds similar articles via ChromaDB vector search
and suggests new links that don't already exist in either direction.

Writes results to wiki/meta/suggestions.md (or prints with --dry-run).

Requires: chromadb, sentence-transformers (same as search_hybrid.py)
  pip install chromadb sentence-transformers

Usage:
  python3 tools/suggest_links.py                  # default threshold 0.65
  python3 tools/suggest_links.py --threshold 0.7  # stricter
  python3 tools/suggest_links.py --dry-run        # print to stdout only
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.common import WIKI_DIR, parse_frontmatter
from search_hybrid import _get_collection, _collect_articles, _rel

SUGGESTIONS_FILE = WIKI_DIR / "meta" / "suggestions.md"


def _load_article_links(articles: list[Path]) -> dict[str, set[str]]:
    """Return {rel_id: set_of_link_targets} for all articles."""
    link_map: dict[str, set[str]] = {}
    for path in articles:
        rel = _rel(path)
        try:
            fm = parse_frontmatter(path.read_text(encoding="utf-8"))
        except Exception:
            link_map[rel] = set()
            continue
        links = fm.get("links", [])
        targets = set()
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and link.get("target"):
                    targets.add(link["target"])
        link_map[rel] = targets
    return link_map


def _already_linked(source_rel: str, target_rel: str, link_map: dict[str, set[str]]) -> bool:
    """Check if there's a link between source and target in either direction."""
    source_slug = Path(source_rel).stem
    target_slug = Path(target_rel).stem

    source_links = link_map.get(source_rel, set())
    if target_slug in source_links or target_rel in source_links:
        return True

    target_links = link_map.get(target_rel, set())
    if source_slug in target_links or source_rel in target_links:
        return True

    return False


def generate_suggestions(threshold: float = 0.65) -> list[dict]:
    """
    For each article, query ChromaDB for top 15 most similar articles.
    Returns list of {article, title, suggestions: [{target, score, title}]} dicts.

    O(n) ChromaDB queries where n = number of articles. Fine up to ~500 articles;
    noticeably slow beyond that. A batch approach (fetch all embeddings, compute
    similarity matrix in numpy) would be O(1) queries at the cost of memory.
    """
    _, col, _ = _get_collection()
    total = col.count()
    if total == 0:
        print("Index is empty. Run: python3 tools/search_hybrid.py --reindex", file=sys.stderr)
        sys.exit(1)

    articles = _collect_articles()
    link_map = _load_article_links(articles)

    all_ids = [_rel(p) for p in articles]
    all_data = col.get(ids=all_ids, include=["metadatas", "embeddings"])

    id_meta_map: dict[str, dict] = {}
    id_emb_map: dict[str, list] = {}
    for doc_id, meta, emb in zip(all_data["ids"], all_data["metadatas"], all_data["embeddings"]):
        id_meta_map[doc_id] = meta
        id_emb_map[doc_id] = emb

    indexed_ids = set(all_data["ids"])
    print(f"Analyzing {len(indexed_ids)} articles...", file=sys.stderr)

    results = []
    n_results = min(16, total)

    for path in articles:
        article_rel = _rel(path)
        if article_rel not in indexed_ids:
            continue

        article_meta = id_meta_map.get(article_rel, {})
        article_title = article_meta.get("title", path.stem)
        article_emb = id_emb_map.get(article_rel)
        if article_emb is None:
            continue

        try:
            qr = col.query(
                query_embeddings=[article_emb],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            print(f"  Warning: query failed for {article_rel}: {e}", file=sys.stderr)
            continue

        suggestions = []
        for sim_id, sim_meta, dist in zip(qr["ids"][0], qr["metadatas"][0], qr["distances"][0]):
            if sim_id == article_rel:
                continue
            if sim_meta.get("section") == "meta":
                continue

            similarity = 1.0 - dist
            if similarity < threshold:
                continue

            if _already_linked(article_rel, sim_id, link_map):
                continue

            suggestions.append({
                "target": sim_id,
                "score": similarity,
                "title": sim_meta.get("title", Path(sim_id).stem),
            })

        if suggestions:
            suggestions.sort(key=lambda x: x["score"], reverse=True)
            results.append({
                "article": article_rel,
                "title": article_title,
                "suggestions": suggestions,
            })

    results.sort(key=lambda x: len(x["suggestions"]), reverse=True)
    return results


def format_suggestions(results: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Link Suggestions",
        "",
        "Auto-generated by tools/suggest_links.py. Review and add links manually or re-run compile.",
        "",
        f"*Last updated: {now}*",
        "",
        "---",
        "",
    ]

    for entry in results:
        lines.append(f"## {entry['article']} -- {entry['title']}")
        lines.append("")
        for sug in entry["suggestions"]:
            lines.append(f"- **{sug['target']}** (score: {sug['score']:.3f}) -- {sug['title']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest missing wiki links via vector similarity")
    parser.add_argument("--threshold", type=float, default=0.65, metavar="SCORE",
                        help="Minimum similarity score (default: 0.65)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print to stdout instead of writing suggestions.md")
    args = parser.parse_args()

    results = generate_suggestions(threshold=args.threshold)

    if not results:
        print("No suggestions found above threshold.")
        return

    total_suggestions = sum(len(r["suggestions"]) for r in results)
    print(f"Found {total_suggestions} suggestions across {len(results)} articles.", file=sys.stderr)

    output = format_suggestions(results)

    if args.dry_run:
        print(output)
    else:
        SUGGESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SUGGESTIONS_FILE.write_text(output, encoding="utf-8")
        print(f"Written to {SUGGESTIONS_FILE}")
        print("\n--- Preview (first 30 lines) ---")
        for line in output.splitlines()[:30]:
            print(line)


if __name__ == "__main__":
    main()
