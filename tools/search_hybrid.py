#!/usr/bin/env python3
"""
Orchestra Hybrid Search -- tools/search_hybrid.py

Hybrid BM25 + vector search over wiki/ articles using ChromaDB and
sentence-transformers (all-MiniLM-L6-v2). Incremental indexing based
on file mtime.

Requires: chromadb, sentence-transformers
  pip install chromadb sentence-transformers

Usage:
  python3 tools/search_hybrid.py "query string"           # search, default top 5
  python3 tools/search_hybrid.py "query string" --top 10  # custom result count
  python3 tools/search_hybrid.py --reindex                 # force full reindex
  python3 tools/search_hybrid.py --stats                   # show index stats
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import KB_ROOT, WIKI_DIR
from lib.retrieval import (
    COLLECTION_NAME,
    build_document,
    collect_articles,
    rel_id,
    search as retrieve_articles,
)

CHROMA_DIR = KB_ROOT / ".chroma"


# -- Article discovery ---------------------------------------------------------

def _collect_articles() -> list[Path]:
    """Return all indexable .md files (exclude _index.md, meta/, sources)."""
    return collect_articles(WIKI_DIR)


def _rel(path: Path) -> str:
    return rel_id(path, WIKI_DIR)


# -- ChromaDB + sentence-transformers setup ------------------------------------

def _get_collection(chroma_dir: Path | None = None):
    """Return (chroma_client, collection, embedding_function)."""
    import chromadb
    from chromadb.utils import embedding_functions

    d = str(chroma_dir or CHROMA_DIR)
    client = chromadb.PersistentClient(path=d)

    # sentence-transformers pulls PyTorch (~1-2GB). A lighter alternative is fastembed
    # (~100MB, ONNX-based), which ChromaDB supports natively:
    #   pip install fastembed  (instead of sentence-transformers)
    #   ef = embedding_functions.FastEmbedEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    # Switching requires --reindex to rebuild the index with the new model's embeddings.
    # We haven't validated quality parity yet — tracked, not forgotten.
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return client, col, ef


# -- Indexing ------------------------------------------------------------------

def _build_doc(path: Path, rel: str) -> tuple[str, dict, str] | None:
    """Return (doc_id, metadata, document_text) for a .md file, or None on error."""
    result = build_document(path, WIKI_DIR)
    if result is None:
        print(f"  Warning: cannot read {rel}", file=sys.stderr)
    return result


def index_articles(force: bool = False, verbose: bool = True) -> int:
    """Incrementally index all articles into ChromaDB. Returns count added/updated."""
    _, col, _ = _get_collection()
    articles = _collect_articles()
    article_ids = [_rel(p) for p in articles]

    existing: dict[str, str] = {}
    try:
        existing_data = col.get(ids=article_ids, include=["metadatas"])
        for doc_id, meta in zip(existing_data["ids"], existing_data["metadatas"]):
            existing[doc_id] = meta.get("indexed_at", "0")
    except Exception:
        pass

    # Remove index entries for articles that no longer exist on disk.
    try:
        all_indexed = col.get(include=[])
        stale_ids = [doc_id for doc_id in all_indexed.get("ids", []) if doc_id not in set(article_ids)]
        if stale_ids:
            col.delete(ids=stale_ids)
            if verbose:
                print(f"  Pruned {len(stale_ids)} stale indexed article(s).")
    except Exception:
        pass

    to_upsert_ids = []
    to_upsert_docs = []
    to_upsert_metas = []

    for path in articles:
        rel = _rel(path)
        mtime = str(int(path.stat().st_mtime))
        if not force and existing.get(rel) == mtime:
            continue

        result = _build_doc(path, rel)
        if result is None:
            continue
        doc_id, chroma_meta, document = result
        to_upsert_ids.append(doc_id)
        to_upsert_docs.append(document)
        to_upsert_metas.append(chroma_meta)

    if not to_upsert_ids:
        if verbose:
            print(f"Index up to date. {len(articles)} articles tracked, 0 changed.")
        return 0

    batch_size = 50
    for i in range(0, len(to_upsert_ids), batch_size):
        col.upsert(
            ids=to_upsert_ids[i:i + batch_size],
            documents=to_upsert_docs[i:i + batch_size],
            metadatas=to_upsert_metas[i:i + batch_size],
        )
        if verbose:
            print(f"  Indexed {min(i + batch_size, len(to_upsert_ids))}/{len(to_upsert_ids)} articles...")

    if verbose:
        print(f"Done. {len(to_upsert_ids)} article(s) indexed/updated ({len(articles)} total).")
    return len(to_upsert_ids)


# -- Hybrid search -------------------------------------------------------------

def hybrid_search(query: str, top_n: int = 5, full_corpus: bool = False) -> list[dict]:
    """Hybrid BM25 + vector search with lexical fallback.

    The full_corpus flag is retained for backward compatibility; the shared
    retrieval layer always scores BM25 across the full wiki corpus.
    """
    try:
        index_articles(force=False, verbose=False)
    except Exception:
        pass
    return retrieve_articles(
        query,
        wiki_dir=WIKI_DIR,
        mode="hybrid",
        top_k=top_n,
        chroma_dir=CHROMA_DIR,
    )


# -- Stats ---------------------------------------------------------------------

def show_stats() -> None:
    _, col, _ = _get_collection()
    count = col.count()
    articles = _collect_articles()

    last_ts = "never"
    if count > 0:
        try:
            sample = col.get(limit=count, include=["metadatas"])
            timestamps = [int(m.get("indexed_at", 0)) for m in sample["metadatas"] if m.get("indexed_at")]
            if timestamps:
                last_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(max(timestamps)))
        except Exception:
            pass

    print(f"ChromaDB index: {CHROMA_DIR}")
    print(f"Collection:     {COLLECTION_NAME}")
    print(f"Indexed docs:   {count}")
    print(f"Wiki articles:  {len(articles)} (indexable)")
    print(f"Last indexed:   {last_ts}")


# -- CLI output ----------------------------------------------------------------

def print_results(results: list[dict], query: str) -> None:
    if not results:
        print(f'No results for "{query}"')
        return

    print(f'\nSearch: "{query}" ({len(results)} results)\n')
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['id']} -- {r['title']} [score: {r['fused_score']:.3f}]")
        tags_str = r["tags"] if r["tags"] else "(none)"
        updated_str = r["updated"] if r["updated"] else "unknown"
        print(f"   Tags: {tags_str} | Updated: {updated_str}")
        if r["snippet"]:
            print(f"   {r['snippet']}")
        print()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid BM25 + vector search over wiki")
    parser.add_argument("query", nargs="?", help="Search query string")
    parser.add_argument("--top", type=int, default=5, metavar="N", help="Number of results (default 5)")
    parser.add_argument("--mode", choices=["lexical", "hybrid"], default="hybrid",
                        help="Retrieval mode: lexical BM25 or hybrid BM25+vector RRF")
    parser.add_argument("--reindex", action="store_true", help="Force full reindex of all articles")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Deprecated: BM25 is always scored across the full wiki corpus",
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.reindex:
        print("Reindexing all articles...")
        index_articles(force=True, verbose=True)
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    if args.mode == "hybrid":
        try:
            index_articles(force=False, verbose=False)
        except Exception:
            pass
    results = retrieve_articles(
        args.query,
        wiki_dir=WIKI_DIR,
        mode=args.mode,
        top_k=args.top,
        chroma_dir=CHROMA_DIR,
    )
    print_results(results, args.query)


if __name__ == "__main__":
    main()
