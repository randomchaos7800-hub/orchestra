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
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import KB_ROOT, WIKI_DIR, split_frontmatter

CHROMA_DIR = KB_ROOT / ".chroma"
COLLECTION_NAME = "wiki_articles"


# -- Article discovery ---------------------------------------------------------

def _collect_articles() -> list[Path]:
    """Return all indexable .md files (exclude _index.md, meta/, sources)."""
    result = []
    for p in WIKI_DIR.rglob("*.md"):
        if p.name == "_index.md":
            continue
        if p.parent.name == "meta":
            continue
        result.append(p)
    return sorted(result)


def _rel(path: Path) -> str:
    return str(path.relative_to(WIKI_DIR))


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
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  Warning: cannot read {rel}: {e}", file=sys.stderr)
        return None

    meta_fm, body = split_frontmatter(raw)

    title = meta_fm.get("title", path.stem.replace("-", " ").title())
    tags = meta_fm.get("tags", "")
    if isinstance(tags, list):
        tags = ", ".join(tags)
    updated = meta_fm.get("updated", meta_fm.get("last_compiled", ""))
    section = path.parent.name

    chroma_meta = {
        "path": rel,
        "title": title,
        "tags": str(tags),
        "updated": str(updated),
        "section": section,
        "indexed_at": str(int(path.stat().st_mtime)),
    }
    return rel, chroma_meta, f"{title}\n\n{body.strip()}"


def index_articles(force: bool = False, verbose: bool = True) -> int:
    """Incrementally index all articles into ChromaDB. Returns count added/updated."""
    _, col, _ = _get_collection()
    articles = _collect_articles()

    existing: dict[str, str] = {}
    try:
        existing_data = col.get(ids=[_rel(p) for p in articles], include=["metadatas"])
        for doc_id, meta in zip(existing_data["ids"], existing_data["metadatas"]):
            existing[doc_id] = meta.get("indexed_at", "0")
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


# -- BM25 (stdlib-only) -------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _bm25_score(query_terms: list[str], docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Simple BM25 scorer. Returns a score per document."""
    N = len(docs)
    if N == 0:
        return []

    tokenized = [_tokenize(d) for d in docs]
    avg_dl = sum(len(t) for t in tokenized) / N

    df: dict[str, int] = {}
    for term in query_terms:
        df[term] = sum(1 for t in tokenized if term in t)

    scores = []
    for tokens in tokenized:
        dl = len(tokens)
        tf_map: dict[str, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        scores.append(score)

    return scores


# -- Reciprocal Rank Fusion ----------------------------------------------------

def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Merge ranked lists via Reciprocal Rank Fusion."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


# -- Hybrid search -------------------------------------------------------------

def hybrid_search(query: str, top_n: int = 5) -> list[dict]:
    """Hybrid BM25 + vector search with RRF fusion. Returns results sorted by score."""
    _, col, _ = _get_collection()

    total_count = col.count()
    if total_count == 0:
        return []

    n_vec = min(20, total_count)

    vec_results = col.query(
        query_texts=[query],
        n_results=n_vec,
        include=["metadatas", "documents", "distances"],
    )
    vec_ids: list[str] = vec_results["ids"][0]
    vec_metas: list[dict] = vec_results["metadatas"][0]
    vec_docs: list[str] = vec_results["documents"][0]
    vec_dists: list[float] = vec_results["distances"][0]

    vec_scores = {vid: 1.0 - vd for vid, vd in zip(vec_ids, vec_dists)}

    # BM25 scores only the top-N vector candidates, not the full corpus. Fast and good
    # enough for most queries, but exact-match queries (acronyms, proper names, code)
    # may fall outside the top-N and get no BM25 boost. A --full-corpus flag would fix
    # this at the cost of scoring every article on every query.
    query_terms = _tokenize(query)
    bm25_raw = _bm25_score(query_terms, vec_docs)
    bm25_ranking = [doc_id for doc_id, _ in sorted(zip(vec_ids, bm25_raw), key=lambda x: x[1], reverse=True)]
    bm25_scores = dict(zip(vec_ids, bm25_raw))

    fused = _rrf([vec_ids, bm25_ranking], k=60)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]

    meta_map = dict(zip(vec_ids, vec_metas))
    doc_map = dict(zip(vec_ids, vec_docs))

    results = []
    for doc_id, fused_score in ranked:
        meta = meta_map.get(doc_id, {})
        doc = doc_map.get(doc_id, "")
        body_lines = doc.split("\n", 2)
        snippet = body_lines[2].strip()[:120] if len(body_lines) > 2 else ""
        results.append({
            "id": doc_id,
            "title": meta.get("title", doc_id),
            "tags": meta.get("tags", ""),
            "updated": meta.get("updated", ""),
            "section": meta.get("section", ""),
            "fused_score": fused_score,
            "vec_score": vec_scores.get(doc_id, 0.0),
            "bm25_score": bm25_scores.get(doc_id, 0.0),
            "snippet": snippet,
        })
    return results


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
    parser.add_argument("--reindex", action="store_true", help="Force full reindex of all articles")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
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

    index_articles(force=False, verbose=False)
    results = hybrid_search(args.query, top_n=args.top)
    print_results(results, args.query)


if __name__ == "__main__":
    main()
