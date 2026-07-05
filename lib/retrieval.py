"""Shared retrieval utilities for Orchestra wiki tools.

The default path is full-corpus lexical retrieval over wiki article bodies.
Hybrid mode adds optional vector results from the ChromaDB index and fuses the
two ranked lists with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from lib.common import KB_ROOT, WIKI_DIR, split_frontmatter

DEFAULT_CHROMA_DIR = KB_ROOT / ".chroma"
COLLECTION_NAME = "wiki_articles"


def tokenize(text: str) -> list[str]:
    """Tokenize prose and code identifiers for lexical retrieval."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def collect_articles(wiki_dir: Path | None = None) -> list[Path]:
    """Return indexable wiki articles, excluding _index.md and meta reports."""
    wd = wiki_dir or WIKI_DIR
    if not wd.exists():
        return []
    result = []
    for path in wd.rglob("*.md"):
        if path.name == "_index.md":
            continue
        if path.parent.name == "meta":
            continue
        result.append(path)
    return sorted(result)


def rel_id(path: Path, wiki_dir: Path | None = None) -> str:
    """Return the retrieval document id for an article path."""
    return str(path.relative_to(wiki_dir or WIKI_DIR))


def build_document(path: Path, wiki_dir: Path | None = None) -> tuple[str, dict, str] | None:
    """Return (doc_id, metadata, document_text) for an article, or None on read errors."""
    wd = wiki_dir or WIKI_DIR
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None

    frontmatter, body = split_frontmatter(raw)
    title = frontmatter.get("title", path.stem.replace("-", " ").title())
    tags = frontmatter.get("tags", "")
    if isinstance(tags, list):
        tags = ", ".join(str(tag) for tag in tags)
    updated = frontmatter.get("updated", frontmatter.get("last_compiled", ""))
    section = path.parent.name

    metadata = {
        "path": rel_id(path, wd),
        "title": str(title),
        "tags": str(tags),
        "updated": str(updated),
        "section": section,
        "indexed_at": str(int(path.stat().st_mtime)),
    }
    return metadata["path"], metadata, f"{title}\n\n{body.strip()}"


def _snippet(document: str) -> str:
    body_lines = document.split("\n", 2)
    return body_lines[2].strip()[:160] if len(body_lines) > 2 else ""


def _exact_query_boost(query: str, document: str) -> float:
    """Boost exact substring matches so code identifiers rank predictably."""
    query_norm = query.strip().lower()
    if not query_norm:
        return 0.0
    document_norm = document.lower()
    if query_norm in document_norm:
        return 2.0
    return sum(0.25 for token in set(tokenize(query)) if token in document_norm)


def _article_rows(wiki_dir: Path | None = None) -> list[tuple[str, dict, str]]:
    rows = []
    wd = wiki_dir or WIKI_DIR
    for path in collect_articles(wd):
        row = build_document(path, wd)
        if row is not None:
            rows.append(row)
    return rows


def _lexical_results(query: str, wiki_dir: Path | None = None, top_k: int = 10) -> list[dict]:
    rows = _article_rows(wiki_dir)
    if not rows:
        return []

    query_terms = tokenize(query)
    tokenized_docs = [tokenize(document) for _, _, document in rows]
    bm25 = BM25Okapi(tokenized_docs)
    raw_scores = bm25.get_scores(query_terms) if query_terms else [0.0] * len(rows)

    results = []
    for (doc_id, metadata, document), raw_score in zip(rows, raw_scores):
        lexical_score = float(raw_score) + _exact_query_boost(query, document)
        results.append({
            "id": doc_id,
            "path": metadata["path"],
            "title": metadata.get("title", doc_id),
            "tags": metadata.get("tags", ""),
            "updated": metadata.get("updated", ""),
            "section": metadata.get("section", ""),
            "score": lexical_score,
            "lexical_score": lexical_score,
            "vector_score": 0.0,
            "fused_score": lexical_score,
            "snippet": _snippet(document),
            "mode": "lexical",
        })

    results.sort(key=lambda item: (item["lexical_score"], item["title"]), reverse=True)
    return results[:top_k]


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Merge ranked document ids with Reciprocal Rank Fusion."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _get_vector_collection(chroma_dir: Path | None = None):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(chroma_dir or DEFAULT_CHROMA_DIR))
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def _vector_results(query: str, top_k: int = 20, chroma_dir: Path | None = None) -> list[dict]:
    collection = _get_vector_collection(chroma_dir)
    total = collection.count()
    if total == 0:
        return []

    result_count = min(max(top_k, 1), total)
    raw = collection.query(
        query_texts=[query],
        n_results=result_count,
        include=["metadatas", "documents", "distances"],
    )
    results = []
    for doc_id, metadata, document, distance in zip(
        raw["ids"][0],
        raw["metadatas"][0],
        raw["documents"][0],
        raw["distances"][0],
    ):
        vector_score = 1.0 - float(distance)
        results.append({
            "id": doc_id,
            "path": metadata.get("path", doc_id),
            "title": metadata.get("title", doc_id),
            "tags": metadata.get("tags", ""),
            "updated": metadata.get("updated", ""),
            "section": metadata.get("section", ""),
            "score": vector_score,
            "lexical_score": 0.0,
            "vector_score": vector_score,
            "fused_score": vector_score,
            "snippet": _snippet(document or ""),
            "mode": "vector",
        })
    return results


def _hybrid_results(
    query: str,
    wiki_dir: Path | None = None,
    top_k: int = 10,
    chroma_dir: Path | None = None,
) -> list[dict]:
    lexical = _lexical_results(query, wiki_dir, top_k=max(top_k * 4, 20))
    lexical_by_id = {row["id"]: row for row in lexical}
    lexical_ranking = [row["id"] for row in lexical]

    try:
        vector = _vector_results(query, top_k=max(top_k * 4, 20), chroma_dir=chroma_dir)
    except Exception:
        for row in lexical[:top_k]:
            row["mode"] = "lexical"
        return lexical[:top_k]

    if not vector:
        for row in lexical[:top_k]:
            row["mode"] = "lexical"
        return lexical[:top_k]

    vector_by_id = {row["id"]: row for row in vector}
    vector_ranking = [row["id"] for row in vector]
    fused_scores = _rrf([lexical_ranking, vector_ranking])

    combined_ids = set(lexical_by_id) | set(vector_by_id)
    combined = []
    for doc_id in combined_ids:
        base = dict(lexical_by_id.get(doc_id) or vector_by_id[doc_id])
        lexical_score = lexical_by_id.get(doc_id, {}).get("lexical_score", 0.0)
        vector_score = vector_by_id.get(doc_id, {}).get("vector_score", 0.0)
        base.update({
            "score": fused_scores.get(doc_id, 0.0),
            "lexical_score": lexical_score,
            "vector_score": vector_score,
            "fused_score": fused_scores.get(doc_id, 0.0),
            "mode": "hybrid",
        })
        combined.append(base)

    combined.sort(key=lambda item: item["fused_score"], reverse=True)
    return combined[:top_k]


def search(
    query: str,
    wiki_dir: Path | None = None,
    mode: str = "lexical",
    top_k: int = 10,
    chroma_dir: Path | None = None,
) -> list[dict]:
    """Search wiki articles.

    Args:
        query: User search query.
        wiki_dir: Wiki root directory.
        mode: "lexical" for full-corpus BM25, "hybrid" for BM25 + vector RRF.
        top_k: Maximum results to return.
        chroma_dir: Optional ChromaDB persistence directory for hybrid mode.
    """
    if mode not in {"lexical", "hybrid"}:
        raise ValueError("mode must be 'lexical' or 'hybrid'")
    if top_k <= 0:
        return []
    if mode == "hybrid":
        return _hybrid_results(query, wiki_dir, top_k=top_k, chroma_dir=chroma_dir)
    return _lexical_results(query, wiki_dir, top_k=top_k)
