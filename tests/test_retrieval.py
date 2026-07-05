"""Tests for the shared retrieval layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import retrieval


def _write_article(path: Path, title: str, body: str, tags: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_text = ", ".join(tags or ["test"])
    path.write_text(
        f"---\ntitle: {title}\ntags: [{tag_text}]\nupdated: 2026-06-08\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_lexical_finds_code_identifier_not_in_slug(tmp_path: Path):
    wiki = tmp_path / "wiki"
    _write_article(
        wiki / "concepts" / "relay-behavior.md",
        "Relay Behavior",
        "The worker raises LoopDetectedError when recursive routing is detected.",
    )
    _write_article(
        wiki / "concepts" / "routing-overview.md",
        "Routing Overview",
        "General routing notes without the specific exception name.",
    )

    results = retrieval.search("LoopDetectedError", wiki_dir=wiki, mode="lexical", top_k=2)

    assert results[0]["id"] == "concepts/relay-behavior.md"
    assert results[0]["lexical_score"] > 0


def test_hybrid_fuses_lexical_and_vector_results(tmp_path: Path, monkeypatch):
    wiki = tmp_path / "wiki"
    _write_article(
        wiki / "concepts" / "alpha.md",
        "Alpha",
        "Alpha contains the exact FluxCapacitor marker.",
    )
    _write_article(
        wiki / "concepts" / "beta.md",
        "Beta",
        "Semantically similar vector result.",
    )

    monkeypatch.setattr(
        retrieval,
        "_vector_results",
        lambda *args, **kwargs: [{
            "id": "concepts/beta.md",
            "path": "concepts/beta.md",
            "title": "Beta",
            "tags": "test",
            "updated": "2026-06-08",
            "section": "concepts",
            "score": 0.95,
            "lexical_score": 0.0,
            "vector_score": 0.95,
            "fused_score": 0.95,
            "snippet": "Semantically similar vector result.",
            "mode": "vector",
        }],
    )

    results = retrieval.search("FluxCapacitor", wiki_dir=wiki, mode="hybrid", top_k=3)
    ids = {row["id"] for row in results}

    assert "concepts/alpha.md" in ids
    assert "concepts/beta.md" in ids
    assert all(row["mode"] == "hybrid" for row in results)


def test_hybrid_falls_back_to_lexical_when_vector_unavailable(tmp_path: Path, monkeypatch):
    wiki = tmp_path / "wiki"
    _write_article(
        wiki / "concepts" / "relay-behavior.md",
        "Relay Behavior",
        "LoopDetectedError appears in this article body.",
    )
    monkeypatch.setattr(
        retrieval,
        "_vector_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("chromadb missing")),
    )

    results = retrieval.search("LoopDetectedError", wiki_dir=wiki, mode="hybrid", top_k=1)

    assert results[0]["id"] == "concepts/relay-behavior.md"
    assert results[0]["mode"] == "lexical"
