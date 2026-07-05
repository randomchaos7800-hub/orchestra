"""Tests for query and hybrid-search tool behavior."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import query as query_tool
from tools import compile as compile_tool
from tools import search_hybrid


class FakeCollection:
    def __init__(self):
        self.deleted_ids = []
        self.upserted_ids = []

    def get(self, ids=None, include=None):
        if ids is not None:
            return {"ids": [], "metadatas": []}
        return {"ids": ["concepts/stale.md"]}

    def delete(self, ids):
        self.deleted_ids.extend(ids)

    def upsert(self, ids, documents, metadatas):
        self.upserted_ids.extend(ids)


def test_query_prefers_hybrid_search_when_available(tmp_path: Path, monkeypatch):
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    article = wiki_dir / "concepts" / "agent-memory.md"
    article.write_text("# Agent Memory\n\nUseful content.", encoding="utf-8")

    monkeypatch.setattr(query_tool, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(query_tool, "INDEX_FILE", wiki_dir / "_index.md")
    query_tool.INDEX_FILE.write_text("# Index\n", encoding="utf-8")

    monkeypatch.setattr(
        query_tool,
        "retrieve_articles",
        lambda question, wiki_dir, mode, top_k: [{"id": "concepts/agent-memory.md"}],
    )

    result = query_tool._find_relevant_articles("What is agent memory?", mode="hybrid")
    assert result == [article]


def test_query_appends_source_paths(tmp_path: Path, monkeypatch):
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    article = wiki_dir / "concepts" / "agent-memory.md"
    article.write_text("# Agent Memory\n\nUseful content.", encoding="utf-8")
    index_file = wiki_dir / "_index.md"
    index_file.write_text("# Index\n", encoding="utf-8")

    monkeypatch.setattr(query_tool, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(query_tool, "INDEX_FILE", index_file)
    monkeypatch.setattr(query_tool, "_find_relevant_articles", lambda question, mode="lexical": [article])
    monkeypatch.setattr(query_tool, "make_llm_client", lambda: (MagicMock(), "model", 1000))
    monkeypatch.setattr(
        query_tool,
        "cached_llm_call",
        lambda *args, **kwargs: "Answer body.",
    )

    result = query_tool.answer_question("What is agent memory?")
    assert "## Sources" in result
    assert "`concepts/agent-memory.md`" in result


def test_hybrid_index_prunes_deleted_articles(tmp_path: Path, monkeypatch):
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    article = wiki_dir / "concepts" / "fresh.md"
    article.write_text("---\ntitle: Fresh\n---\n\nBody", encoding="utf-8")

    fake_collection = FakeCollection()
    monkeypatch.setattr(search_hybrid, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(search_hybrid, "_get_collection", lambda chroma_dir=None: (None, fake_collection, None))
    monkeypatch.setattr(search_hybrid, "_collect_articles", lambda: [article])

    updated = search_hybrid.index_articles(force=False, verbose=False)

    assert fake_collection.deleted_ids == ["concepts/stale.md"]
    assert updated == 1
    assert fake_collection.upserted_ids == ["concepts/fresh.md"]


def test_hybrid_search_supports_full_corpus_bm25(tmp_path: Path, monkeypatch):
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    exact = wiki_dir / "concepts" / "warp-drive.md"
    exact.write_text("---\ntitle: Warp Drive\n---\n\nWarp drive exact phrase.", encoding="utf-8")
    semantic = wiki_dir / "concepts" / "space-travel.md"
    semantic.write_text("---\ntitle: Space Travel\n---\n\nGeneral travel article.", encoding="utf-8")

    monkeypatch.setattr(search_hybrid, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(
        search_hybrid,
        "retrieve_articles",
        lambda query, wiki_dir, mode, top_k, chroma_dir: [{
            "id": "concepts/warp-drive.md",
            "title": "Warp Drive",
            "tags": "",
            "updated": "",
            "section": "concepts",
            "fused_score": 1.0,
            "snippet": "Warp drive exact phrase.",
        }],
    )

    results = search_hybrid.hybrid_search("warp drive", top_n=3, full_corpus=True)
    ids = [row["id"] for row in results]
    assert "concepts/warp-drive.md" in ids


def test_compile_rejects_invalid_plan_before_writing(tmp_path: Path, monkeypatch):
    wiki_dir = tmp_path / "wiki"
    raw_dir = tmp_path / "raw"
    config_dir = tmp_path / "config"
    (wiki_dir / "concepts").mkdir(parents=True)
    raw_dir.mkdir()
    config_dir.mkdir()

    raw_file = raw_dir / "note.md"
    raw_file.write_text("# Note\n\nUseful source.", encoding="utf-8")
    (config_dir / "compile-rules.md").write_text("Rules.", encoding="utf-8")
    (config_dir / "wiki-style.md").write_text("Style.", encoding="utf-8")

    monkeypatch.setattr(compile_tool, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(compile_tool, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(compile_tool, "KB_ROOT", tmp_path)
    monkeypatch.setattr(compile_tool, "cached_llm_call", lambda *args, **kwargs: '{"articles":[{"action":"create","title":"Bad"}]}')

    touched = compile_tool.compile_file(raw_file, MagicMock(), "model", dry_run=False)
    assert touched is None
    assert list((wiki_dir / "concepts").glob("*.md")) == []
