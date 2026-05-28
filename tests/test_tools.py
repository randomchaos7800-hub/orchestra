"""Tests for query and hybrid-search tool behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import query as query_tool
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

    # Prevent fallback local matching from contributing anything.
    monkeypatch.setattr(query_tool, "get_wiki_sections", lambda: [])

    class FakeHybridModule:
        @staticmethod
        def hybrid_search(question: str, top_n: int = 6):
            return [{"id": "concepts/agent-memory.md"}]

    monkeypatch.setitem(sys.modules, "tools.search_hybrid", FakeHybridModule)

    result = query_tool._find_relevant_articles("What is agent memory?")
    assert result == [article]


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
