"""Integration coverage for the capture -> compile -> query pipeline."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture import extract as capture_extract
from lib import common
from tools import compile as compile_tool
from tools import query as query_tool


def test_capture_compile_query_pipeline_is_idempotent(tmp_path, monkeypatch):
    root = tmp_path
    raw_dir = root / "raw" / "manual"
    wiki_dir = root / "wiki"
    capture_projects = root / "capture" / "projects"
    config_dir = root / "config"

    raw_dir.mkdir(parents=True)
    (wiki_dir / "concepts").mkdir(parents=True)
    (wiki_dir / "entities").mkdir()
    (wiki_dir / "events").mkdir()
    (wiki_dir / "research").mkdir()
    (wiki_dir / "tools").mkdir()
    (wiki_dir / "meta").mkdir()
    capture_projects.mkdir(parents=True)
    config_dir.mkdir()

    (wiki_dir / "_index.md").write_text("# Wiki Index\n", encoding="utf-8")
    (wiki_dir / "_sources.json").write_text(json.dumps({"processed": {}}, indent=2), encoding="utf-8")
    (config_dir / "compile-rules.md").write_text("Rules.", encoding="utf-8")
    (config_dir / "wiki-style.md").write_text("Style.", encoding="utf-8")

    monkeypatch.setattr(common, "KB_ROOT", root)
    monkeypatch.setattr(common, "RAW_DIR", root / "raw")
    monkeypatch.setattr(common, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(common, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(common, "INDEX_FILE", wiki_dir / "_index.md")
    monkeypatch.setattr(common, "SOURCES_FILE", wiki_dir / "_sources.json")

    monkeypatch.setattr(compile_tool, "KB_ROOT", root)
    monkeypatch.setattr(compile_tool, "RAW_DIR", root / "raw")
    monkeypatch.setattr(compile_tool, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(compile_tool, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(compile_tool, "INDEX_FILE", wiki_dir / "_index.md")

    monkeypatch.setattr(query_tool, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(query_tool, "INDEX_FILE", wiki_dir / "_index.md")

    monkeypatch.setattr(capture_extract, "PROJECT_ROOT", root)
    monkeypatch.setattr(capture_extract, "PROJECTS_DIR", capture_projects)
    monkeypatch.setattr(capture_extract, "PROCESSED_PATH", root / "capture" / "processed.json")

    config = {
        "llm": {
            "local_url": "http://127.0.0.1:8010/v1",
            "local_model": "test-model",
            "local_max_tokens": 1000,
        },
        "capture": {
            "projects": {
                "GENERAL": "Cross-cutting insights.",
                "RESEARCH": "Research findings.",
            },
            "min_messages": 1,
            "skip_titles": [],
            "max_conversation_chars": 12000,
        },
        "wiki": {"sections": ["concepts", "entities", "events", "research", "tools"]},
    }
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    conversation = {
        "id": "conv-1",
        "title": "Agent memory architecture",
        "updated_at": "2026-06-08T10:00:00Z",
        "messages": [
            {"role": "user", "content": "Design a memory system for agents."},
            {"role": "assistant", "content": "Use layered memory with summaries, facts, and retrieval."},
            {"role": "user", "content": "Explain the retrieval layer and compaction strategy in detail."},
        ],
    }
    export_file = root / "conversations.json"
    export_file.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(capture_extract, "load_config", lambda _: config)
    monkeypatch.setattr(capture_extract, "detect_and_parse", lambda _: [conversation])
    monkeypatch.setattr(capture_extract, "make_llm_client", lambda config=None: (MagicMock(), "model", 1000))
    monkeypatch.setattr(compile_tool, "make_llm_client", lambda config=None: (MagicMock(), "model", 1000))
    monkeypatch.setattr(query_tool, "make_llm_client", lambda: (MagicMock(), "model", 1000))

    monkeypatch.setattr(
        capture_extract,
        "cached_llm_call",
        lambda *args, **kwargs: json.dumps({
            "projects": ["RESEARCH"],
            "skip_reason": "",
            "entries": [{
                "project": "RESEARCH",
                "title": "Memory architecture",
                "content": "Agents use layered memory with summaries, durable facts, and retrieval over curated notes.",
                "trigger": "",
            }],
        }),
    )

    def compile_llm(*args, **kwargs):
        namespace = kwargs.get("cache_namespace")
        if namespace == "compile-plan":
            return json.dumps({
                "articles": [{
                    "path": "concepts/agent-memory-architecture.md",
                    "action": "create",
                    "title": "Agent Memory Architecture",
                    "summary": "Layered memory for long-running agents.",
                    "tags": ["agents", "memory"],
                    "sections": ["Overview", "Retrieval", "Compaction"],
                    "core_concepts": ["agent memory", "retrieval", "compaction"],
                }]
            })
        if namespace == "compile-article":
            return """---
title: Agent Memory Architecture
tags: [agents, memory]
updated: 2026-06-08
sources: [raw/manual/capture-derived.md]
---

## Overview

Layered memory uses summaries, durable facts, and retrieval over curated notes.

## Retrieval

The retrieval layer serves the right memory shard for the current task.

## Compaction

Compaction preserves durable facts while shrinking verbose transcripts.
"""
        if namespace == "compile-expand-concepts":
            return "long-term memory, episodic memory, retrieval"
        raise AssertionError(f"Unexpected compile namespace: {namespace}")

    monkeypatch.setattr(compile_tool, "cached_llm_call", compile_llm)
    monkeypatch.setattr(
        query_tool,
        "cached_llm_call",
        lambda *args, **kwargs: "Layered memory uses summaries, durable facts, and retrieval.",
    )

    monkeypatch.setattr(sys, "argv", ["extract.py", "--input", str(export_file)])
    capture_extract.main()

    project_file = capture_projects / "RESEARCH.md"
    project_text = project_file.read_text(encoding="utf-8")
    assert "Memory architecture" in project_text
    assert project_text.count("## 2026-06-08 - Memory architecture") == 1

    derived_raw = raw_dir / "capture-derived.md"
    derived_raw.write_text(project_text, encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["compile.py"])
    compile_tool.main()

    article = wiki_dir / "concepts" / "agent-memory-architecture.md"
    assert article.exists()
    article_text = article.read_text(encoding="utf-8")
    assert "## Retrieval" in article_text
    assert "Layered memory uses summaries" in article_text

    index_text = (wiki_dir / "_index.md").read_text(encoding="utf-8")
    assert "[[agent-memory-architecture]]" in index_text

    answer = query_tool.answer_question("What is the agent memory architecture?")
    assert "Layered memory uses summaries" in answer
    assert "`concepts/agent-memory-architecture.md`" in answer

    sources = json.loads((wiki_dir / "_sources.json").read_text(encoding="utf-8"))
    assert list(sources["processed"].keys()) == ["raw/manual/capture-derived.md"]

    monkeypatch.setattr(sys, "argv", ["extract.py", "--input", str(export_file)])
    capture_extract.main()
    assert project_file.read_text(encoding="utf-8").count("## 2026-06-08 - Memory architecture") == 1
    processed = json.loads((root / "capture" / "processed.json").read_text(encoding="utf-8"))
    assert processed["processed_ids"] == ["conv-1"]

    before_article = article.read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compile.py"])
    compile_tool.main()
    assert article.read_text(encoding="utf-8") == before_article
    sources_again = json.loads((wiki_dir / "_sources.json").read_text(encoding="utf-8"))
    assert list(sources_again["processed"].keys()) == ["raw/manual/capture-derived.md"]
