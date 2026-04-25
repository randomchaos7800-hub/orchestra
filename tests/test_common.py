"""Tests for lib/common.py shared utilities."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import (
    parse_frontmatter, split_frontmatter, write_article,
    extract_typed_links, extract_wikilink_slugs,
    parse_llm_json, sanitize_content, llm_call,
    inject_reciprocal_backlinks, load_config,
    LINK_TYPES, INVERSE_LINK_TYPE,
)


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        text = "---\ntitle: Test\ntags: [a, b]\n---\n\nBody here."
        fm = parse_frontmatter(text)
        assert fm["title"] == "Test"
        assert fm["tags"] == ["a", "b"]

    def test_no_frontmatter(self):
        assert parse_frontmatter("Just a body.") == {}

    def test_empty_frontmatter(self):
        fm = parse_frontmatter("---\n---\nBody")
        assert fm == {}

    def test_invalid_yaml(self):
        fm = parse_frontmatter("---\n: invalid: yaml: [[\n---\nBody")
        assert fm == {}

    def test_non_dict_yaml(self):
        fm = parse_frontmatter("---\n- item1\n- item2\n---\nBody")
        assert fm == {}

    def test_missing_end_delimiter(self):
        fm = parse_frontmatter("---\ntitle: Test\nno end")
        assert fm == {}


class TestSplitFrontmatter:
    def test_splits_correctly(self):
        text = "---\ntitle: Test\n---\n\nBody content."
        fm, body = split_frontmatter(text)
        assert fm["title"] == "Test"
        assert body == "Body content."

    def test_no_frontmatter(self):
        fm, body = split_frontmatter("Just body.")
        assert fm == {}
        assert body == "Just body."

    def test_preserves_body(self):
        text = "---\ntitle: X\n---\n\n## Heading\n\nParagraph."
        fm, body = split_frontmatter(text)
        assert "## Heading" in body
        assert "Paragraph." in body


class TestWriteArticle:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "test.md"
        fm = {"title": "Test", "tags": ["a", "b"]}
        body = "## Overview\n\nContent here."
        write_article(path, fm, body)

        text = path.read_text(encoding="utf-8")
        assert text.startswith("---")
        fm2, body2 = split_frontmatter(text)
        assert fm2["title"] == "Test"
        assert "Content here." in body2

    def test_dict_links(self, tmp_path):
        path = tmp_path / "test.md"
        fm = {"title": "X", "links": [{"target": "foo", "type": "references"}]}
        write_article(path, fm, "Body")
        fm2 = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert fm2["links"][0]["target"] == "foo"


class TestExtractTypedLinks:
    def test_typed_link(self):
        links = extract_typed_links("See [[depends_on:llm-agents]] for more.")
        assert len(links) == 1
        assert links[0]["target"] == "llm-agents"
        assert links[0]["type"] == "depends_on"

    def test_bare_link(self):
        links = extract_typed_links("See [[llm-agents]] for more.")
        assert len(links) == 1
        assert links[0]["type"] == "references"

    def test_mixed(self):
        text = "[[depends_on:a]] and [[b]] and [[extends:c]]"
        links = extract_typed_links(text)
        assert len(links) == 3
        targets = {l["target"] for l in links}
        assert targets == {"a", "b", "c"}

    def test_deduplication(self):
        links = extract_typed_links("[[references:x]] and [[x]]")
        assert len(links) == 1  # x only counted once

    def test_no_links(self):
        assert extract_typed_links("No links here.") == []


class TestExtractWikilinkSlugs:
    def test_mixed_slugs(self):
        text = "[[depends_on:a]] and [[b]]"
        slugs = extract_wikilink_slugs(text)
        assert "a" in slugs
        assert "b" in slugs

    def test_empty(self):
        assert extract_wikilink_slugs("No links.") == []


class TestParseLlmJson:
    def test_clean_json(self):
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        result = parse_llm_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_preamble(self):
        result = parse_llm_json('Here is the result:\n{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json(self):
        assert parse_llm_json("not json at all") is None

    def test_empty(self):
        assert parse_llm_json("") is None

    def test_nested_json(self):
        raw = '{"articles": [{"path": "concepts/test.md", "title": "Test"}]}'
        result = parse_llm_json(raw)
        assert len(result["articles"]) == 1


class TestSanitizeContent:
    def test_strips_null_bytes(self):
        assert "\x00" not in sanitize_content("hello\x00world")

    def test_strips_control_chars(self):
        result = sanitize_content("hello\x01world\x02end")
        assert "\x01" not in result
        assert "\x02" not in result

    def test_preserves_newlines_tabs(self):
        result = sanitize_content("hello\nworld\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_truncation(self):
        result = sanitize_content("x" * 20000, max_length=100)
        assert len(result) < 200
        assert "[... truncated ...]" in result

    def test_normal_text_unchanged(self):
        text = "Normal text with no issues."
        assert sanitize_content(text) == text


class TestConstants:
    def test_link_types_complete(self):
        assert "references" in LINK_TYPES
        assert "depends_on" in LINK_TYPES
        assert "extends" in LINK_TYPES
        assert "contradicts" in LINK_TYPES
        assert "related" in LINK_TYPES

    def test_inverse_types_cover_all(self):
        for lt in LINK_TYPES:
            assert lt in INVERSE_LINK_TYPE, f"Missing inverse for {lt}"

    def test_inverse_values_valid(self):
        valid_inverses = {"referenced_by", "related"}
        for lt, inv in INVERSE_LINK_TYPE.items():
            assert inv in valid_inverses, f"Invalid inverse '{inv}' for '{lt}'"


class TestLlmCall:
    def _mock_client(self, content: str):
        client = MagicMock()
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        return client

    def test_returns_response_text(self):
        client = self._mock_client("hello world")
        result = llm_call(client, "model", "sys", "user")
        assert result == "hello world"

    def test_strips_whitespace(self):
        client = self._mock_client("  trimmed  ")
        assert llm_call(client, "model", "sys", "user") == "trimmed"

    def test_none_content_returns_empty(self):
        client = self._mock_client(None)
        assert llm_call(client, "model", "sys", "user") == ""

    def test_passes_temperature_zero(self):
        client = self._mock_client("ok")
        llm_call(client, "mymodel", "sys", "user", max_tokens=100)
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_passes_correct_model(self):
        client = self._mock_client("ok")
        llm_call(client, "specific-model", "sys", "user")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "specific-model"

    def test_retry_on_failure_then_success(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "success"
        choice = MagicMock()
        choice.message = msg
        # Fail twice, succeed on third attempt
        client.chat.completions.create.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            MagicMock(choices=[choice]),
        ]
        with patch("lib.common.time.sleep"):
            result = llm_call(client, "model", "sys", "user")
        assert result == "success"
        assert client.chat.completions.create.call_count == 3

    def test_raises_after_three_failures(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("always fails")
        with patch("lib.common.time.sleep"):
            with pytest.raises(RuntimeError, match="3 attempts"):
                llm_call(client, "model", "sys", "user")
        assert client.chat.completions.create.call_count == 3

    def test_sleeps_between_retries(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "ok"
        choice = MagicMock()
        choice.message = msg
        client.chat.completions.create.side_effect = [
            Exception("fail"),
            MagicMock(choices=[choice]),
        ]
        with patch("lib.common.time.sleep") as mock_sleep:
            llm_call(client, "model", "sys", "user")
        mock_sleep.assert_called_once_with(1)  # 2**0 = 1s after first failure

    def test_request_delay_applied(self):
        client = self._mock_client("ok")
        with patch("lib.common.time.sleep") as mock_sleep:
            llm_call(client, "model", "sys", "user", request_delay=0.5)
        mock_sleep.assert_called_once_with(0.5)


class TestLoadConfig:
    def test_missing_config_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(tmp_path / "nonexistent.json")

    def test_valid_config_loaded(self, tmp_path):
        cfg = {"llm": {"local_url": "http://localhost:8081/v1"}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(cfg), encoding="utf-8")
        result = load_config(config_file)
        assert result["llm"]["local_url"] == "http://localhost:8081/v1"


class TestInjectReciprocalBacklinks:
    def test_backlink_injected(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)

        # Target article exists
        target = concepts / "target-article.md"
        target.write_text(
            "---\ntitle: Target\nlinks: []\n---\n\nBody.",
            encoding="utf-8",
        )

        links = [{"target": "target-article", "type": "references"}]
        inject_reciprocal_backlinks("source-article", links, wiki_dir=wiki)

        from lib.common import parse_frontmatter
        fm = parse_frontmatter(target.read_text(encoding="utf-8"))
        assert any(l["target"] == "source-article" for l in fm.get("links", []))

    def test_no_duplicate_backlinks(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)

        target = concepts / "target.md"
        target.write_text(
            "---\ntitle: T\nlinks:\n- target: source\n  type: referenced_by\n---\n\nBody.",
            encoding="utf-8",
        )

        links = [{"target": "target", "type": "references"}]
        inject_reciprocal_backlinks("source", links, wiki_dir=wiki)

        from lib.common import parse_frontmatter
        fm = parse_frontmatter(target.read_text(encoding="utf-8"))
        source_links = [l for l in fm.get("links", []) if l.get("target") == "source"]
        assert len(source_links) == 1

    def test_missing_target_is_skipped(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        # No articles on disk — should not raise
        links = [{"target": "ghost-article", "type": "references"}]
        inject_reciprocal_backlinks("source", links, wiki_dir=wiki)  # no exception
