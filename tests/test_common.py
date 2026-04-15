"""Tests for lib/common.py shared utilities."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import (
    parse_frontmatter, split_frontmatter, write_article,
    extract_typed_links, extract_wikilink_slugs,
    parse_llm_json, sanitize_content,
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
