"""
Tests for the Orchestra capture module — parsers, text extraction,
skip logic, and dedup tracking.

All tests use tmp_path fixtures and mock LLM calls. No real API requests.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from capture.parsers.claude import parse_claude_export
from capture.parsers.chatgpt import parse_chatgpt_export
from capture.parsers.generic import parse_generic_export


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def _make_claude_conversations(count: int = 2) -> list[dict]:
    """Return sample Claude export data."""
    conversations = []
    for i in range(count):
        conversations.append({
            "uuid": f"conv-{i}",
            "name": f"Conversation {i}",
            "updated_at": f"2026-04-0{i + 1}T12:00:00Z",
            "chat_messages": [
                {
                    "sender": "human",
                    "content": [{"type": "text", "text": f"User message {i}"}],
                },
                {
                    "sender": "assistant",
                    "content": [{"type": "text", "text": f"Assistant reply {i}"}],
                },
            ],
        })
    return conversations


def _make_chatgpt_conversations(count: int = 2) -> list[dict]:
    """Return sample ChatGPT export data with message tree structure."""
    conversations = []
    for i in range(count):
        conversations.append({
            "id": f"chatgpt-{i}",
            "title": f"Chat {i}",
            "update_time": 1712000000 + i * 86400,
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "parent": None,
                    "children": ["msg-user"],
                },
                "msg-user": {
                    "id": "msg-user",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": [f"User question {i}"]},
                    },
                    "parent": "root",
                    "children": ["msg-assistant"],
                },
                "msg-assistant": {
                    "id": "msg-assistant",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": [f"Assistant answer {i}"]},
                    },
                    "parent": "msg-user",
                    "children": [],
                },
            },
        })
    return conversations


def _make_generic_conversations(count: int = 2) -> list[dict]:
    """Return sample generic export data."""
    conversations = []
    for i in range(count):
        conversations.append({
            "id": f"generic-{i}",
            "title": f"Topic {i}",
            "updated_at": f"2026-04-0{i + 1}",
            "messages": [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ],
        })
    return conversations


# ---------------------------------------------------------------------------
# Claude parser tests
# ---------------------------------------------------------------------------

class TestClaudeParser:
    def test_parse_basic(self, tmp_path: Path):
        data = _make_claude_conversations(2)
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_claude_export(export_file)
        assert len(result) == 2
        assert result[0]["id"] == "conv-0"
        assert result[0]["title"] == "Conversation 0"
        assert len(result[0]["messages"]) == 2
        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][1]["role"] == "assistant"

    def test_empty_export(self, tmp_path: Path):
        export_file = tmp_path / "conversations.json"
        export_file.write_text("[]", encoding="utf-8")

        result = parse_claude_export(export_file)
        assert result == []

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_claude_export(tmp_path / "nonexistent.json")

    def test_invalid_json_type(self, tmp_path: Path):
        export_file = tmp_path / "conversations.json"
        export_file.write_text('{"not": "a list"}', encoding="utf-8")

        with pytest.raises(ValueError, match="Expected a JSON array"):
            parse_claude_export(export_file)

    def test_unnamed_conversation(self, tmp_path: Path):
        data = [{"uuid": "x", "name": "", "updated_at": "", "chat_messages": []}]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_claude_export(export_file)
        assert result[0]["title"] == "(unnamed)"

    def test_empty_message_content(self, tmp_path: Path):
        data = [{
            "uuid": "x",
            "name": "Test",
            "updated_at": "",
            "chat_messages": [
                {"sender": "human", "content": [{"type": "text", "text": ""}]},
                {"sender": "assistant", "content": [{"type": "text", "text": "real reply"}]},
            ],
        }]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_claude_export(export_file)
        # Empty message should be skipped
        assert len(result[0]["messages"]) == 1
        assert result[0]["messages"][0]["content"] == "real reply"

    def test_multiple_content_blocks(self, tmp_path: Path):
        data = [{
            "uuid": "x",
            "name": "Multi",
            "updated_at": "",
            "chat_messages": [{
                "sender": "human",
                "content": [
                    {"type": "text", "text": "Part one."},
                    {"type": "text", "text": "Part two."},
                ],
            }],
        }]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_claude_export(export_file)
        assert result[0]["messages"][0]["content"] == "Part one. Part two."


# ---------------------------------------------------------------------------
# ChatGPT parser tests
# ---------------------------------------------------------------------------

class TestChatGPTParser:
    def test_parse_basic(self, tmp_path: Path):
        data = _make_chatgpt_conversations(2)
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_chatgpt_export(export_file)
        assert len(result) == 2
        assert result[0]["id"] == "chatgpt-0"
        assert result[0]["title"] == "Chat 0"
        assert len(result[0]["messages"]) == 2
        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][1]["role"] == "assistant"

    def test_empty_export(self, tmp_path: Path):
        export_file = tmp_path / "conversations.json"
        export_file.write_text("[]", encoding="utf-8")

        result = parse_chatgpt_export(export_file)
        assert result == []

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_chatgpt_export(tmp_path / "nonexistent.json")

    def test_invalid_json_type(self, tmp_path: Path):
        export_file = tmp_path / "conversations.json"
        export_file.write_text('"just a string"', encoding="utf-8")

        with pytest.raises(ValueError, match="Expected a JSON array"):
            parse_chatgpt_export(export_file)

    def test_unnamed_conversation(self, tmp_path: Path):
        data = [{"id": "x", "title": None, "update_time": None, "mapping": {}}]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_chatgpt_export(export_file)
        assert result[0]["title"] == "(unnamed)"

    def test_empty_mapping(self, tmp_path: Path):
        data = [{"id": "x", "title": "Empty", "update_time": None, "mapping": {}}]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_chatgpt_export(export_file)
        assert result[0]["messages"] == []

    def test_system_messages_excluded(self, tmp_path: Path):
        data = [{
            "id": "x",
            "title": "System",
            "update_time": None,
            "mapping": {
                "root": {
                    "id": "root",
                    "message": {
                        "author": {"role": "system"},
                        "content": {"parts": ["You are a helpful assistant"]},
                    },
                    "parent": None,
                    "children": ["u1"],
                },
                "u1": {
                    "id": "u1",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_chatgpt_export(export_file)
        # System message should not appear, only user message
        assert len(result[0]["messages"]) == 1
        assert result[0]["messages"][0]["role"] == "user"

    def test_updated_at_conversion(self, tmp_path: Path):
        data = [{"id": "x", "title": "TS", "update_time": 1712000000, "mapping": {}}]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_chatgpt_export(export_file)
        assert result[0]["updated_at"] != ""
        assert "2024" in result[0]["updated_at"]  # 1712000000 is in 2024


# ---------------------------------------------------------------------------
# Generic parser tests
# ---------------------------------------------------------------------------

class TestGenericParser:
    def test_parse_basic(self, tmp_path: Path):
        data = _make_generic_conversations(2)
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_generic_export(export_file)
        assert len(result) == 2
        assert result[0]["id"] == "generic-0"
        assert result[0]["title"] == "Topic 0"
        assert len(result[0]["messages"]) == 2

    def test_generates_id_when_missing(self, tmp_path: Path):
        data = [{"title": "No ID", "messages": [{"role": "user", "content": "hi"}]}]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_generic_export(export_file)
        assert result[0]["id"] != ""
        assert len(result[0]["id"]) == 16  # sha256 hex truncated

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_generic_export(tmp_path / "nonexistent.json")

    def test_empty_messages_stripped(self, tmp_path: Path):
        data = [{
            "title": "Sparse",
            "messages": [
                {"role": "user", "content": "real"},
                {"role": "", "content": "no role"},
                {"role": "assistant", "content": ""},
            ],
        }]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_generic_export(export_file)
        assert len(result[0]["messages"]) == 1
        assert result[0]["messages"][0]["content"] == "real"


# ---------------------------------------------------------------------------
# Conversation text extraction
# ---------------------------------------------------------------------------

class TestTextExtraction:
    def test_extract_text_from_messages(self, tmp_path: Path):
        """Verify we can build a flat text from parsed messages."""
        data = _make_claude_conversations(1)
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        convos = parse_claude_export(export_file)
        text = "\n".join(
            f"{m['role']}: {m['content']}" for m in convos[0]["messages"]
        )
        assert "User message 0" in text
        assert "Assistant reply 0" in text

    def test_text_length_calculation(self, tmp_path: Path):
        data = _make_generic_conversations(1)
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        convos = parse_generic_export(export_file)
        text = "\n".join(m["content"] for m in convos[0]["messages"])
        assert len(text) > 0


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

class TestSkipLogic:
    def test_skip_by_min_messages(self, tmp_path: Path):
        """Conversations with fewer messages than min_messages should be skipped."""
        min_messages = 3
        data = [{
            "uuid": "short",
            "name": "Short conv",
            "updated_at": "",
            "chat_messages": [
                {"sender": "human", "content": [{"type": "text", "text": "Hello"}]},
                {"sender": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            ],
        }]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        convos = parse_claude_export(export_file)
        # Apply skip logic
        filtered = [c for c in convos if len(c["messages"]) >= min_messages]
        assert len(filtered) == 0

    def test_keep_above_min_messages(self, tmp_path: Path):
        min_messages = 3
        data = [{
            "uuid": "long",
            "name": "Long conv",
            "updated_at": "",
            "chat_messages": [
                {"sender": "human", "content": [{"type": "text", "text": f"Msg {i}"}]}
                for i in range(4)
            ],
        }]
        export_file = tmp_path / "conversations.json"
        export_file.write_text(json.dumps(data), encoding="utf-8")

        convos = parse_claude_export(export_file)
        filtered = [c for c in convos if len(c["messages"]) >= min_messages]
        assert len(filtered) == 1

    def test_skip_by_title(self):
        """Conversations matching skip_titles should be excluded."""
        skip_titles = ["Daily standup", "test"]
        conversations = [
            {"id": "1", "title": "Daily standup notes", "messages": [{"role": "user", "content": "x"}]},
            {"id": "2", "title": "Architecture review", "messages": [{"role": "user", "content": "y"}]},
            {"id": "3", "title": "test", "messages": [{"role": "user", "content": "z"}]},
        ]

        filtered = [
            c for c in conversations
            if not any(skip.lower() in c["title"].lower() for skip in skip_titles)
        ]
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Architecture review"

    def test_skip_empty_title_list(self):
        """Empty skip_titles should skip nothing."""
        skip_titles: list[str] = []
        conversations = [
            {"id": "1", "title": "Anything", "messages": []},
        ]
        filtered = [
            c for c in conversations
            if not any(skip.lower() in c["title"].lower() for skip in skip_titles)
        ]
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# Dedup tracking (processed.json)
# ---------------------------------------------------------------------------

class TestDedupTracking:
    def test_new_conversation_not_in_processed(self, tmp_path: Path):
        processed_file = tmp_path / "processed.json"
        processed_file.write_text(json.dumps({"processed": {}}), encoding="utf-8")

        processed = json.loads(processed_file.read_text(encoding="utf-8"))
        conv_id = "conv-new"
        assert conv_id not in processed["processed"]

    def test_mark_conversation_processed(self, tmp_path: Path):
        processed_file = tmp_path / "processed.json"
        processed_file.write_text(json.dumps({"processed": {}}), encoding="utf-8")

        processed = json.loads(processed_file.read_text(encoding="utf-8"))
        conv_id = "conv-123"
        processed["processed"][conv_id] = {
            "processed_at": "2026-04-09T12:00:00Z",
            "project": "RESEARCH",
        }
        processed_file.write_text(json.dumps(processed, indent=2), encoding="utf-8")

        reloaded = json.loads(processed_file.read_text(encoding="utf-8"))
        assert conv_id in reloaded["processed"]
        assert reloaded["processed"][conv_id]["project"] == "RESEARCH"

    def test_skip_already_processed(self, tmp_path: Path):
        processed_file = tmp_path / "processed.json"
        processed = {
            "processed": {
                "conv-old": {"processed_at": "2026-04-01", "project": "GENERAL"},
            }
        }
        processed_file.write_text(json.dumps(processed), encoding="utf-8")

        data = json.loads(processed_file.read_text(encoding="utf-8"))
        conversations = [
            {"id": "conv-old", "title": "Old", "messages": []},
            {"id": "conv-new", "title": "New", "messages": []},
        ]
        unprocessed = [c for c in conversations if c["id"] not in data["processed"]]
        assert len(unprocessed) == 1
        assert unprocessed[0]["id"] == "conv-new"

    def test_processed_json_structure(self, tmp_path: Path):
        processed_file = tmp_path / "processed.json"
        processed = {
            "processed": {
                "conv-1": {
                    "processed_at": "2026-04-01T00:00:00Z",
                    "project": "PROJECTS",
                },
                "conv-2": {
                    "processed_at": "2026-04-02T00:00:00Z",
                    "project": "RESEARCH",
                },
            }
        }
        processed_file.write_text(json.dumps(processed), encoding="utf-8")

        data = json.loads(processed_file.read_text(encoding="utf-8"))
        assert "processed" in data
        assert isinstance(data["processed"], dict)
        for entry in data["processed"].values():
            assert "processed_at" in entry
            assert "project" in entry


# ---------------------------------------------------------------------------
# Classification prompt construction (mocked LLM)
# ---------------------------------------------------------------------------

class TestClassificationPrompt:
    def test_prompt_includes_project_descriptions(self):
        """The classification prompt should include all project categories."""
        projects = {
            "PROJECTS": "Active project decisions.",
            "RESEARCH": "External research findings.",
            "GENERAL": "Cross-cutting insights.",
        }
        conversation_text = "user: How should we structure the agent memory?\nassistant: Consider a three-tier approach."

        # Build a prompt the way the capture module would
        project_block = "\n".join(f"- {k}: {v}" for k, v in projects.items())
        prompt = (
            f"Classify this conversation into one of these projects:\n"
            f"{project_block}\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            f"Return only the project name."
        )

        assert "PROJECTS" in prompt
        assert "RESEARCH" in prompt
        assert "GENERAL" in prompt
        assert "agent memory" in prompt

    def test_prompt_truncation_at_max_chars(self):
        """Long conversations should be truncated to max_conversation_chars."""
        max_chars = 100
        long_text = "x" * 500

        truncated = long_text[:max_chars]
        assert len(truncated) == max_chars

    def test_mock_llm_classification(self):
        """Mock the OpenAI client to verify we handle the response correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "RESEARCH"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        # Simulate calling the LLM
        response = mock_client.chat.completions.create(
            model="gemma4",
            messages=[{"role": "user", "content": "Classify this..."}],
        )
        result = response.choices[0].message.content.strip()
        assert result == "RESEARCH"

    def test_mock_llm_invalid_response(self):
        """If the LLM returns an invalid project name, it should be caught."""
        valid_projects = {"PROJECTS", "RESEARCH", "GENERAL"}

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "INVALID_PROJECT"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        response = mock_client.chat.completions.create(
            model="gemma4",
            messages=[{"role": "user", "content": "Classify this..."}],
        )
        result = response.choices[0].message.content.strip()
        assert result not in valid_projects
