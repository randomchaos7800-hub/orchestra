"""
Parser for Claude.ai conversation exports.

Claude exports contain a conversations.json file where each conversation has:
- uuid: unique identifier
- name: conversation title
- updated_at: ISO timestamp
- chat_messages: list of message objects with sender and content blocks
"""

import json
from pathlib import Path
from typing import Any


def parse_claude_export(conversations_json_path: str | Path) -> list[dict[str, Any]]:
    """Parse a Claude.ai conversations.json export into a normalized format.

    Args:
        conversations_json_path: Path to the conversations.json file from a Claude export.

    Returns:
        List of conversation dicts, each with keys:
            - id: str (conversation UUID)
            - title: str
            - updated_at: str (ISO timestamp)
            - messages: list[dict] with keys 'role' and 'content'
    """
    path = Path(conversations_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Claude export not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw).__name__}")

    conversations = []
    for conv in raw:
        conv_id = conv.get("uuid", conv.get("id", ""))
        title = (conv.get("name", "") or "").strip() or "(unnamed)"
        updated_at = conv.get("updated_at", "")

        messages = []
        for msg in conv.get("chat_messages", []):
            sender = msg.get("sender", "unknown")
            role = "user" if sender == "human" else "assistant"

            text_parts = []
            for block in msg.get("content", []):
                if block.get("type") == "text":
                    t = block.get("text", "").strip()
                    if t:
                        text_parts.append(t)

            content = " ".join(text_parts).strip()
            if content:
                messages.append({"role": role, "content": content})

        conversations.append({
            "id": conv_id,
            "title": title,
            "updated_at": updated_at,
            "messages": messages,
        })

    return conversations
