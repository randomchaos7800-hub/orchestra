"""
Parser for ChatGPT conversation exports.

ChatGPT exports contain a conversations.json file where each conversation has:
- title: conversation title
- create_time: unix timestamp
- update_time: unix timestamp
- mapping: dict of node_id -> node objects forming a message tree
  Each node has: id, message (with author.role, content.parts), parent, children
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _walk_message_tree(mapping: dict[str, Any]) -> list[dict[str, str]]:
    """Walk the ChatGPT message tree in order and extract messages.

    ChatGPT stores messages as a tree (mapping of node_id -> node).
    We find the root (no parent or parent not in mapping), then walk
    children depth-first to reconstruct the linear conversation.
    """
    if not mapping:
        return []

    # Find root node(s) -- nodes whose parent is None or not in mapping
    roots = []
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            roots.append(node_id)

    if not roots:
        return []

    # Walk the tree from root, following first child at each level
    messages = []
    current = roots[0]

    while current and current in mapping:
        node = mapping[current]
        msg = node.get("message")

        if msg is not None:
            author = msg.get("author", {}).get("role", "")
            content = msg.get("content", {})
            parts = content.get("parts", [])

            # Only keep user and assistant messages
            if author in ("user", "assistant"):
                text_parts = []
                for part in parts:
                    if isinstance(part, str):
                        stripped = part.strip()
                        if stripped:
                            text_parts.append(stripped)
                text = " ".join(text_parts).strip()
                if text:
                    messages.append({"role": author, "content": text})

        # Follow the first child (main conversation thread)
        children = node.get("children", [])
        current = children[0] if children else None

    return messages


def _unix_to_iso(ts: float | int | None) -> str:
    """Convert a Unix timestamp to an ISO 8601 string, or return empty string."""
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return ""


def parse_chatgpt_export(conversations_json_path: str | Path) -> list[dict[str, Any]]:
    """Parse a ChatGPT conversations.json export into a normalized format.

    Args:
        conversations_json_path: Path to the conversations.json file from a ChatGPT export.

    Returns:
        List of conversation dicts, each with keys:
            - id: str (conversation ID)
            - title: str
            - updated_at: str (ISO timestamp)
            - messages: list[dict] with keys 'role' and 'content'
    """
    path = Path(conversations_json_path)
    if not path.exists():
        raise FileNotFoundError(f"ChatGPT export not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw).__name__}")

    conversations = []
    for conv in raw:
        conv_id = conv.get("id", conv.get("conversation_id", ""))
        title = (conv.get("title", "") or "").strip() or "(unnamed)"
        updated_at = _unix_to_iso(conv.get("update_time") or conv.get("create_time"))

        mapping = conv.get("mapping", {})
        messages = _walk_message_tree(mapping)

        conversations.append({
            "id": conv_id,
            "title": title,
            "updated_at": updated_at,
            "messages": messages,
        })

    return conversations
