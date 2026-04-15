"""
Generic parser for simple JSON conversation exports.

Expected format: a JSON array of objects, each with:
- title: str
- messages: list of {role: str, content: str}
- id: str (optional, generated if missing)
- updated_at: str (optional)
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def parse_generic_export(conversations_json_path: str | Path) -> list[dict[str, Any]]:
    """Parse a generic conversations JSON file into a normalized format.

    This is essentially a pass-through parser for data already in the
    normalized format. It validates structure and fills in missing fields.

    Args:
        conversations_json_path: Path to a JSON file containing a list of conversations.

    Returns:
        List of conversation dicts, each with keys:
            - id: str
            - title: str
            - updated_at: str
            - messages: list[dict] with keys 'role' and 'content'
    """
    path = Path(conversations_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Generic export not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw).__name__}")

    conversations = []
    for i, conv in enumerate(raw):
        title = (conv.get("title", "") or "").strip() or "(unnamed)"

        # Generate a stable ID from title + index if not provided
        conv_id = conv.get("id", "")
        if not conv_id:
            conv_id = hashlib.sha256(f"{title}-{i}".encode()).hexdigest()[:16]

        updated_at = conv.get("updated_at", "")

        messages = []
        for msg in conv.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                messages.append({"role": role, "content": content.strip()})

        conversations.append({
            "id": conv_id,
            "title": title,
            "updated_at": updated_at,
            "messages": messages,
        })

    return conversations
