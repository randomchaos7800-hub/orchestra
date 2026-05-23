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


def _conversation_fingerprint(title: str, updated_at: str, messages: list[dict[str, str]]) -> str:
    """Build a stable content fingerprint for conversations without IDs."""
    payload = {
        "title": title,
        "updated_at": updated_at,
        "messages": [
            {
                "role": (msg.get("role", "") or "").strip(),
                "content": (msg.get("content", "") or "").strip(),
            }
            for msg in messages
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
    for conv in raw:
        title = (conv.get("title", "") or "").strip() or "(unnamed)"
        updated_at = conv.get("updated_at", "")

        messages = []
        for msg in conv.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                messages.append({"role": role, "content": content.strip()})

        conv_id = (conv.get("id", "") or "").strip()
        if not conv_id:
            conv_id = _conversation_fingerprint(title, updated_at, messages)

        conversations.append({
            "id": conv_id,
            "title": title,
            "updated_at": updated_at,
            "messages": messages,
        })

    return conversations
