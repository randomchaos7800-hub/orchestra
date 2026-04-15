"""
Orchestra capture module -- extract and classify insights from AI conversation exports.

Reads conversation exports (Claude.ai, ChatGPT, or generic JSON), classifies
insights by project category using an LLM, and appends structured entries
to project files.

Configuration loaded from config/config.json (relative to project root).

Usage:
  python capture/extract.py --input /path/to/export/
  python capture/extract.py --input /path/to/conversations.json
  python capture/extract.py --input /path/to/export/ --dry-run
"""

import argparse
import fcntl
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
PROJECTS_DIR = PROJECT_ROOT / "capture" / "projects"
PROCESSED_PATH = PROJECT_ROOT / "capture" / "processed.json"


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> dict | None:
    """Parse LLM output as JSON with fallbacks for common formatting issues.

    1. Strip markdown code fences (```json ... ```)
    2. Try json.loads directly
    3. Fall back to regex extraction of first {...} block
    4. Return None on failure (never crashes)
    """
    # Strip markdown code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Regex fallback: extract first {...} block (greedy, handles nested braces)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    print(f"  WARNING: failed to parse LLM JSON response ({len(raw)} chars)", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Output sanitization
# ---------------------------------------------------------------------------

def _sanitize_content(text: str) -> str:
    """Sanitize LLM-generated content before writing to disk.

    - Strips null bytes
    - Strips control characters (preserves newlines and tabs)
    - Limits length to 10000 chars
    """
    # Strip null bytes
    text = text.replace("\x00", "")
    # Strip control characters except \n (0x0a) and \t (0x09)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Limit length
    if len(text) > 10000:
        text = text[:10000] + "\n[... truncated ...]"
    return text


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------

@contextmanager
def _locked_open(path, mode="a"):
    """Open file with exclusive lock to prevent concurrent write corruption."""
    with open(path, mode, encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# File path validation
# ---------------------------------------------------------------------------

_VALID_PROJECT_RE = re.compile(r"^[A-Z][A-Z0-9_-]*$")


def _validate_project_name(name: str) -> bool:
    """Validate project name to prevent path traversal attacks."""
    if not _VALID_PROJECT_RE.match(name):
        return False
    if "/" in name or "\\" in name or ".." in name or "." in name:
        return False
    return True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    """Load configuration from config/config.json."""
    if not CONFIG_PATH.exists():
        print(f"Error: config not found at {CONFIG_PATH}", file=sys.stderr)
        print("Copy config/config.json.example or create config/config.json", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dedup tracking
# ---------------------------------------------------------------------------

def load_processed() -> set[str]:
    """Load the set of already-processed conversation IDs."""
    if not PROCESSED_PATH.exists():
        return set()
    with open(PROCESSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("processed_ids", []))


def save_processed(ids: set[str]) -> None:
    """Persist the set of processed conversation IDs."""
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump({"processed_ids": sorted(ids)}, f, indent=2)


# ---------------------------------------------------------------------------
# Format detection and parsing
# ---------------------------------------------------------------------------

def detect_and_parse(input_path: Path) -> list[dict[str, Any]]:
    """Auto-detect export format and parse conversations.

    Accepts either a directory (searches for conversations.json inside)
    or a direct path to a conversations.json file.

    Returns normalized conversation list from the appropriate parser.
    """
    if input_path.is_dir():
        conv_file = input_path / "conversations.json"
        if not conv_file.exists():
            print(f"Error: no conversations.json found in {input_path}", file=sys.stderr)
            sys.exit(1)
    elif input_path.is_file():
        conv_file = input_path
    else:
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    with open(conv_file, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list) or len(raw) == 0:
        print(f"Error: {conv_file} is not a non-empty JSON array", file=sys.stderr)
        sys.exit(1)

    sample = raw[0]
    fmt = _detect_format(sample)
    print(f"Detected format: {fmt}")

    # Import parsers here to keep them optional at module level
    if fmt == "claude":
        from capture.parsers.claude import parse_claude_export
        return parse_claude_export(conv_file)
    elif fmt == "chatgpt":
        from capture.parsers.chatgpt import parse_chatgpt_export
        return parse_chatgpt_export(conv_file)
    else:
        from capture.parsers.generic import parse_generic_export
        return parse_generic_export(conv_file)


def _detect_format(sample: dict) -> str:
    """Detect the export format from the first conversation object.

    Returns one of: 'claude', 'chatgpt', 'generic'.
    """
    # Claude exports have chat_messages and uuid
    if "chat_messages" in sample:
        return "claude"

    # ChatGPT exports have mapping (message tree) and create_time
    if "mapping" in sample and "create_time" in sample:
        return "chatgpt"

    return "generic"


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def make_client(config: dict) -> tuple[OpenAI, str]:
    """Return (client, model) -- local LLM first, fallback second."""
    llm_cfg = config.get("llm", {})
    local_url = llm_cfg.get("local_url", "http://127.0.0.1:8081/v1")
    local_model = llm_cfg.get("local_model", "gemma4")
    fallback_url = llm_cfg.get("fallback_url", "")
    fallback_model = llm_cfg.get("fallback_model", "")
    fallback_key_env = llm_cfg.get("fallback_api_key_env", "")

    # Try local
    try:
        client = OpenAI(base_url=local_url, api_key="local")
        resp = client.chat.completions.create(
            model=local_model,
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
            timeout=6,
        )
        if resp.choices:
            print(f"LLM: local ({local_model} at {local_url})")
            return client, local_model
    except Exception:
        pass

    # Fallback
    if fallback_url and fallback_model:
        api_key = os.environ.get(fallback_key_env, "") if fallback_key_env else ""
        if api_key:
            print(f"LLM: fallback ({fallback_model})")
            return OpenAI(base_url=fallback_url, api_key=api_key), fallback_model

    print(
        "Error: local LLM unavailable and fallback not configured "
        f"(need {fallback_key_env} env var)",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Conversation processing
# ---------------------------------------------------------------------------

def get_conversation_text(messages: list[dict], max_chars: int = 12000) -> str:
    """Build a readable text representation of a conversation."""
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")

    full = "\n".join(lines)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n[... truncated ...]"
    return full


def should_skip(conv: dict, config: dict) -> str | None:
    """Check if a conversation should be skipped.

    Returns a reason string if it should be skipped, None otherwise.
    """
    capture_cfg = config.get("capture", {})
    skip_titles = capture_cfg.get("skip_titles", [])
    min_messages = capture_cfg.get("min_messages", 3)

    title = conv.get("title", "")
    for skip in skip_titles:
        if skip.lower() in title.lower():
            return f"title match: {skip}"

    if len(conv.get("messages", [])) < min_messages:
        return f"too few messages ({len(conv.get('messages', []))} < {min_messages})"

    return None


def extract_insights(
    client: OpenAI,
    model: str,
    conv: dict,
    config: dict,
) -> dict | None:
    """Use LLM to classify and extract insights from a conversation."""
    capture_cfg = config.get("capture", {})
    projects = capture_cfg.get("projects", {})
    max_chars = capture_cfg.get("max_conversation_chars", 12000)

    title = conv.get("title", "(unnamed)")
    date = conv.get("updated_at", "")[:10]
    text = get_conversation_text(conv.get("messages", []), max_chars)

    if not text or len(text) < 100:
        return None

    project_list = "\n".join([f"- {p}: {d}" for p, d in projects.items()])
    project_names = list(projects.keys())

    prompt = f"""You are extracting knowledge from an AI conversation for a structured knowledge vault.

Conversation title: {title}
Date: {date}

Project categories:
{project_list}

Conversation:
---
{text}
---

Your task:
1. Classify each extractable insight into the correct project category. One insight can go to multiple categories if genuinely cross-cutting.
2. Extract concrete, specific content -- not "they discussed X" but the actual substance of X.
3. For SPECULATIVE entries: identify what needs to change (tech, API, hardware, capability) for this idea to become buildable. This is the trigger.
4. Skip if purely personal/trivial with no technical, strategic, or intellectual value.

Valid project categories: {json.dumps(project_names)}

Respond in this exact JSON format:
{{
  "projects": ["CATEGORY1", "CATEGORY2"],
  "skip_reason": "reason if nothing worth capturing, empty string otherwise",
  "entries": [
    {{
      "project": "CATEGORY1",
      "title": "Short descriptive title",
      "trigger": "only for SPECULATIVE -- what needs to happen for this to be buildable",
      "content": "The actual insight/decision/idea in clear prose. Be specific. 2-6 sentences."
    }}
  ]
}}

Only include entries worth preserving. Quality over quantity. If nothing is worth capturing, return empty projects array and set skip_reason."""

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        result = _parse_llm_json(raw)
        if result is None:
            print(f"  ERROR parsing LLM response for '{title}'")
        return result
    except Exception as e:
        print(f"  ERROR extracting '{title}': {e}")
        return None


def append_to_project(
    project: str,
    date: str,
    title: str,
    conv_title: str,
    content: str,
    trigger: str = "",
    conv_id: str = "",
) -> None:
    """Append an entry to a project markdown file."""
    # Validate project name to prevent path traversal
    if not _validate_project_name(project):
        print(f"  ERROR: invalid project name '{project}' -- skipping", file=sys.stderr)
        return

    # Sanitize content before writing
    content = _sanitize_content(content)
    title = _sanitize_content(title)

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    project_file = PROJECTS_DIR / f"{project}.md"

    trigger_line = ""
    if trigger and project == "SPECULATIVE":
        trigger_line = f"**Trigger:** {trigger}\n"

    captured_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    meta_line = f"**Source:** {conv_title} | **ID:** {conv_id} | **Captured:** {captured_ts}"

    entry = f"\n## {date} - {title}\n{meta_line}\n{trigger_line}\n{content}\n\n---\n"

    with _locked_open(project_file, "a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestra capture -- extract insights from AI conversation exports"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to export directory (containing conversations.json) or direct path to JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and print results without writing project files",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json (default: config/config.json in project root)",
    )
    args = parser.parse_args()

    # Load config
    global CONFIG_PATH
    if args.config:
        CONFIG_PATH = Path(args.config).resolve()

    config = load_config()

    # Parse input
    input_path = Path(args.input).resolve()
    print(f"Input: {input_path}")

    conversations = detect_and_parse(input_path)
    conversations.sort(key=lambda c: c.get("updated_at", ""))
    print(f"Loaded {len(conversations)} conversations")

    # Dedup
    processed_ids = load_processed()
    capture_cfg = config.get("capture", {})
    project_names = list(capture_cfg.get("projects", {}).keys())
    default_project = project_names[0] if project_names else "GENERAL"

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")
    else:
        # Connect to LLM
        pass

    client, model = make_client(config)
    print("Processing...\n")

    stats = {
        "total": len(conversations),
        "processed": 0,
        "skipped_filter": 0,
        "skipped_dedup": 0,
        "skipped_llm": 0,
        "entries_written": 0,
        "errors": 0,
    }

    newly_processed = set()

    for i, conv in enumerate(conversations):
        title = conv.get("title", "(unnamed)")
        date = conv.get("updated_at", "")[:10]
        conv_id = conv.get("id", "")
        msg_count = len(conv.get("messages", []))

        # Dedup check
        if conv_id and conv_id in processed_ids:
            print(f"[{i+1:03d}] DEDUP {title[:60]}")
            stats["skipped_dedup"] += 1
            continue

        # Filter check
        skip_reason = should_skip(conv, config)
        if skip_reason:
            print(f"[{i+1:03d}] SKIP  {title[:60]} ({skip_reason})")
            stats["skipped_filter"] += 1
            if conv_id:
                newly_processed.add(conv_id)
            continue

        print(f"[{i+1:03d}] ...   {title[:60]} ({msg_count} msgs)")

        result = extract_insights(client, model, conv, config)
        stats["processed"] += 1

        if not result:
            stats["errors"] += 1
            if conv_id:
                newly_processed.add(conv_id)
            continue

        projects = result.get("projects", [])
        if not projects:
            reason = result.get("skip_reason", "no relevant content")
            print(f"       -> skip: {reason}")
            stats["skipped_llm"] += 1
            if conv_id:
                newly_processed.add(conv_id)
            continue

        entries = result.get("entries", [])
        for entry in entries:
            proj = entry.get("project", default_project)
            if proj not in project_names:
                proj = default_project
            entry_title = entry.get("title", title)
            content = entry.get("content", "")
            trigger = entry.get("trigger", "")

            if not content:
                continue

            if args.dry_run:
                tag = f" [trigger: {trigger[:40]}]" if trigger else ""
                print(f"       -> [{proj}] {entry_title[:50]}{tag}")
                print(f"          {content[:120]}...")
            else:
                append_to_project(proj, date, entry_title, title, content, trigger, conv_id=conv_id)
                tag = f" [trigger: {trigger[:40]}]" if trigger else ""
                print(f"       -> [{proj}] {entry_title[:50]}{tag}")

            stats["entries_written"] += 1

        if conv_id:
            newly_processed.add(conv_id)

    # Save dedup state (unless dry run)
    if not args.dry_run and newly_processed:
        all_processed = processed_ids | newly_processed
        save_processed(all_processed)
        print(f"\nDedup: {len(newly_processed)} new IDs tracked ({len(all_processed)} total)")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total conversations: {stats['total']}")
    print(f"Sent to LLM:        {stats['processed']}")
    print(f"Skipped (filter):    {stats['skipped_filter']}")
    print(f"Skipped (dedup):     {stats['skipped_dedup']}")
    print(f"Skipped (LLM):       {stats['skipped_llm']}")
    print(f"Entries written:     {stats['entries_written']}")
    print(f"Errors:              {stats['errors']}")
    if args.dry_run:
        print("[DRY RUN] No files were modified.")


if __name__ == "__main__":
    main()
