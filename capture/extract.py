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
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.common import (
    locked_open, load_config, make_llm_client,
    parse_llm_json, sanitize_content, git_auto_commit,
)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = PROJECT_ROOT / "capture" / "projects"
PROCESSED_PATH = PROJECT_ROOT / "capture" / "processed.json"

# Project name validation: uppercase letters, digits, hyphens, underscores
_VALID_PROJECT_RE = re.compile(r"^[A-Z][A-Z0-9_-]*$")


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
    """Auto-detect export format and parse conversations."""
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

    # Claude exports have chat_messages
    if "chat_messages" in sample:
        fmt = "claude"
        from capture.parsers.claude import parse_claude_export
        parser = parse_claude_export
    # ChatGPT exports have mapping + create_time
    elif "mapping" in sample and "create_time" in sample:
        fmt = "chatgpt"
        from capture.parsers.chatgpt import parse_chatgpt_export
        parser = parse_chatgpt_export
    else:
        fmt = "generic"
        from capture.parsers.generic import parse_generic_export
        parser = parse_generic_export

    print(f"Detected format: {fmt}")
    return parser(conv_file)


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
    """Check if a conversation should be skipped. Returns reason or None."""
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


def extract_insights(client, model: str, conv: dict, config: dict) -> dict | None:
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
1. Classify each extractable insight into the correct project category.
2. Extract concrete, specific content -- the actual substance, not "they discussed X".
3. For SPECULATIVE entries: identify the trigger (what needs to change for this to be buildable).
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
      "trigger": "only for SPECULATIVE",
      "content": "The actual insight in clear prose. 2-6 sentences."
    }}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        result = parse_llm_json(raw)
        if result is None:
            print(f"  ERROR parsing LLM response for '{title}'")
        return result
    except Exception as e:
        print(f"  ERROR extracting '{title}': {e}")
        return None


def append_to_project(project: str, date: str, title: str, conv_title: str,
                      content: str, trigger: str = "", conv_id: str = "") -> None:
    """Append an entry to a project markdown file."""
    if not _VALID_PROJECT_RE.match(project):
        print(f"  ERROR: invalid project name '{project}' -- skipping", file=sys.stderr)
        return

    content = sanitize_content(content)
    title = sanitize_content(title)

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    project_file = PROJECTS_DIR / f"{project}.md"

    trigger_line = f"**Trigger:** {trigger}\n" if trigger and project == "SPECULATIVE" else ""
    captured_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    meta_line = f"**Source:** {conv_title} | **ID:** {conv_id} | **Captured:** {captured_ts}"
    entry = f"\n## {date} - {title}\n{meta_line}\n{trigger_line}\n{content}\n\n---\n"

    with locked_open(project_file, "a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestra capture -- extract insights from AI conversation exports"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--git", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve() if args.config else None
    config = load_config(config_path)

    input_path = Path(args.input).resolve()
    print(f"Input: {input_path}")

    conversations = detect_and_parse(input_path)
    conversations.sort(key=lambda c: c.get("updated_at", ""))
    print(f"Loaded {len(conversations)} conversations")

    processed_ids = load_processed()
    capture_cfg = config.get("capture", {})
    project_names = list(capture_cfg.get("projects", {}).keys())
    default_project = project_names[0] if project_names else "GENERAL"

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    client, model, _ = make_llm_client(config)
    print("Processing...\n")

    stats = {
        "total": len(conversations), "processed": 0, "skipped_filter": 0,
        "skipped_dedup": 0, "skipped_llm": 0, "entries_written": 0, "errors": 0,
    }
    newly_processed: set[str] = set()

    for i, conv in enumerate(conversations):
        title = conv.get("title", "(unnamed)")
        date = conv.get("updated_at", "")[:10]
        conv_id = conv.get("id", "")
        msg_count = len(conv.get("messages", []))

        if conv_id and conv_id in processed_ids:
            print(f"[{i+1:03d}] DEDUP {title[:60]}")
            stats["skipped_dedup"] += 1
            continue

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
            print(f"       -> skip: {result.get('skip_reason', 'no relevant content')}")
            stats["skipped_llm"] += 1
            if conv_id:
                newly_processed.add(conv_id)
            continue

        for entry in result.get("entries", []):
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

    if not args.dry_run and newly_processed:
        all_processed = processed_ids | newly_processed
        save_processed(all_processed)
        print(f"\nDedup: {len(newly_processed)} new IDs tracked ({len(all_processed)} total)")

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

    if args.git and not args.dry_run and stats["entries_written"] > 0:
        msg = f"capture: {stats['entries_written']} entries from {stats['processed']} conversations"
        if git_auto_commit(["projects/", "capture/processed.json"], msg, PROJECT_ROOT):
            print(f"Git: committed ({msg})")
        else:
            print("Git: nothing to commit or git not available")


if __name__ == "__main__":
    main()
