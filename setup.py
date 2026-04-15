#!/usr/bin/env python3
"""
Orchestra Setup Wizard

Creates the directory structure and initial configuration for a new
Orchestra knowledge base. Run once to initialize, safe to re-run
(will not overwrite existing files).

Usage:
    python setup.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def create_directories() -> None:
    """Create the standard Orchestra directory structure."""
    dirs = [
        "raw",
        "raw/manual",
        "wiki",
        "wiki/concepts",
        "wiki/entities",
        "wiki/events",
        "wiki/research",
        "wiki/meta",
        "projects",
        "config",
    ]
    for d in dirs:
        path = BASE_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [ok] {d}/")


def copy_example_config() -> bool:
    """Copy config.example.json to config.json if it doesn't exist.

    Returns True if a new config was created, False if one already existed.
    """
    example = BASE_DIR / "config" / "config.example.json"
    target = BASE_DIR / "config" / "config.json"

    if target.exists():
        print("  [skip] config/config.json already exists")
        return False

    if not example.exists():
        print("  [warn] config/config.example.json not found, creating default config")
        config = {
            "llm": {
                "local_url": "http://127.0.0.1:8081/v1",
                "local_model": "gemma4",
                "local_max_tokens": 6000,
                "fallback_url": "https://openrouter.ai/api/v1",
                "fallback_model": "z-ai/glm-4.7-flash",
                "fallback_api_key_env": "OPENROUTER_API_KEY",
            },
            "capture": {
                "projects": {
                    "PROJECTS": "Concrete decisions and architecture for active projects.",
                    "RESEARCH": "External research, papers, model releases, findings.",
                    "STRATEGY": "Direction, planning, positioning, revenue strategy.",
                    "GENERAL": "Cross-cutting insights that don't fit a specific project.",
                    "SPECULATIVE": "Ideas ahead of current tech -- half-formed, not yet buildable.",
                },
                "skip_titles": [],
                "min_messages": 3,
                "max_conversation_chars": 12000,
            },
            "wiki": {
                "sections": ["concepts", "entities", "events", "research", "tools"],
                "stale_days": 30,
            },
        }
    else:
        with open(example, encoding="utf-8") as f:
            config = json.load(f)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print("  [ok] config/config.json created from example")
    return True


def create_wiki_files() -> None:
    """Create empty wiki/_index.md and wiki/_sources.json if they don't exist."""
    index_path = BASE_DIR / "wiki" / "_index.md"
    sources_path = BASE_DIR / "wiki" / "_sources.json"

    if not index_path.exists():
        index_path.write_text(
            "# Wiki Index\n\nArticle count: 0\n\n"
            "## Concepts\n\n## Entities\n\n## Events\n\n## Research\n",
            encoding="utf-8",
        )
        print("  [ok] wiki/_index.md created")
    else:
        print("  [skip] wiki/_index.md already exists")

    if not sources_path.exists():
        sources_path.write_text(
            json.dumps({"processed": {}}, indent=2) + "\n",
            encoding="utf-8",
        )
        print("  [ok] wiki/_sources.json created")
    else:
        print("  [skip] wiki/_sources.json already exists")


def create_gitignore() -> None:
    """Create .gitignore if it doesn't exist."""
    gitignore_path = BASE_DIR / ".gitignore"
    if gitignore_path.exists():
        print("  [skip] .gitignore already exists")
        return

    gitignore_path.write_text(
        "config/config.json\n"
        "*.pyc\n"
        "__pycache__/\n"
        ".env\n"
        ".venv/\n"
        "venv/\n"
        "*.egg-info/\n"
        "dist/\n"
        "build/\n",
        encoding="utf-8",
    )
    print("  [ok] .gitignore created")


def prompt_llm_config() -> None:
    """Prompt the user for LLM endpoint configuration and update config.json."""
    config_path = BASE_DIR / "config" / "config.json"
    if not config_path.exists():
        print("  [warn] config/config.json not found, skipping LLM setup")
        return

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    print("\n--- LLM Configuration ---")
    print("Orchestra can use a local LLM server (e.g., llama.cpp, Ollama) or a")
    print("remote API (e.g., OpenRouter). You can configure both; the local server")
    print("is tried first and the remote API is used as fallback.\n")

    # Local endpoint
    current_local = config.get("llm", {}).get("local_url", "http://127.0.0.1:8081/v1")
    local_url = input(f"Local LLM URL [{current_local}]: ").strip()
    if local_url:
        config.setdefault("llm", {})["local_url"] = local_url
    else:
        print(f"  Using default: {current_local}")

    current_model = config.get("llm", {}).get("local_model", "gemma4")
    local_model = input(f"Local model name [{current_model}]: ").strip()
    if local_model:
        config["llm"]["local_model"] = local_model

    # Fallback endpoint
    print()
    current_fallback = config.get("llm", {}).get("fallback_url", "https://openrouter.ai/api/v1")
    fallback_url = input(f"Fallback API URL [{current_fallback}]: ").strip()
    if fallback_url:
        config.setdefault("llm", {})["fallback_url"] = fallback_url

    current_key_env = config.get("llm", {}).get("fallback_api_key_env", "OPENROUTER_API_KEY")
    key_env = input(f"Env var for fallback API key [{current_key_env}]: ").strip()
    if key_env:
        config["llm"]["fallback_api_key_env"] = key_env

    # Write updated config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print("\n  [ok] config/config.json updated with LLM settings")


def print_next_steps() -> None:
    """Print guidance for what to do after setup."""
    print("\n" + "=" * 60)
    print("  Setup complete.")
    print("=" * 60)
    print()
    print("Next steps:")
    print()
    print("  1. Export your conversations from Claude or ChatGPT")
    print("     and place the JSON files in raw/")
    print()
    print("  2. Run the capture pipeline to classify conversations:")
    print("     python capture/extract.py --input /path/to/export/")
    print()
    print("  3. Run the wiki compiler to generate articles:")
    print("     python tools/compile.py")
    print()
    print("  4. If using a remote API, set the environment variable:")
    api_key_env = "OPENROUTER_API_KEY"
    try:
        with open(BASE_DIR / "config" / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
            api_key_env = cfg.get("llm", {}).get("fallback_api_key_env", api_key_env)
    except Exception:
        pass
    print(f"     export {api_key_env}=your-key-here")
    print()
    print("  5. Run tests to verify everything:")
    print("     pytest tests/")
    print()


def main() -> None:
    print("Orchestra Setup Wizard")
    print("=" * 60)

    print("\nCreating directory structure...")
    create_directories()

    print("\nSetting up configuration...")
    copy_example_config()

    print("\nCreating wiki files...")
    create_wiki_files()

    print("\nChecking .gitignore...")
    create_gitignore()

    # Only prompt interactively if stdin is a terminal
    if sys.stdin.isatty():
        prompt_llm_config()
    else:
        print("\n  [skip] Non-interactive mode, skipping LLM configuration prompts")

    print_next_steps()


if __name__ == "__main__":
    main()
