#!/usr/bin/env python3
"""
Wiki Health Check -- tools/healthcheck.py

Audits wiki/ for structural issues and generates suggestions.
Writes results to wiki/meta/.

LLM configuration loaded from config/config.json.

Checks:
  - Orphaned articles (no backlinks pointing to them)
  - Broken wikilinks (links to articles that don't exist)
  - Stale articles (not updated in N days, default 30)
  - Missing articles (frequently mentioned slugs that don't have files)

Usage:
  python3 tools/healthcheck.py
  python3 tools/healthcheck.py --stale-days 60
  python3 tools/healthcheck.py --suggest    # LLM-assisted gap analysis
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

KB_ROOT = Path(__file__).parent.parent
WIKI_DIR = KB_ROOT / "wiki"
CONFIG_DIR = KB_ROOT / "config"

ORPHANS_FILE = WIKI_DIR / "meta" / "orphans.md"
STALE_FILE = WIKI_DIR / "meta" / "stale.md"
SUGGESTIONS_FILE = WIKI_DIR / "meta" / "suggestions.md"

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def _load_llm_config() -> dict:
    """Load LLM settings from config/config.json."""
    config_path = CONFIG_DIR / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("llm", {})


def _all_articles() -> dict[str, Path]:
    """Return {slug: path} for all wiki articles."""
    articles = {}
    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        if "meta" in md_file.parts:
            continue
        articles[md_file.stem] = md_file
    return articles


def _find_all_wikilinks() -> dict[str, list[str]]:
    """Return {slug: [files_that_link_to_it]} -- backlink map."""
    backlinks: dict[str, list[str]] = {}
    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
            rel = str(md_file.relative_to(WIKI_DIR))
            for m in WIKILINK_PATTERN.finditer(text):
                target = m.group(1).strip()
                backlinks.setdefault(target, []).append(rel)
        except Exception:
            pass
    return backlinks


def check_orphans(articles: dict[str, Path], backlinks: dict[str, list[str]]) -> list[str]:
    """Articles with no backlinks pointing to them."""
    orphans = []
    for slug, path in articles.items():
        if slug not in backlinks:
            orphans.append(str(path.relative_to(WIKI_DIR)))
    return sorted(orphans)


def check_broken_links(articles: dict[str, Path]) -> list[tuple[str, str]]:
    """Wikilinks that point to non-existent articles. Returns [(source_file, broken_link)]."""
    broken = []
    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
            rel = str(md_file.relative_to(WIKI_DIR))
            for m in WIKILINK_PATTERN.finditer(text):
                target = m.group(1).strip()
                if target not in articles:
                    broken.append((rel, target))
        except Exception:
            pass
    return sorted(broken)


def check_stale(articles: dict[str, Path], stale_days: int = 30) -> list[tuple[str, str]]:
    """Articles not updated in stale_days days. Returns [(file, last_updated)]."""
    cutoff = datetime.now() - timedelta(days=stale_days)
    stale = []
    for slug, path in articles.items():
        try:
            text = path.read_text(encoding="utf-8")
            fm_match = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
            if fm_match:
                updated_match = re.search(r"updated:\s*(\d{4}-\d{2}-\d{2})", fm_match.group(1))
                if updated_match:
                    updated = datetime.fromisoformat(updated_match.group(1))
                    if updated < cutoff:
                        stale.append((str(path.relative_to(WIKI_DIR)), updated_match.group(1)))
                    continue
            # Fall back to file mtime
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if mtime < cutoff:
                stale.append((str(path.relative_to(WIKI_DIR)), mtime.strftime("%Y-%m-%d")))
        except Exception:
            pass
    return sorted(stale, key=lambda x: x[1])


def find_unwritten_concepts(articles: dict[str, Path]) -> list[tuple[str, int]]:
    """Slugs mentioned via [[wikilink]] but without their own article. Returns [(slug, mention_count)]."""
    mention_counts: dict[str, int] = {}
    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
            for m in WIKILINK_PATTERN.finditer(text):
                target = m.group(1).strip()
                if target not in articles:
                    mention_counts[target] = mention_counts.get(target, 0) + 1
        except Exception:
            pass
    return sorted(mention_counts.items(), key=lambda x: -x[1])


def _llm_suggestions(articles: dict[str, Path], orphans: list[str], unwritten: list[tuple[str, int]]) -> str:
    """Ask LLM to suggest new articles and connections based on current wiki state."""
    try:
        import httpx
        from openai import OpenAI

        llm_cfg = _load_llm_config()
        local_url = llm_cfg.get("local_url", "http://127.0.0.1:8081/v1")
        local_model = llm_cfg.get("local_model", "gemma4")

        client = model = None
        try:
            r = httpx.get(local_url.replace("/v1", "/health"), timeout=2)
            if r.status_code == 200:
                client = OpenAI(base_url=local_url, api_key="local")
                model = local_model
        except Exception:
            pass

        if not client:
            fallback_url = llm_cfg.get("fallback_url", "")
            fallback_model = llm_cfg.get("fallback_model", "")
            api_key_env = llm_cfg.get("fallback_api_key_env", "OPENROUTER_API_KEY")
            api_key = os.environ.get(api_key_env, "")
            if fallback_url and api_key:
                client = OpenAI(base_url=fallback_url, api_key=api_key)
                model = fallback_model

        if not client:
            return "*(LLM unavailable -- skipping suggestions)*"

        index_file = WIKI_DIR / "_index.md"
        index_text = index_file.read_text(encoding="utf-8")[:3000] if index_file.exists() else "(empty)"

        orphan_list = "\n".join(f"- {o}" for o in orphans[:10]) or "(none)"
        unwritten_list = "\n".join(f"- [[{s}]] ({c} mentions)" for s, c in unwritten[:10]) or "(none)"

        prompt = f"""You are reviewing a knowledge base wiki for structural completeness.

Current wiki index:
{index_text}

Orphaned articles (no backlinks):
{orphan_list}

Mentioned but not written:
{unwritten_list}

Suggest:
1. 3-5 new articles worth creating (based on gaps and orphan context)
2. 3-5 connection pairs that should have backlinks but don't

Return as clean markdown, no preamble."""

        response = client.chat.completions.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    except Exception as e:
        return f"*(LLM suggestion failed: {e})*"


def _write_meta(path: Path, title: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(
        f"# {title}\n_Last updated: {now}_\n\n{content}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Wiki health check")
    parser.add_argument("--stale-days", type=int, default=30, help="Days before article is considered stale (default: 30)")
    parser.add_argument("--suggest", action="store_true", help="Run LLM-assisted gap analysis for suggestions")
    args = parser.parse_args()

    articles = _all_articles()
    if not articles:
        print("Wiki is empty. Run compile.py first.")
        return

    print(f"Checking {len(articles)} articles...\n")

    # Backlinks
    backlinks = _find_all_wikilinks()

    # Orphans
    orphans = check_orphans(articles, backlinks)
    print(f"Orphaned articles: {len(orphans)}")
    orphan_content = "\n".join(f"- [{o}]({o})" for o in orphans) or "*(none -- all articles have backlinks)*"
    _write_meta(ORPHANS_FILE, "Orphaned Articles", orphan_content)

    # Broken links
    broken = check_broken_links(articles)
    print(f"Broken wikilinks: {len(broken)}")
    if broken:
        for source, target in broken[:10]:
            print(f"  {source} -> [[{target}]] (missing)")

    # Stale
    stale = check_stale(articles, stale_days=args.stale_days)
    print(f"Stale articles (>{args.stale_days}d): {len(stale)}")
    stale_content = "\n".join(f"- [{f}]({f}) -- last updated {d}" for f, d in stale) or f"*(none -- all articles updated within {args.stale_days} days)*"
    _write_meta(STALE_FILE, f"Stale Articles (>{args.stale_days} days)", stale_content)

    # Unwritten concepts
    unwritten = find_unwritten_concepts(articles)
    print(f"Mentioned but unwritten: {len(unwritten)}")
    if unwritten:
        for slug, count in unwritten[:5]:
            print(f"  [[{slug}]] -- {count} mention(s)")

    # Suggestions
    if args.suggest:
        print("\nRunning LLM gap analysis...")
        suggestions = _llm_suggestions(articles, orphans, unwritten)
        _write_meta(SUGGESTIONS_FILE, "Suggested Articles and Connections", suggestions)
        print(f"Suggestions written to wiki/meta/suggestions.md")
    else:
        unwritten_content = "\n".join(f"- `[[{s}]]` -- {c} mention(s)" for s, c in unwritten[:20]) or "*(none)*"
        _write_meta(SUGGESTIONS_FILE, "Mentioned But Not Written",
                    "Run `healthcheck.py --suggest` for LLM-assisted gap analysis.\n\n## Frequently Mentioned (no article)\n\n" + unwritten_content)

    print(f"\nHealth check complete. Results written to wiki/meta/")


if __name__ == "__main__":
    main()
