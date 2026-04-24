#!/usr/bin/env python3
"""
Orchestra Wiki Compiler -- tools/compile.py

Scans raw/ for unprocessed markdown files.
For each new file: calls LLM to extract knowledge, creates/updates wiki/ articles,
updates _index.md and _sources.json.

Usage:
  python3 tools/compile.py                   # process all new raw files
  python3 tools/compile.py --dry-run         # show what would change, no writes
  python3 tools/compile.py --force           # reprocess all files, not just new ones
  python3 tools/compile.py --source PATH     # compile a specific raw file only
  python3 tools/compile.py --verbose         # show LLM prompts and responses
  python3 tools/compile.py --recompile-stale # recompile articles whose sources are newer
"""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import (
    KB_ROOT, RAW_DIR, WIKI_DIR, CONFIG_DIR, INDEX_FILE,
    locked_open, make_llm_client, llm_call, load_config,
    get_wiki_sections,
    load_sources, save_sources,
    parse_frontmatter, inject_metadata,
    extract_typed_links, parse_llm_json, git_auto_commit,
    load_index, read_existing_summaries, rebuild_index,
    inject_reciprocal_backlinks, staleness_check,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -- Source tracking -----------------------------------------------------------

def _mark_processed(sources: dict, raw_path: Path, articles_touched: list[str]) -> None:
    rel = str(raw_path.relative_to(KB_ROOT))
    sources["processed"][rel] = {
        "processed_at": datetime.now().isoformat(),
        "articles": articles_touched,
    }


# -- Concept expansion --------------------------------------------------------

def _expand_concepts(client, model: str, concepts: list[str]) -> list[str]:
    """Generate alternate phrasings for cross-reference matching."""
    if not concepts:
        return []
    prompt = (
        f"Generate 3-5 alternate phrasings or semantically related terms for each of "
        f"these concepts: {', '.join(concepts)}\n"
        f"Return ONLY a flat comma-separated list of terms."
    )
    try:
        raw = llm_call(client, model, "You are a semantic expansion tool.", prompt, max_tokens=300)
        terms = [t.strip().lower() for t in raw.split(",") if t.strip()]
        original_lower = {c.lower() for c in concepts}
        return [t for t in dict.fromkeys(terms) if t not in original_lower]
    except Exception:
        return []


# -- Backlink context gathering -----------------------------------------------

def _build_backlink_index(wiki_dir: Path) -> dict[str, list[str]]:
    """Scan all wiki articles and build {slug -> [slugs that link to it]}."""
    backlinks: dict[str, list[str]] = {}
    for section in get_wiki_sections(wiki_dir):
        section_dir = wiki_dir / section
        if not section_dir.exists():
            continue
        for md_file in section_dir.rglob("*.md"):
            source_slug = md_file.stem
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            fm = parse_frontmatter(text)
            links_fm = fm.get("links", [])

            targets = []
            if links_fm and isinstance(links_fm, list):
                for link in links_fm:
                    if isinstance(link, dict):
                        target = link.get("target", "")
                        t = Path(target).stem if target else ""
                        if t:
                            targets.append(t)
            else:
                targets = [l["target"] for l in extract_typed_links(text)]

            for t in targets:
                backlinks.setdefault(t, [])
                if source_slug not in backlinks[t]:
                    backlinks[t].append(source_slug)

    return backlinks


def _gather_backlink_context(slug: str, backlink_index: dict, wiki_dir: Path, max_articles: int = 10) -> str:
    """Walk backlinks up to depth 2 for related context."""
    depth1 = backlink_index.get(slug, [])
    depth2 = []
    for d1 in depth1:
        for d2 in backlink_index.get(d1, []):
            if d2 != slug and d2 not in depth1 and d2 not in depth2:
                depth2.append(d2)

    candidates = [(s, 1) for s in depth1] + [(s, 2) for s in depth2]
    candidates = candidates[:max_articles]

    if not candidates:
        return ""

    lines = ["## Related Context",
             "_Articles linking to this one -- use for cross-reference enrichment only._", ""]

    for s, depth in candidates:
        article_path = None
        for section in get_wiki_sections(wiki_dir):
            p = wiki_dir / section / f"{s}.md"
            if p.exists():
                article_path = p
                break
        if not article_path:
            continue

        try:
            text = article_path.read_text(encoding="utf-8")
        except Exception:
            continue

        body = re.sub(r"^---.*?---\n", "", text, flags=re.DOTALL).strip()
        body = re.sub(r"^#[^\n]+\n", "", body).strip()
        body = re.sub(r"^\*\*[^\n]+\*\*\n?", "", body).strip()
        body = re.sub(r"^##[^\n]+\n", "", body).strip()

        sentences = re.split(r"(?<=[.!?])\s+", body)
        excerpt = " ".join(sentences[:3]).strip()
        if len(excerpt) > 400:
            excerpt = excerpt[:397] + "..."

        lines.append(f"- **[[{s}]]** (depth {depth}): {excerpt}")

    return "\n".join(lines)


def _write_stale_report(stale: list[dict], wiki_dir: Path) -> None:
    """Write wiki/meta/stale.md."""
    meta_dir = wiki_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [f"# Stale Articles", f"_Last updated: {now}_", ""]
    if not stale:
        lines.append("*(none -- all compiled articles are current)*")
    else:
        lines.append(f"{len(stale)} article(s) have newer source material:\n")
        for s in stale:
            lines.append(
                f"- **{s['article']}** -- compiled {s['last_compiled']}, "
                f"source {s['newest_source_date']} ({s['stale_days']}d stale) "
                f"via `{s['newest_source']}`"
            )

    with locked_open(meta_dir / "stale.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Stale report: {len(stale)} stale article(s)")


# -- Compile a single raw file ------------------------------------------------

def compile_file(
    raw_path: Path, client, model: str,
    max_tokens: int = 6000, dry_run: bool = False, verbose: bool = False,
) -> list[str]:
    """Two-pass LLM compile: JSON plan then markdown content per article."""
    compile_rules = (CONFIG_DIR / "compile-rules.md").read_text(encoding="utf-8")
    wiki_style = (CONFIG_DIR / "wiki-style.md").read_text(encoding="utf-8")
    raw_content = raw_path.read_text(encoding="utf-8")

    index = load_index()
    article_titles = "\n".join(
        f"- [{d['title']}]({d['path']})" for _, d in sorted(index.items())
    ) or "(wiki is empty -- create new articles freely)"

    # -- Pass 1: Article plan --
    plan_system = f"""{compile_rules}

You must respond with ONLY a JSON object. No markdown fences, no prose.
The JSON must have this exact structure:
{{
  "articles": [
    {{
      "path": "concepts/slug-here.md",
      "action": "create",
      "title": "Human Title",
      "summary": "One sentence summary.",
      "tags": ["tag1", "tag2"],
      "sections": ["Overview", "Key Points", "Connections"],
      "core_concepts": ["concept1", "concept2", "concept3"]
    }}
  ]
}}
Or if nothing worth writing: {{"articles": [], "skipped_reason": "reason here"}}
All string values must be single-line. No newlines inside any string value.
core_concepts: 3-5 key terms central to this article (used for cross-reference expansion)."""

    plan_user = f"""Plan which wiki articles to create or update based on this raw source.

RAW SOURCE FILE: {raw_path.relative_to(KB_ROOT)}
---
{raw_content[:5000]}
---

EXISTING WIKI ARTICLES:
{article_titles}

Today's date: {datetime.now().strftime("%Y-%m-%d")}

Return the JSON plan only."""

    if verbose:
        logger.info(f"=== PASS 1 PROMPT ({len(plan_user)} chars) ===\n{plan_user[:300]}...")

    try:
        plan_response = llm_call(client, model, plan_system, plan_user, max_tokens=2000)
    except Exception as e:
        logger.error(f"Pass 1 failed for {raw_path.name}: {e}")
        return []

    if verbose:
        logger.info(f"=== PASS 1 RESPONSE ===\n{plan_response[:500]}")

    plan_data = parse_llm_json(plan_response)
    if plan_data is None:
        logger.error(f"Pass 1 JSON parse failed for {raw_path.name}")
        return []

    articles_plan = plan_data.get("articles", [])
    if not articles_plan:
        logger.info(f"  Skipped: {plan_data.get('skipped_reason', 'no articles planned')}")
        return []

    has_updates = any(a.get("action") == "update" for a in articles_plan)
    backlink_index = _build_backlink_index(WIKI_DIR) if has_updates else {}

    # -- Pass 2: Generate content per article --
    touched = []
    summaries_update: dict[str, str] = {}

    content_system = f"""{wiki_style}

Write a wiki article in plain markdown. Start directly with the frontmatter block (---).
For cross-reference links use [[type:slug]] syntax where type is one of:
references, depends_on, extends, contradicts, related
Bare [[slug]] is also valid and defaults to 'references'."""

    for article in articles_plan:
        path_str = article.get("path", "")
        action = article.get("action", "create")
        title = article.get("title", "")
        summary = article.get("summary", "")
        tags = article.get("tags", [])
        sections = article.get("sections", ["Overview", "Key Points", "Connections"])
        core_concepts = article.get("core_concepts", [])
        slug = Path(path_str).stem

        if not path_str or not title:
            continue

        if dry_run:
            print(f"  [DRY RUN] Would {action}: {path_str} -- {summary}")
            touched.append(path_str)
            continue

        article_path = WIKI_DIR / path_str
        existing_text = ""
        if article_path.exists() and action == "update":
            try:
                existing_text = f"\n\nEXISTING ARTICLE TO UPDATE:\n{article_path.read_text(encoding='utf-8')[:3000]}"
            except Exception:
                pass

        backlink_context = ""
        if action == "update" and backlink_index:
            backlink_context = _gather_backlink_context(slug, backlink_index, WIKI_DIR)

        content_user = f"""Write a wiki article.

Title: {title}
Path: {path_str}
Tags: {', '.join(tags)}
Sections to include:
{chr(10).join(f'- {s}' for s in sections)}

Source material:
---
{raw_content[:4000]}
---
Today's date: {datetime.now().strftime("%Y-%m-%d")}
{existing_text}
{backlink_context}

Start with YAML frontmatter (title, tags, updated, sources). Then write each section.
Use [[type:slug]] or [[slug]] syntax for cross-references."""

        if verbose:
            logger.info(f"=== PASS 2 PROMPT for {path_str} ===\n{content_user[:300]}...")

        try:
            content = llm_call(client, model, content_system, content_user, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Pass 2 failed for {path_str}: {e}")
            continue

        # Strip accidental code fences
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()

        article_path.parent.mkdir(parents=True, exist_ok=True)
        with locked_open(article_path, "w") as f:
            f.write(content)
        logger.info(f"  {action.upper()}: {path_str}")
        touched.append(path_str)
        if summary:
            summaries_update[slug] = summary

        # Post-write metadata
        metadata = {"last_compiled": datetime.now().strftime("%Y-%m-%d")}
        links = extract_typed_links(content)
        if links:
            metadata["links"] = links

        existing_fm = parse_frontmatter(article_path.read_text(encoding="utf-8"))
        if core_concepts:
            metadata["core_concepts"] = core_concepts
            existing_concepts = existing_fm.get("core_concepts", [])
            needs_expansion = (
                not existing_fm.get("expanded_terms")
                or sorted(existing_concepts) != sorted(core_concepts)
            )
            if needs_expansion:
                expanded = _expand_concepts(client, model, core_concepts)
                if expanded:
                    metadata["expanded_terms"] = expanded

        inject_metadata(article_path, metadata)

        if links:
            inject_reciprocal_backlinks(slug, links, WIKI_DIR)

    if not dry_run and touched:
        fresh_index = load_index()
        existing_summaries = read_existing_summaries()
        existing_summaries.update(summaries_update)
        rebuild_index(fresh_index, existing_summaries)
        logger.info(f"Index rebuilt: {len(fresh_index)} articles")

    return touched


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Orchestra wiki compiler")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--recompile-stale", action="store_true")
    parser.add_argument("--git", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    client, model, max_tokens = make_llm_client()
    sources = load_sources()

    # -- Stale recompile mode --
    if args.recompile_stale:
        stale = staleness_check(sources, WIKI_DIR)
        if not stale:
            logger.info("No stale articles found.")
            return

        logger.info(f"Found {len(stale)} stale article(s) -- recompiling...")
        sources_to_rerun: set[str] = set()
        article_to_recompile = {s["article"] for s in stale}
        for source_rel, entry in sources.get("processed", {}).items():
            if any(a in article_to_recompile for a in entry.get("articles", [])):
                sources_to_rerun.add(source_rel)

        for source_rel in sorted(sources_to_rerun):
            raw_path = KB_ROOT / source_rel
            if not raw_path.exists():
                continue
            logger.info(f"Recompiling: {source_rel}")
            try:
                touched = compile_file(raw_path, client, model, max_tokens=max_tokens,
                                       dry_run=args.dry_run, verbose=args.verbose)
                if not args.dry_run and touched:
                    _mark_processed(sources, raw_path, touched)
                    save_sources(sources)
            except Exception as e:
                logger.error(f"Failed recompiling {source_rel}: {e}")

        if not args.dry_run:
            _write_stale_report(staleness_check(sources, WIKI_DIR), WIKI_DIR)
        return

    # -- Normal compile mode --
    if args.source:
        p = Path(args.source)
        raw_files = [p if p.is_absolute() else KB_ROOT / p]
    else:
        raw_files = sorted(RAW_DIR.rglob("*.md"))

    if not raw_files:
        logger.info("No raw files found. Drop .md files into raw/ to get started.")
        return

    if not args.force and not args.source:
        already = set(sources.get("processed", {}).keys())
        raw_files = [f for f in raw_files if str(f.relative_to(KB_ROOT)) not in already]

    total_articles = 0
    if not raw_files:
        logger.info("No new files to process. Use --force to recompile all.")
    else:
        logger.info(f"Processing {len(raw_files)} file(s)...")
        for raw_path in raw_files:
            logger.info(f"Compiling: {raw_path.relative_to(KB_ROOT)}")
            try:
                touched = compile_file(raw_path, client, model, max_tokens=max_tokens,
                                       dry_run=args.dry_run, verbose=args.verbose)
                total_articles += len(touched)
                if not args.dry_run:
                    _mark_processed(sources, raw_path, touched)
                    save_sources(sources)
            except Exception as e:
                logger.error(f"Failed: {raw_path.name}: {e}")

        logger.info(f"Done. {len(raw_files)} file(s), {total_articles} article(s).")

    # Staleness report
    if not args.dry_run:
        stale = staleness_check(load_sources(), WIKI_DIR)
        logger.info(f"Stale: {len(stale)} article(s)" if stale else "Stale: 0 -- all current")
        _write_stale_report(stale, WIKI_DIR)

    if args.git and not args.dry_run:
        msg = f"compile: {total_articles} article(s) from {len(raw_files)} source(s)"
        if git_auto_commit(["wiki/", "raw/"], msg):
            logger.info(f"Git: committed ({msg})")
        else:
            logger.info("Git: nothing to commit or git not available")


if __name__ == "__main__":
    main()
