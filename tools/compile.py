#!/usr/bin/env python3
"""
Orchestra Wiki Compiler -- tools/compile.py

Scans raw/ for unprocessed markdown files.
For each new file: calls LLM to extract knowledge, creates/updates wiki/ articles,
updates _index.md and _sources.json.

LLM configuration loaded from config/config.json.

Usage:
  python3 tools/compile.py                   # process all new raw files
  python3 tools/compile.py --dry-run         # show what would change, no writes
  python3 tools/compile.py --force           # reprocess all files, not just new ones
  python3 tools/compile.py --source PATH     # compile a specific raw file only
  python3 tools/compile.py --verbose         # show LLM prompts and responses
  python3 tools/compile.py --recompile-stale # recompile articles whose sources are newer
"""

import argparse
import fcntl
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI

KB_ROOT = Path(__file__).parent.parent
RAW_DIR = KB_ROOT / "raw"
WIKI_DIR = KB_ROOT / "wiki"
CONFIG_DIR = KB_ROOT / "config"
SOURCES_FILE = WIKI_DIR / "_sources.json"
INDEX_FILE = WIKI_DIR / "_index.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WIKI_SECTIONS_DEFAULT = ["concepts", "entities", "events", "research"]

LINK_TYPES = ["references", "depends_on", "extends", "contradicts", "related"]
INVERSE_TYPES = {
    "depends_on": "depended_on_by",
    "extends": "extended_by",
    "contradicts": "contradicted_by",
    "references": "referenced_by",
    "related": "related",
}


# -- File locking --------------------------------------------------------------

@contextmanager
def _locked_open(path, mode="a"):
    """Open file with exclusive lock to prevent concurrent write corruption."""
    with open(path, mode, encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# -- LLM config loading -------------------------------------------------------

def _load_llm_config() -> dict:
    """Load LLM settings from config/config.json."""
    config_path = CONFIG_DIR / "config.json"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("llm", {})


# -- Wiki section discovery ----------------------------------------------------

def _get_wiki_sections() -> list[str]:
    """Return all top-level directories in wiki/ that contain .md files."""
    sections = set(WIKI_SECTIONS_DEFAULT)
    if WIKI_DIR.exists():
        for d in WIKI_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name != "meta":
                if any(d.rglob("*.md")):
                    sections.add(d.name)
    return sorted(sections)


# -- LLM client ---------------------------------------------------------------

def _make_client():
    """Return (client, model, max_tokens) using config/config.json settings."""
    import httpx
    llm_cfg = _load_llm_config()

    local_url = llm_cfg.get("local_url", "http://127.0.0.1:8081/v1")
    local_model = llm_cfg.get("local_model", "gemma4")
    max_tokens = llm_cfg.get("local_max_tokens", 6000)

    try:
        r = httpx.get(local_url.replace("/v1", "/health"), timeout=2)
        if r.status_code == 200:
            logger.info("Using local LLM server")
            return OpenAI(base_url=local_url, api_key="local"), local_model, max_tokens
    except Exception:
        pass

    fallback_url = llm_cfg.get("fallback_url", "")
    fallback_model = llm_cfg.get("fallback_model", "")
    api_key_env = llm_cfg.get("fallback_api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "")

    if fallback_url and api_key:
        logger.info("Local server unavailable, using fallback LLM")
        return OpenAI(base_url=fallback_url, api_key=api_key), fallback_model, max_tokens

    logger.error(
        f"No LLM available: local server down and {api_key_env} not set"
    )
    sys.exit(1)


def _llm_call(client, model: str, system: str, user: str, max_tokens: int = 6000) -> str:
    """Single LLM call. Returns response text or raises."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# -- Source tracking -----------------------------------------------------------

def _load_sources() -> dict:
    if SOURCES_FILE.exists():
        return json.loads(SOURCES_FILE.read_text())
    return {"processed": {}}


def _save_sources(sources: dict) -> None:
    with _locked_open(SOURCES_FILE, "w") as f:
        f.write(json.dumps(sources, indent=2))


def _mark_processed(sources: dict, raw_path: Path, articles_touched: list[str]) -> None:
    rel = str(raw_path.relative_to(KB_ROOT))
    sources["processed"][rel] = {
        "processed_at": datetime.now().isoformat(),
        "articles": articles_touched,
    }


# -- Frontmatter parsing and injection ----------------------------------------

def _parse_frontmatter_yaml(text: str) -> dict:
    """Extract YAML frontmatter using pyyaml. Returns {} if none or parse error."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    try:
        result = yaml.safe_load(fm_text)
        return result if isinstance(result, dict) else {}
    except yaml.YAMLError:
        return {}


def _parse_frontmatter(text: str) -> dict:
    """Alias for yaml-based parser. Kept for backward compat with callers."""
    return _parse_frontmatter_yaml(text)


def _inject_metadata(path: Path, fields: dict) -> None:
    """
    Inject or update fields in an article's YAML frontmatter.
    Reads the file, merges fields into existing frontmatter, rewrites the file.
    Creates frontmatter block if none exists.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read {path} for metadata injection: {e}")
        return

    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm_text = text[3:end].strip()
            body = text[end + 3:].lstrip("\n")
            try:
                fm = yaml.safe_load(fm_text) or {}
            except yaml.YAMLError:
                fm = {}
        else:
            fm = {}
            body = text
    else:
        fm = {}
        body = text

    fm.update(fields)

    class _Dumper(yaml.Dumper):
        pass

    def _list_representer(dumper, data):
        if data and isinstance(data[0], dict):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    _Dumper.add_representer(list, _list_representer)

    fm_out = yaml.dump(fm, Dumper=_Dumper, default_flow_style=False, allow_unicode=True).strip()
    new_text = f"---\n{fm_out}\n---\n\n{body}"
    path.write_text(new_text, encoding="utf-8")


# -- Link extraction -----------------------------------------------------------

def _extract_links_from_content(content: str) -> list[dict]:
    """
    Extract all [[type:slug]] and [[slug]] wikilinks from article content.
    Returns deduplicated list of {target, type} dicts.
    Bare [[slug]] defaults to 'references'.
    """
    seen = {}
    # Typed links: [[type:slug]]
    for m in re.finditer(r"\[\[(" + "|".join(LINK_TYPES) + r"):([^\]]+)\]\]", content):
        link_type, target = m.group(1), m.group(2).strip()
        slug = Path(target).stem if "/" not in target else target.rsplit("/", 1)[-1].replace(".md", "")
        if slug not in seen:
            seen[slug] = {"target": slug, "type": link_type}
    # Bare links: [[slug]] -- skip if already captured as typed
    for m in re.finditer(r"\[\[([^\]:]+)\]\]", content):
        target = m.group(1).strip()
        slug = Path(target).stem
        if slug not in seen:
            seen[slug] = {"target": slug, "type": "references"}
    return list(seen.values())


INVERSE_LINK_TYPE = {
    "depends_on": "referenced_by",
    "extends": "referenced_by",
    "contradicts": "related",
    "references": "referenced_by",
    "related": "related",
}


def _inject_reciprocal_backlinks(source_slug: str, links: list[dict], wiki_dir: Path) -> None:
    """For each outbound link from source, inject a backlink into the target article."""
    for link in links:
        target_slug = link.get("target", "")
        link_type = link.get("type", "references")
        if not target_slug or target_slug == source_slug:
            continue

        # Find target article
        target_path = None
        for section in _get_wiki_sections():
            p = wiki_dir / section / f"{target_slug}.md"
            if p.exists():
                target_path = p
                break
        if not target_path:
            continue

        inverse_type = INVERSE_LINK_TYPE.get(link_type, "related")
        backlink = {"target": source_slug, "type": inverse_type}

        try:
            text = target_path.read_text(encoding="utf-8")
            fm = _parse_frontmatter_yaml(text)
            existing_links = fm.get("links", [])
            if not isinstance(existing_links, list):
                existing_links = []

            # Skip if already linked
            if any(l.get("target") == source_slug for l in existing_links if isinstance(l, dict)):
                continue

            existing_links.append(backlink)
            _inject_metadata(target_path, {"links": existing_links})
            logger.debug(f"  Backlink: {target_slug} <- {source_slug} ({inverse_type})")
        except Exception as e:
            logger.warning(f"  Failed to inject backlink into {target_slug}: {e}")


# -- Multi-query expansion -----------------------------------------------------

def expand_concepts(client, model: str, concepts: list[str]) -> list[str]:
    """
    Generate alternate phrasings / semantically related terms for a list of concepts.
    Returns a flat deduplicated list of expansion terms.
    Single LLM call -- short prompt, low stakes.
    """
    if not concepts:
        return []
    concepts_str = ", ".join(concepts)
    prompt = (
        f"Generate 3-5 alternate phrasings or semantically related terms for each of "
        f"these concepts: {concepts_str}\n"
        f"Return ONLY a flat comma-separated list of terms. No explanations, no numbering, "
        f"no grouping. Example output: term1, term2, term3, term4"
    )
    try:
        raw = _llm_call(client, model, "You are a semantic expansion tool.", prompt, max_tokens=300)
        terms = [t.strip().lower() for t in raw.split(",") if t.strip()]
        original_lower = {c.lower() for c in concepts}
        return [t for t in dict.fromkeys(terms) if t not in original_lower]
    except Exception as e:
        logger.warning(f"expand_concepts failed: {e}")
        return []


# -- Backlink index ------------------------------------------------------------

def _build_backlink_index(wiki_dir: Path) -> dict[str, list[str]]:
    """
    Scan all wiki articles and build {slug -> [slugs that link to it]}.
    Prefers links: frontmatter block; falls back to [[slug]] regex scan.
    """
    backlinks: dict[str, list[str]] = {}

    for section in _get_wiki_sections():
        section_dir = wiki_dir / section
        if not section_dir.exists():
            continue
        for md_file in section_dir.rglob("*.md"):
            source_slug = md_file.stem
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            fm = _parse_frontmatter_yaml(text)
            links_fm = fm.get("links", [])

            if links_fm and isinstance(links_fm, list):
                for link in links_fm:
                    if isinstance(link, dict):
                        target = link.get("target", "")
                        target_slug = Path(target).stem if target else ""
                        if target_slug:
                            backlinks.setdefault(target_slug, [])
                            if source_slug not in backlinks[target_slug]:
                                backlinks[target_slug].append(source_slug)
            else:
                for link in _extract_links_from_content(text):
                    target_slug = link["target"]
                    backlinks.setdefault(target_slug, [])
                    if source_slug not in backlinks[target_slug]:
                        backlinks[target_slug].append(source_slug)

    return backlinks


def _gather_backlink_context(
    slug: str,
    backlink_index: dict[str, list[str]],
    wiki_dir: Path,
    max_articles: int = 10,
) -> str:
    """
    Walk backlinks up to depth 2. Return a 'Related Context' block for the
    Pass 2 compilation prompt.

    Priority: depth-1 first, then depth-2.
    Cap at max_articles total.
    Extracts first 2-3 sentences of each article's Overview for context.
    """
    depth1 = backlink_index.get(slug, [])
    depth2 = []
    for d1 in depth1:
        for d2 in backlink_index.get(d1, []):
            if d2 != slug and d2 not in depth1 and d2 not in depth2:
                depth2.append(d2)

    candidates = [(s, 1) for s in depth1] + [(s, 2) for s in depth2]

    def sort_key(item):
        s, depth = item
        for section in _get_wiki_sections():
            p = wiki_dir / section / f"{s}.md"
            if p.exists():
                try:
                    fm = _parse_frontmatter_yaml(p.read_text(encoding="utf-8"))
                    lc = str(fm.get("last_compiled", "1970-01-01"))
                    return (depth, lc)
                except Exception:
                    pass
        return (depth, "1970-01-01")

    candidates.sort(key=sort_key)
    candidates = candidates[:max_articles]

    if not candidates:
        return ""

    lines = ["## Related Context",
             "_Articles linking to this one -- use for cross-reference enrichment only. "
             "Do not shift this article's focus._", ""]

    for s, depth in candidates:
        article_path = None
        for section in _get_wiki_sections():
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

        depth_label = f"depth {depth}"
        lines.append(f"- **[[{s}]]** ({depth_label}): {excerpt}")

    return "\n".join(lines)


# -- Staleness detection -------------------------------------------------------

def staleness_check(sources: dict, wiki_dir: Path) -> list[dict]:
    """
    Identify wiki articles whose last_compiled date is older than the
    most recent source that contributed to them.

    Returns list of {article, last_compiled, newest_source, stale_days} dicts.
    """
    article_sources: dict[str, list[dict]] = {}
    for source_rel, entry in sources.get("processed", {}).items():
        processed_at = entry.get("processed_at", "")
        for article_path in entry.get("articles", []):
            article_sources.setdefault(article_path, []).append({
                "source": source_rel,
                "processed_at": processed_at,
            })

    stale = []
    for article_rel, contributing in article_sources.items():
        article_path = wiki_dir / article_rel
        if not article_path.exists():
            continue
        try:
            text = article_path.read_text(encoding="utf-8")
        except Exception:
            continue
        fm = _parse_frontmatter_yaml(text)
        last_compiled = fm.get("last_compiled", "")
        if not last_compiled:
            continue

        newest_source = max(contributing, key=lambda x: x["processed_at"])
        newest_date = newest_source["processed_at"][:10]

        if newest_date > str(last_compiled):
            try:
                lc_dt = datetime.strptime(str(last_compiled), "%Y-%m-%d")
                ns_dt = datetime.strptime(newest_date, "%Y-%m-%d")
                stale_days = (ns_dt - lc_dt).days
            except ValueError:
                stale_days = -1

            stale.append({
                "article": article_rel,
                "last_compiled": str(last_compiled),
                "newest_source": newest_source["source"],
                "newest_source_date": newest_date,
                "stale_days": stale_days,
            })

    stale.sort(key=lambda x: x["stale_days"], reverse=True)
    return stale


def _write_stale_report(stale: list[dict], wiki_dir: Path) -> None:
    """Write wiki/meta/stale.md with current staleness data."""
    meta_dir = wiki_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    stale_file = meta_dir / "stale.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [f"# Stale Articles", f"_Last updated: {now}_", ""]
    if not stale:
        lines.append("*(none -- all compiled articles are current)*")
    else:
        lines.append(f"{len(stale)} article(s) have newer source material than their last compile:\n")
        for s in stale:
            lines.append(
                f"- **{s['article']}** -- last compiled {s['last_compiled']}, "
                f"newer source {s['newest_source_date']} ({s['stale_days']}d stale) "
                f"via `{s['newest_source']}`"
            )

    with _locked_open(stale_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Stale report written: {len(stale)} stale article(s)")


# -- Index management ----------------------------------------------------------

def _load_index() -> dict[str, dict]:
    """Return {slug: {path, title, summary, tags, updated, section}} for all wiki articles."""
    index: dict[str, dict] = {}
    for section in _get_wiki_sections():
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            continue
        for md_file in section_dir.rglob("*.md"):
            slug = md_file.stem
            rel_path = str(md_file.relative_to(WIKI_DIR))
            try:
                text = md_file.read_text(encoding="utf-8")
                fm = _parse_frontmatter_yaml(text)
                index[slug] = {
                    "path": rel_path,
                    "title": fm.get("title", slug),
                    "summary": "",
                    "tags": fm.get("tags", []),
                    "updated": fm.get("updated", ""),
                    "section": section,
                }
            except Exception:
                index[slug] = {
                    "path": rel_path,
                    "title": slug,
                    "summary": "",
                    "tags": [],
                    "updated": "",
                    "section": section,
                }
    return index


def _rebuild_index(index: dict[str, dict], summaries: dict[str, str]) -> None:
    """Rewrite _index.md from current state of wiki/."""
    total = len(index)
    now = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"# Wiki Index",
        f"_Last updated: {now} | {total} article{'s' if total != 1 else ''}_",
        "",
        "---",
        "",
    ]

    for section in _get_wiki_sections():
        articles = {slug: data for slug, data in index.items() if data["section"] == section}
        lines.append(f"## {section}/ ({len(articles)})")
        lines.append("")
        if not articles:
            lines.append("*(empty)*")
            lines.append("")
            continue
        for slug, data in sorted(articles.items()):
            summary = summaries.get(slug) or data.get("summary") or data.get("title", slug)
            tags_str = ", ".join(data.get("tags", [])) if data.get("tags") else ""
            updated = data.get("updated", "")
            entry = f"**[[{slug}]]** -- {summary}"
            if tags_str:
                entry += f" Tags: {tags_str}."
            if updated:
                entry += f" Updated: {updated}."
            lines.append(entry)
        lines.append("")

    with _locked_open(INDEX_FILE, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Index rebuilt: {total} articles")


# -- Strip JSON fence ----------------------------------------------------------

def _strip_json_fence(raw: str) -> str:
    """Strip markdown code fences and extract JSON.

    Falls back to regex extraction of first {...} block if direct parse fails.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    # Verify it's valid JSON; if not, try regex extraction
    try:
        json.loads(cleaned)
        return cleaned
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex fallback: extract first {...} block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            json.loads(m.group(0))
            return m.group(0)
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(f"_strip_json_fence: could not extract valid JSON ({len(raw)} chars)")
    return cleaned


# -- Compile a single raw file -------------------------------------------------

def compile_file(
    raw_path: Path,
    client,
    model: str,
    max_tokens: int = 6000,
    dry_run: bool = False,
    verbose: bool = False,
) -> list[str]:
    """
    Process one raw file using two LLM passes.
    Pass 1: JSON plan (paths, titles, tags, sections, core_concepts).
    Pass 2: Plain markdown content for each article.
    Post-write: inject last_compiled, links, expanded_terms metadata.
    Returns list of article paths touched.
    """
    compile_rules = (CONFIG_DIR / "compile-rules.md").read_text(encoding="utf-8")
    wiki_style = (CONFIG_DIR / "wiki-style.md").read_text(encoding="utf-8")
    raw_content = raw_path.read_text(encoding="utf-8")

    index = _load_index()
    article_titles = "\n".join(
        f"- [{data['title']}]({data['path']})"
        for slug, data in sorted(index.items())
    ) or "(wiki is empty -- create new articles freely)"

    existing_content_parts = []
    raw_lower = raw_content.lower()
    for slug, data in index.items():
        slug_plain = slug.replace("-", " ")
        matched = slug_plain in raw_lower or slug in raw_lower
        if not matched:
            article_path = WIKI_DIR / data["path"]
            if article_path.exists():
                try:
                    afm = _parse_frontmatter_yaml(article_path.read_text(encoding="utf-8"))
                    expanded = afm.get("expanded_terms", [])
                    if isinstance(expanded, list):
                        for term in expanded:
                            if term.lower() in raw_lower:
                                matched = True
                                break
                except Exception:
                    pass
        if matched:
            article_path = WIKI_DIR / data["path"]
            if article_path.exists():
                try:
                    existing_content_parts.append(
                        f"### EXISTING: {data['path']}\n"
                        f"{article_path.read_text(encoding='utf-8')[:2000]}"
                    )
                except Exception:
                    pass

    existing_block = (
        "\n\n".join(existing_content_parts)
        if existing_content_parts
        else "(no existing articles to update)"
    )

    # -- Pass 1: Article plan --------------------------------------------------
    plan_system = f"""{compile_rules}

You must respond with ONLY a JSON object. No markdown fences, no prose, no explanation.
The JSON must have this exact structure -- no other fields, no nested markdown:
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
core_concepts: 3-5 key terms/concepts central to this article (used for cross-reference expansion)."""

    plan_user = f"""Plan which wiki articles to create or update based on this raw source.

RAW SOURCE FILE: {raw_path.relative_to(KB_ROOT)}
---
{raw_content[:5000]}
---

EXISTING WIKI ARTICLES:
{article_titles}

Today's date: {datetime.now().strftime("%Y-%m-%d")}

Return the JSON plan only. No content yet -- just paths, titles, summaries, tags, section names, and core_concepts."""

    if verbose:
        logger.info(f"=== PASS 1 PROMPT ({len(plan_user)} chars) ===\n{plan_user[:300]}...")

    try:
        plan_response = _llm_call(client, model, plan_system, plan_user, max_tokens=2000)
    except Exception as e:
        logger.error(f"Pass 1 LLM call failed for {raw_path.name}: {e}")
        return []

    if verbose:
        logger.info(f"=== PASS 1 RESPONSE ===\n{plan_response[:500]}")

    try:
        clean = _strip_json_fence(plan_response)
        plan_data = json.loads(clean)
    except json.JSONDecodeError as e:
        m = re.search(r"\{.*\}", plan_response, re.DOTALL)
        if m:
            try:
                plan_data = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.error(f"Pass 1 JSON parse failed for {raw_path.name}: {e}")
                logger.debug(f"Raw response: {plan_response[:500]}")
                return []
        else:
            logger.error(f"Pass 1 JSON parse failed for {raw_path.name}: {e}")
            logger.debug(f"Raw response: {plan_response[:500]}")
            return []

    articles_plan = plan_data.get("articles", [])
    if not articles_plan:
        reason = plan_data.get("skipped_reason", "no articles planned")
        logger.info(f"  Skipped: {reason}")
        return []

    # Build backlink index once if any updates planned (for backlink context injection)
    has_updates = any(a.get("action") == "update" for a in articles_plan)
    backlink_index = _build_backlink_index(WIKI_DIR) if has_updates else {}

    # -- Pass 2: Generate content for each article -----------------------------
    touched = []
    summaries_update: dict[str, str] = {}

    content_system = f"""{wiki_style}

Write a wiki article in plain markdown. Start directly with the frontmatter block (---).
No preamble, no explanation, just the article.

For cross-reference links use [[type:slug]] syntax where type is one of:
references, depends_on, extends, contradicts, related
Example: [[depends_on:llm-agents]], [[extends:saver-framework]]
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
            logger.warning(f"  Skipping malformed plan entry: {article}")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would {action}: {path_str} -- {summary}")
            touched.append(path_str)
            continue

        article_path = WIKI_DIR / path_str
        existing_text = ""
        if article_path.exists() and action == "update":
            try:
                existing_text = (
                    f"\n\nEXISTING ARTICLE TO UPDATE:\n"
                    f"{article_path.read_text(encoding='utf-8')[:3000]}"
                )
            except Exception:
                pass

        tags_str = ", ".join(tags)
        sections_str = "\n".join(f"- {s}" for s in sections)

        # Gather backlink context for updates
        backlink_context = ""
        if action == "update" and backlink_index:
            backlink_context = _gather_backlink_context(slug, backlink_index, WIKI_DIR)

        content_user = f"""Write a wiki article.

Title: {title}
Path: {path_str}
Tags: {tags_str}
Sections to include:
{sections_str}

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
            content = _llm_call(
                client, model, content_system, content_user, max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"Pass 2 LLM call failed for {path_str}: {e}")
            continue

        if verbose:
            logger.info(f"=== PASS 2 RESPONSE ===\n{content[:300]}")

        # Strip any accidental code fences wrapping the whole article
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()

        article_path.parent.mkdir(parents=True, exist_ok=True)
        with _locked_open(article_path, "w") as f:
            f.write(content)
        logger.info(f"  {action.upper()}: {path_str}")
        touched.append(path_str)
        if summary:
            summaries_update[slug] = summary

        # -- Post-write metadata injection -------------------------------------
        metadata = {"last_compiled": datetime.now().strftime("%Y-%m-%d")}

        # Extract typed links from written content
        links = _extract_links_from_content(content)
        if links:
            metadata["links"] = links

        # Expand core concepts (skip if article already has expanded_terms)
        existing_fm = _parse_frontmatter_yaml(article_path.read_text(encoding="utf-8"))
        if core_concepts and not existing_fm.get("expanded_terms"):
            expanded = expand_concepts(client, model, core_concepts)
            if expanded:
                metadata["expanded_terms"] = expanded

        _inject_metadata(article_path, metadata)

        # -- Reciprocal backlink injection -------------------------------------
        # For each outbound link, add a backlink in the target's frontmatter
        if links:
            _inject_reciprocal_backlinks(slug, links, WIKI_DIR)

    if not dry_run and touched:
        fresh_index = _load_index()
        existing_summaries = _read_existing_summaries()
        existing_summaries.update(summaries_update)
        _rebuild_index(fresh_index, existing_summaries)

    return touched


def _read_existing_summaries() -> dict[str, str]:
    """Extract summaries from current _index.md."""
    summaries = {}
    if not INDEX_FILE.exists():
        return summaries
    for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(
            r"\*\*\[\[([^\]]+)\]\]\*\* -- ([^.]+(?:\.[^T][^a][^g][^s])*?)(?:\s+Tags:|\s+Updated:|$)",
            line,
        )
        if m:
            slug, summary = m.group(1), m.group(2).strip().rstrip(".")
            summaries[slug] = summary
    return summaries


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Orchestra wiki compiler")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, no writes")
    parser.add_argument("--force", action="store_true", help="Reprocess all files, not just new ones")
    parser.add_argument("--source", type=str, help="Compile only a specific raw file (path)")
    parser.add_argument("--verbose", action="store_true", help="Show LLM prompts and responses")
    parser.add_argument(
        "--recompile-stale",
        action="store_true",
        help="Recompile articles whose source material is newer than their last_compiled date",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    client, model, max_tokens = _make_client()
    sources = _load_sources()

    # -- Stale recompile mode --------------------------------------------------
    if args.recompile_stale:
        stale = staleness_check(sources, WIKI_DIR)
        if not stale:
            logger.info("No stale articles found.")
            return

        logger.info(f"Found {len(stale)} stale article(s) -- recompiling...")

        article_to_recompile = {s["article"] for s in stale}
        sources_to_rerun: set[str] = set()
        for source_rel, entry in sources.get("processed", {}).items():
            for article in entry.get("articles", []):
                if article in article_to_recompile:
                    sources_to_rerun.add(source_rel)

        for source_rel in sorted(sources_to_rerun):
            raw_path = KB_ROOT / source_rel
            if not raw_path.exists():
                logger.warning(f"Source file not found: {source_rel}")
                continue
            logger.info(f"Recompiling: {source_rel}")
            try:
                touched = compile_file(
                    raw_path, client, model, max_tokens=max_tokens,
                    dry_run=args.dry_run, verbose=args.verbose,
                )
                if not args.dry_run and touched:
                    _mark_processed(sources, raw_path, touched)
                    _save_sources(sources)
            except Exception as e:
                logger.error(f"Failed recompiling {source_rel}: {e}")

        updated_stale = staleness_check(sources, WIKI_DIR)
        if updated_stale:
            logger.info(f"Stale after recompile: {len(updated_stale)} article(s)")
        else:
            logger.info("All articles current after recompile.")
        if not args.dry_run:
            _write_stale_report(updated_stale, WIKI_DIR)
        return

    # -- Normal compile mode ---------------------------------------------------

    if args.source:
        p = Path(args.source)
        raw_files = [p if p.is_absolute() else KB_ROOT / p]
    else:
        raw_files = sorted(RAW_DIR.rglob("*.md"))

    if not raw_files:
        logger.info("No raw files found. Drop .md files into raw/ to get started.")
        return

    if not args.force and not args.source:
        already_processed = set(sources.get("processed", {}).keys())
        raw_files = [
            f for f in raw_files
            if str(f.relative_to(KB_ROOT)) not in already_processed
        ]

    if not raw_files:
        logger.info("No new files to process. Use --force to recompile all.")
    else:
        logger.info(f"Processing {len(raw_files)} file(s)...")

        total_articles = 0
        for raw_path in raw_files:
            logger.info(f"Compiling: {raw_path.relative_to(KB_ROOT)}")
            try:
                touched = compile_file(
                    raw_path, client, model, max_tokens=max_tokens,
                    dry_run=args.dry_run, verbose=args.verbose,
                )
                total_articles += len(touched)
                if not args.dry_run:
                    _mark_processed(sources, raw_path, touched)
                    _save_sources(sources)
            except Exception as e:
                logger.error(f"Failed: {raw_path.name}: {e}")
                continue

        logger.info(
            f"Done. {len(raw_files)} file(s) processed, {total_articles} article(s) created/updated."
        )

    # -- Staleness report (always) ---------------------------------------------
    if not args.dry_run:
        sources = _load_sources()
        stale = staleness_check(sources, WIKI_DIR)
        if stale:
            logger.info(f"Stale: {len(stale)} article(s)")
            for s in stale:
                logger.info(
                    f"  {s['article']} (compiled {s['last_compiled']}, "
                    f"source {s['newest_source_date']}, {s['stale_days']}d)"
                )
        else:
            logger.info("Stale: 0 articles -- all current")
        _write_stale_report(stale, WIKI_DIR)


if __name__ == "__main__":
    main()
