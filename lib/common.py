"""
Shared utilities for Orchestra tools.

Centralizes duplicated code: frontmatter parsing, wiki section discovery,
source tracking, file locking, LLM client creation, link extraction,
robust JSON parsing, YAML article writing, and shared constants.
"""

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path constants (all relative to project root)
# ---------------------------------------------------------------------------

KB_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = KB_ROOT / "raw"
WIKI_DIR = KB_ROOT / "wiki"
CONFIG_DIR = KB_ROOT / "config"
SOURCES_FILE = WIKI_DIR / "_sources.json"
INDEX_FILE = WIKI_DIR / "_index.md"

WIKI_SECTIONS_DEFAULT = ["concepts", "entities", "events", "research"]

# ---------------------------------------------------------------------------
# Link type constants (single source of truth)
# ---------------------------------------------------------------------------

LINK_TYPES = ["references", "depends_on", "extends", "contradicts", "related"]

# Precompiled regex patterns for link extraction
_TYPED_LINK_RE = re.compile(r"\[\[(" + "|".join(LINK_TYPES) + r"):([^\]]+)\]\]")
_BARE_LINK_RE = re.compile(r"\[\[([^\]:]+)\]\]")
_ANY_LINK_RE = re.compile(r"\[\[[a-z_]+:([^\]]+)\]\]")

# Forward link -> inverse link (used for reciprocal backlink injection)
INVERSE_LINK_TYPE = {
    "depends_on": "referenced_by",
    "extends": "referenced_by",
    "contradicts": "related",
    "references": "referenced_by",
    "related": "related",
}

# ---------------------------------------------------------------------------
# Cross-platform file locking
# ---------------------------------------------------------------------------

try:
    import fcntl

    def _lock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

except ImportError:
    # Windows fallback
    try:
        import msvcrt

        def _lock(f):
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

        def _unlock(f):
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

    except ImportError:
        # No locking available (rare) — proceed without
        def _lock(f):
            pass

        def _unlock(f):
            pass


@contextmanager
def locked_open(path, mode="a"):
    """Open file with exclusive lock to prevent concurrent write corruption."""
    with open(path, mode, encoding="utf-8") as f:
        _lock(f)
        try:
            yield f
        finally:
            _unlock(f)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict:
    """Load full config from config/config.json."""
    path = config_path or (CONFIG_DIR / "config.json")
    if not path.exists():
        print(f"Error: config not found at {path}", file=sys.stderr)
        print("Run setup.py or copy config/config.example.json to config/config.json", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_llm_config(config_path: Path | None = None) -> dict:
    """Load just the LLM section from config."""
    return load_config(config_path).get("llm", {})


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def make_llm_client(config: dict | None = None) -> tuple:
    """Return (client, model, max_tokens). Tries local first, then fallback.

    Accepts either a full config dict or just the llm section.
    """
    from openai import OpenAI
    if config is None:
        llm_cfg = load_llm_config()
    elif "llm" in config:
        llm_cfg = config["llm"]
    else:
        llm_cfg = config

    local_url = llm_cfg.get("local_url", "http://127.0.0.1:8081/v1")
    local_model = llm_cfg.get("local_model", "gemma4")
    max_tokens = llm_cfg.get("local_max_tokens", 6000)

    # Try local with a health check first (faster than a full completion)
    try:
        import httpx
        r = httpx.get(local_url.replace("/v1", "/health"), timeout=2)
        if r.status_code == 200:
            return OpenAI(base_url=local_url, api_key="local"), local_model, max_tokens
    except Exception:
        pass

    # Try local with a minimal completion as fallback health check
    try:
        client = OpenAI(base_url=local_url, api_key="local")
        resp = client.chat.completions.create(
            model=local_model, max_tokens=5,
            messages=[{"role": "user", "content": "hi"}], timeout=6,
        )
        if resp.choices:
            return client, local_model, max_tokens
    except Exception:
        pass

    # Fallback to remote API
    fallback_url = llm_cfg.get("fallback_url", "")
    fallback_model = llm_cfg.get("fallback_model", "")
    fallback_key_env = llm_cfg.get("fallback_api_key_env", "")
    api_key = os.environ.get(fallback_key_env, "") if fallback_key_env else ""

    if fallback_url and fallback_model and api_key:
        return OpenAI(base_url=fallback_url, api_key=api_key), fallback_model, max_tokens

    print(
        f"Error: no LLM available (local down, {fallback_key_env or 'fallback_api_key_env'} not set)",
        file=sys.stderr,
    )
    sys.exit(1)


def llm_call(client, model: str, system: str, user: str, max_tokens: int = 6000) -> str:
    """LLM call with temperature=0.0 and 3-attempt exponential-backoff retry.

    Client should be an OpenAI client instance from make_llm_client().
    """
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Wiki section discovery
# ---------------------------------------------------------------------------

def get_wiki_sections(wiki_dir: Path | None = None) -> list[str]:
    """Return all wiki section directories (concepts, entities, etc.)."""
    wd = wiki_dir or WIKI_DIR
    sections = set(WIKI_SECTIONS_DEFAULT)
    if wd.exists():
        for d in wd.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name != "meta":
                if any(d.rglob("*.md")):
                    sections.add(d.name)
    return sorted(sections)


def all_articles(wiki_dir: Path | None = None) -> dict[str, Path]:
    """Return {slug: Path} for every wiki article."""
    wd = wiki_dir or WIKI_DIR
    articles = {}
    for section in get_wiki_sections(wd):
        section_dir = wd / section
        if not section_dir.exists():
            continue
        for f in section_dir.rglob("*.md"):
            articles[f.stem] = f
    return articles


def count_articles(wiki_dir: Path | None = None) -> int:
    """Count all wiki articles."""
    return len(all_articles(wiki_dir))


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------

def load_sources(sources_file: Path | None = None) -> dict:
    """Load _sources.json."""
    sf = sources_file or SOURCES_FILE
    if sf.exists():
        return json.loads(sf.read_text(encoding="utf-8"))
    return {"processed": {}}


def save_sources(sources: dict, sources_file: Path | None = None) -> None:
    """Write _sources.json with file locking."""
    sf = sources_file or SOURCES_FILE
    with locked_open(sf, "w") as f:
        f.write(json.dumps(sources, indent=2))


# ---------------------------------------------------------------------------
# YAML frontmatter parsing and writing
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter from article text. Returns {} if none."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    try:
        result = yaml.safe_load(text[3:end].strip())
        return result if isinstance(result, dict) else {}
    except yaml.YAMLError:
        return {}


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split text into (frontmatter_dict, body_str)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    try:
        fm = yaml.safe_load(text[3:end].strip()) or {}
    except yaml.YAMLError:
        fm = {}
    body = text[end + 3:].lstrip("\n")
    return fm, body


def write_article(path: Path, fm: dict, body: str) -> None:
    """Write article with YAML frontmatter."""
    class _Dumper(yaml.Dumper):
        pass

    def _list_representer(dumper, data):
        if data and isinstance(data[0], dict):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    _Dumper.add_representer(list, _list_representer)
    fm_out = yaml.dump(fm, Dumper=_Dumper, default_flow_style=False, allow_unicode=True).strip()
    path.write_text(f"---\n{fm_out}\n---\n\n{body}", encoding="utf-8")


def inject_metadata(path: Path, fields: dict) -> None:
    """Inject or update fields in an article's YAML frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return

    fm, body = split_frontmatter(text)
    fm.update(fields)
    write_article(path, fm, body)


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------

def extract_typed_links(content: str) -> list[dict]:
    """Extract all [[type:slug]] and [[slug]] wikilinks from content.

    Returns deduplicated list of {target, type} dicts.
    Bare [[slug]] defaults to 'references'.
    """
    seen = {}
    for m in _TYPED_LINK_RE.finditer(content):
        link_type, target = m.group(1), m.group(2).strip()
        slug = Path(target).stem if "/" not in target else target.rsplit("/", 1)[-1].replace(".md", "")
        if slug not in seen:
            seen[slug] = {"target": slug, "type": link_type}
    for m in _BARE_LINK_RE.finditer(content):
        target = m.group(1).strip()
        slug = Path(target).stem
        if slug not in seen:
            seen[slug] = {"target": slug, "type": "references"}
    return list(seen.values())


def extract_wikilink_slugs(content: str) -> list[str]:
    """Extract all slug targets from [[type:slug]] and [[slug]] links."""
    slugs = []
    for m in _ANY_LINK_RE.finditer(content):
        slugs.append(Path(m.group(1).strip()).stem)
    for m in _BARE_LINK_RE.finditer(content):
        slugs.append(m.group(1).strip())
    return slugs


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------

def parse_llm_json(raw: str) -> dict | None:
    """Parse LLM output as JSON with fallbacks for common formatting issues.

    1. Strip markdown code fences
    2. Try json.loads directly
    3. Fall back to regex extraction of first {...} block
    4. Return None on failure
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Content sanitization
# ---------------------------------------------------------------------------

def sanitize_content(text: str, max_length: int = 10000) -> str:
    """Sanitize LLM-generated content before writing to disk.

    Strips null bytes and control characters (preserves newlines/tabs),
    limits length.
    """
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    if len(text) > max_length:
        text = text[:max_length] + "\n[... truncated ...]"
    return text


# ---------------------------------------------------------------------------
# Git auto-commit helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def load_index(wiki_dir: Path | None = None) -> dict[str, dict]:
    """Return {slug: {path, title, summary, tags, updated, section}} for all articles."""
    wd = wiki_dir or WIKI_DIR
    index = {}
    for section in get_wiki_sections(wd):
        section_dir = wd / section
        if not section_dir.exists():
            continue
        for md_file in section_dir.rglob("*.md"):
            slug = md_file.stem
            rel_path = str(md_file.relative_to(wd))
            try:
                fm = parse_frontmatter(md_file.read_text(encoding="utf-8"))
                index[slug] = {
                    "path": rel_path, "title": fm.get("title", slug),
                    "summary": "", "tags": fm.get("tags", []),
                    "updated": fm.get("updated", ""), "section": section,
                }
            except Exception:
                index[slug] = {
                    "path": rel_path, "title": slug, "summary": "",
                    "tags": [], "updated": "", "section": section,
                }
    return index


def read_existing_summaries(index_file: Path | None = None) -> dict[str, str]:
    """Extract summaries from current _index.md."""
    idx = index_file or INDEX_FILE
    summaries = {}
    if not idx.exists():
        return summaries
    for line in idx.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\*\*\[\[([^\]]+)\]\]\*\* -- (.+?)(?:\s+Tags:|\s+Updated:|$)", line)
        if m:
            summaries[m.group(1)] = m.group(2).strip().rstrip(".")
    return summaries


def rebuild_index(index: dict[str, dict] | None = None,
                  summaries: dict[str, str] | None = None,
                  wiki_dir: Path | None = None) -> int:
    """Rewrite _index.md. Returns article count.

    If index is None, loads from disk. If summaries is None, reads from existing _index.md.
    """
    from datetime import datetime
    wd = wiki_dir or WIKI_DIR
    idx_file = wd / "_index.md"

    if index is None:
        index = load_index(wd)
    if summaries is None:
        summaries = read_existing_summaries(idx_file)

    total = len(index)
    now = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"# Wiki Index",
        f"_Last updated: {now} | {total} article{'s' if total != 1 else ''}_",
        "", "---", "",
    ]

    for section in get_wiki_sections(wd):
        articles = {s: d for s, d in index.items() if d["section"] == section}
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

    with locked_open(idx_file, "w") as f:
        f.write("\n".join(lines))
    return total


# ---------------------------------------------------------------------------
# Reciprocal backlink injection
# ---------------------------------------------------------------------------

def inject_reciprocal_backlinks(source_slug: str, links: list[dict],
                                 wiki_dir: Path | None = None) -> None:
    """For each outbound link from source, inject a backlink into the target article."""
    wd = wiki_dir or WIKI_DIR
    for link in links:
        target_slug = link.get("target", "")
        link_type = link.get("type", "references")
        if not target_slug or target_slug == source_slug:
            continue

        target_path = None
        for section in get_wiki_sections(wd):
            p = wd / section / f"{target_slug}.md"
            if p.exists():
                target_path = p
                break
        if not target_path:
            continue

        inverse_type = INVERSE_LINK_TYPE.get(link_type, "related")
        try:
            text = target_path.read_text(encoding="utf-8")
            fm = parse_frontmatter(text)
            existing_links = fm.get("links", [])
            if not isinstance(existing_links, list):
                existing_links = []
            if any(l.get("target") == source_slug for l in existing_links if isinstance(l, dict)):
                continue
            existing_links.append({"target": source_slug, "type": inverse_type})
            inject_metadata(target_path, {"links": existing_links})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

def staleness_check(sources: dict, wiki_dir: Path | None = None) -> list[dict]:
    """Identify articles whose last_compiled is older than newest contributing source."""
    from datetime import datetime
    wd = wiki_dir or WIKI_DIR

    article_sources: dict[str, list[dict]] = {}
    for source_rel, entry in sources.get("processed", {}).items():
        processed_at = entry.get("processed_at", "")
        for article_path in entry.get("articles", []):
            article_sources.setdefault(article_path, []).append({
                "source": source_rel, "processed_at": processed_at,
            })

    stale = []
    for article_rel, contributing in article_sources.items():
        article_path = wd / article_rel
        if not article_path.exists():
            continue
        try:
            fm = parse_frontmatter(article_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        last_compiled = fm.get("last_compiled", "")
        if not last_compiled:
            continue

        newest = max(contributing, key=lambda x: x["processed_at"])
        newest_date = newest["processed_at"][:10]

        if newest_date > str(last_compiled):
            try:
                stale_days = (
                    datetime.strptime(newest_date, "%Y-%m-%d")
                    - datetime.strptime(str(last_compiled), "%Y-%m-%d")
                ).days
            except ValueError:
                stale_days = -1
            stale.append({
                "article": article_rel,
                "last_compiled": str(last_compiled),
                "newest_source": newest["source"],
                "newest_source_date": newest_date,
                "stale_days": stale_days,
            })

    stale.sort(key=lambda x: x["stale_days"], reverse=True)
    return stale


# ---------------------------------------------------------------------------
# Git auto-commit helper
# ---------------------------------------------------------------------------

def git_auto_commit(paths: list[str], message: str, cwd: Path | None = None) -> bool:
    """Stage paths and commit. Returns True on success."""
    import subprocess
    work_dir = str(cwd or KB_ROOT)
    try:
        subprocess.run(["git", "add"] + paths, cwd=work_dir, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=work_dir, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
