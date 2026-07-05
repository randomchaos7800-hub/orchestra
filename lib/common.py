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
import tempfile
import hashlib
import importlib.util
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import urlparse

import yaml

T = TypeVar("T")

# Load .env if present — lets users set OPENROUTER_API_KEY in a .env file
# instead of exporting it in their shell. Falls back silently if not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Path constants (all relative to project root)
# ---------------------------------------------------------------------------

KB_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = KB_ROOT / "raw"
WIKI_DIR = KB_ROOT / "wiki"
CONFIG_DIR = KB_ROOT / "config"
SOURCES_FILE = WIKI_DIR / "_sources.json"
INDEX_FILE = WIKI_DIR / "_index.md"
CACHE_DIR = KB_ROOT / "cache"
LLM_CACHE_DIR = CACHE_DIR / "llm"

WIKI_SECTIONS_DEFAULT = ["concepts", "entities", "events", "research", "tools"]

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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        _lock(f)
        try:
            yield f
        finally:
            _unlock(f)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write a file atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding=encoding) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def locked_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically write plain text under an exclusive lock."""
    lock_path = path.parent / f".{path.name}.lock"
    with locked_open(lock_path, "a"):
        atomic_write_text(path, content, encoding=encoding)


def read_json_file(path: Path, default):
    """Read JSON under lock, or return a deep-copied default if missing."""
    lock_path = path.parent / f".{path.name}.lock"
    with locked_open(lock_path, "a"):
        if not path.exists() or path.stat().st_size == 0:
            return deepcopy(default)
        return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, data) -> None:
    """Atomically write JSON under an exclusive lock."""
    lock_path = path.parent / f".{path.name}.lock"
    with locked_open(lock_path, "a"):
        atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def update_json_file(path: Path, default, updater: Callable[[dict | list], T]) -> T:
    """Read-modify-write a JSON file under lock with atomic replacement."""
    lock_path = path.parent / f".{path.name}.lock"
    with locked_open(lock_path, "a"):
        if path.exists() and path.stat().st_size > 0:
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = deepcopy(default)
        result = updater(data)
        atomic_write_text(path, json.dumps(data, indent=2) + "\n")
        return result


def update_text_file(path: Path, default: str, updater: Callable[[str], tuple[str, T]]) -> T:
    """Read-modify-write a text file under lock with atomic replacement."""
    lock_path = path.parent / f".{path.name}.lock"
    with locked_open(lock_path, "a"):
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = default
        new_text, result = updater(text)
        atomic_write_text(path, new_text)
        return result


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class ConfigValidationError(ValueError):
    """Raised when config/config.json is missing required settings."""


def _is_valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_config(config: dict, *, require_capture: bool = False) -> None:
    """Validate an Orchestra config dict.

    Raises:
        ConfigValidationError: with actionable messages for each invalid field.
    """
    errors: list[str] = []

    if not isinstance(config, dict):
        raise ConfigValidationError("Config must be a JSON object.")

    llm = config.get("llm")
    if not isinstance(llm, dict):
        errors.append("Config missing llm. Run setup.py or add it to config.json.")
    else:
        local_url = llm.get("local_url")
        if not isinstance(local_url, str) or not local_url.strip():
            errors.append("Config missing llm.local_url. Run setup.py or add it to config.json.")
        elif not _is_valid_http_url(local_url):
            errors.append(
                f"Invalid LLM URL for llm.local_url: {local_url!r}. "
                "Use an http(s) OpenAI-compatible endpoint such as http://127.0.0.1:8010/v1."
            )

        local_model = llm.get("local_model")
        if not isinstance(local_model, str) or not local_model.strip():
            errors.append("Config missing llm.local_model. Run setup.py or add it to config.json.")

        max_tokens = llm.get("local_max_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            errors.append("Config missing llm.local_max_tokens. Add a positive integer to config.json.")

        fallback_fields = {
            "fallback_url": llm.get("fallback_url", ""),
            "fallback_model": llm.get("fallback_model", ""),
            "fallback_api_key_env": llm.get("fallback_api_key_env", ""),
        }
        fallback_enabled = any(bool(str(v).strip()) for v in fallback_fields.values())
        if fallback_enabled:
            for field, value in fallback_fields.items():
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"Config missing llm.{field}. Run setup.py or add it to config.json.")
            fallback_url = fallback_fields["fallback_url"]
            if isinstance(fallback_url, str) and fallback_url.strip() and not _is_valid_http_url(fallback_url):
                errors.append(
                    f"Invalid LLM URL for llm.fallback_url: {fallback_url!r}. "
                    "Use an http(s) OpenAI-compatible endpoint."
                )

    if require_capture:
        capture = config.get("capture")
        if not isinstance(capture, dict):
            errors.append("Config missing capture. Run setup.py or add it to config.json.")
        else:
            projects = capture.get("projects")
            if not isinstance(projects, dict) or not projects:
                errors.append("Config missing capture.projects. Add at least one project category.")

    wiki = config.get("wiki")
    if wiki is not None:
        sections = wiki.get("sections") if isinstance(wiki, dict) else None
        if sections is not None and (
            not isinstance(sections, list)
            or any(not isinstance(section, str) or not section.strip() for section in sections)
        ):
            errors.append("Config field wiki.sections must be a list of non-empty strings.")

    if errors:
        raise ConfigValidationError("\n".join(errors))


def check_dependencies(packages: dict[str, str] | list[str] | tuple[str, ...]) -> list[str]:
    """Return package names whose import modules are unavailable.

    Args:
        packages: either {package_name: import_name} or an iterable where the
            package name and import name are identical.
    """
    if isinstance(packages, dict):
        package_map = packages
    else:
        package_map = {name: name for name in packages}

    missing = []
    for package_name, import_name in package_map.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
    return missing


def print_dependency_error(tool_name: str, missing: list[str], install_command: str) -> None:
    """Print an actionable optional dependency error for CLI tools."""
    print(f"Error: {tool_name} requires optional dependencies: {', '.join(missing)}", file=sys.stderr)
    print(f"Install them with: {install_command}", file=sys.stderr)


def load_config(config_path: Path | None = None) -> dict:
    """Load full config from config/config.json."""
    path = config_path or (CONFIG_DIR / "config.json")
    if not path.exists():
        print(f"Error: config not found at {path}", file=sys.stderr)
        print("Run setup.py or copy config/config.example.json to config/config.json", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    try:
        validate_config(config)
    except ConfigValidationError as exc:
        print(f"Error: invalid config at {path}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("Run setup.py or update config.json from config/config.example.json", file=sys.stderr)
        sys.exit(1)
    return config


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

    local_url = llm_cfg.get("local_url", "http://127.0.0.1:8010/v1")
    local_model = llm_cfg.get("local_model", "gemma4")
    max_tokens = llm_cfg.get("local_max_tokens", 6000)
    local_error = "not attempted"

    # Try local with a health check first (faster than a full completion)
    try:
        import httpx
        r = httpx.get(local_url.replace("/v1", "/health"), timeout=2)
        if r.status_code == 200:
            return OpenAI(base_url=local_url, api_key="local"), local_model, max_tokens
        local_error = f"health check returned HTTP {r.status_code}"
    except Exception as exc:
        local_error = f"health check failed: {exc}"

    # Try local with a minimal completion as fallback health check
    try:
        client = OpenAI(base_url=local_url, api_key="local")
        resp = client.chat.completions.create(
            model=local_model, max_tokens=5,
            messages=[{"role": "user", "content": "hi"}], timeout=6,
        )
        if resp.choices:
            return client, local_model, max_tokens
        local_error = "test completion returned no choices"
    except Exception as exc:
        local_error = f"test completion failed: {exc}"

    # Fallback to remote API
    fallback_url = llm_cfg.get("fallback_url", "")
    fallback_model = llm_cfg.get("fallback_model", "")
    fallback_key_env = llm_cfg.get("fallback_api_key_env", "")
    api_key = os.environ.get(fallback_key_env, "") if fallback_key_env else ""

    fallback_error = ""
    if fallback_url and fallback_model and fallback_key_env:
        if not api_key:
            fallback_error = (
                f"fallback API key missing: set {fallback_key_env}=your-key "
                "in .env or your shell"
            )
        else:
            try:
                client = OpenAI(base_url=fallback_url, api_key=api_key)
                resp = client.chat.completions.create(
                    model=fallback_model, max_tokens=5,
                    messages=[{"role": "user", "content": "hi"}], timeout=8,
                )
                if resp.choices:
                    return client, fallback_model, max_tokens
                fallback_error = "fallback test completion returned no choices"
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code in (401, 403):
                    fallback_error = (
                        f"fallback API key rejected by {fallback_url} "
                        f"(HTTP {status_code}); check {fallback_key_env}"
                    )
                else:
                    fallback_error = f"fallback endpoint failed: {fallback_url} ({exc})"
    else:
        fallback_error = "fallback is not fully configured in config.json"

    print(
        "Error: no LLM available.",
        file=sys.stderr,
    )
    print(f"  Local endpoint failed: {local_url} ({local_error})", file=sys.stderr)
    print(f"  {fallback_error}", file=sys.stderr)
    print("Remediation: start your local OpenAI-compatible server, or configure fallback_url, fallback_model, and fallback_api_key_env.", file=sys.stderr)
    sys.exit(1)


def llm_call(client, model: str, system: str, user: str, max_tokens: int = 6000,
             request_delay: float = 0.0) -> str:
    """LLM call with temperature=0.0 and 3-attempt exponential-backoff retry.

    Args:
        client: OpenAI client instance from make_llm_client().
        request_delay: seconds to sleep after a successful call (rate limiting).
                       Can also be set via ORCHESTRA_REQUEST_DELAY env var.
    """
    delay = request_delay or float(os.environ.get("ORCHESTRA_REQUEST_DELAY", "0"))
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
            text = (response.choices[0].message.content or "").strip()
            if delay > 0:
                time.sleep(delay)
            return text
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_exc}") from last_exc


def build_llm_cache_key(model: str, system: str, user: str,
                        max_tokens: int, prompt_version: str = "v1") -> str:
    """Build a stable cache key for an LLM prompt."""
    payload = {
        "model": model,
        "system": system,
        "user": user,
        "max_tokens": max_tokens,
        "prompt_version": prompt_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_llm_call(client, model: str, system: str, user: str,
                    max_tokens: int = 6000, request_delay: float = 0.0,
                    cache_namespace: str = "default", prompt_version: str = "v1",
                    use_cache: bool = True) -> str:
    """Run an LLM call with an on-disk content-addressed cache."""
    if not use_cache:
        return llm_call(
            client, model, system, user, max_tokens=max_tokens,
            request_delay=request_delay,
        )

    cache_key = build_llm_cache_key(model, system, user, max_tokens, prompt_version=prompt_version)
    cache_dir = LLM_CACHE_DIR / cache_namespace
    cache_path = cache_dir / f"{cache_key}.json"
    cached = read_json_file(cache_path, {})
    if isinstance(cached, dict) and isinstance(cached.get("response"), str):
        return cached["response"]

    response = llm_call(
        client, model, system, user, max_tokens=max_tokens,
        request_delay=request_delay,
    )
    write_json_file(cache_path, {
        "cache_key": cache_key,
        "namespace": cache_namespace,
        "prompt_version": prompt_version,
        "model": model,
        "max_tokens": max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
        "response": response,
    })
    return response


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
    return read_json_file(sf, {"processed": {}})


def save_sources(sources: dict, sources_file: Path | None = None) -> None:
    """Write _sources.json with file locking."""
    sf = sources_file or SOURCES_FILE
    write_json_file(sf, sources)


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
    locked_write_text(path, render_article_text(fm, body))


def render_article_text(fm: dict, body: str) -> str:
    """Render article content with YAML frontmatter."""
    class _Dumper(yaml.Dumper):
        pass

    def _list_representer(dumper, data):
        if data and isinstance(data[0], dict):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    _Dumper.add_representer(list, _list_representer)
    fm_out = yaml.dump(fm, Dumper=_Dumper, default_flow_style=False, allow_unicode=True).strip()
    return f"---\n{fm_out}\n---\n\n{body}"


def update_article(path: Path, updater: Callable[[dict, str], tuple[dict, str, T]]) -> T:
    """Update article frontmatter/body under a single file lock."""
    def _apply(text: str) -> tuple[str, T]:
        fm, body = split_frontmatter(text)
        new_fm, new_body, result = updater(fm, body)
        return render_article_text(new_fm, new_body), result

    return update_text_file(path, "", _apply)


def inject_metadata(path: Path, fields: dict) -> None:
    """Inject or update fields in an article's YAML frontmatter."""
    if not path.exists():
        return
    update_article(path, lambda fm, body: (fm | fields, body, None))


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


def validate_capture_response(
    data: dict,
    valid_projects: list[str] | set[str],
    default_project: str = "GENERAL",
) -> dict:
    """Validate and normalize a capture extraction payload.

    Returns a sanitized structure suitable for writing to disk.
    Raises ValueError when the payload is structurally invalid.
    """
    if not isinstance(data, dict):
        raise ValueError("capture response must be a JSON object")

    projects = data.get("projects", [])
    entries = data.get("entries", [])
    skip_reason = str(data.get("skip_reason", "") or "").strip()

    if not isinstance(projects, list):
        raise ValueError("capture response 'projects' must be a list")
    if not isinstance(entries, list):
        raise ValueError("capture response 'entries' must be a list")

    valid = set(valid_projects)
    normalized_projects: list[str] = []
    for project in projects:
        if not isinstance(project, str):
            continue
        proj = project.strip() or default_project
        if proj not in valid:
            proj = default_project
        if proj not in normalized_projects:
            normalized_projects.append(proj)

    normalized_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        project = str(entry.get("project", "") or "").strip() or default_project
        if project not in valid:
            project = default_project

        title = sanitize_content(str(entry.get("title", "") or "").strip(), max_length=200)
        content = sanitize_content(str(entry.get("content", "") or "").strip())
        trigger = sanitize_content(str(entry.get("trigger", "") or "").strip(), max_length=500)

        if not title or not content:
            continue

        normalized_entries.append({
            "project": project,
            "title": title,
            "trigger": trigger,
            "content": content,
        })
        if project not in normalized_projects:
            normalized_projects.append(project)

    if entries and not normalized_entries:
        raise ValueError("capture response contained no valid entries")

    return {
        "projects": normalized_projects,
        "skip_reason": skip_reason,
        "entries": normalized_entries,
    }


def validate_compile_plan(data: dict) -> dict:
    """Validate and normalize a compile plan before article generation starts."""
    if not isinstance(data, dict):
        raise ValueError("compile plan must be a JSON object")

    articles = data.get("articles", [])
    skipped_reason = str(data.get("skipped_reason", "") or "").strip()

    if not isinstance(articles, list):
        raise ValueError("compile plan 'articles' must be a list")

    normalized_articles = []
    valid_actions = {"create", "update"}
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError("each planned article must be an object")

        path = str(article.get("path", "") or "").strip()
        action = str(article.get("action", "") or "").strip().lower()
        title = sanitize_content(str(article.get("title", "") or "").strip(), max_length=200)

        if not path:
            raise ValueError("planned article missing required field 'path'")
        if action not in valid_actions:
            raise ValueError(f"planned article has invalid action: {action!r}")
        if not title:
            raise ValueError("planned article missing required field 'title'")

        tags = article.get("tags", [])
        sections = article.get("sections", ["Overview", "Key Points", "Connections"])
        core_concepts = article.get("core_concepts", [])

        if tags is None:
            tags = []
        if sections is None:
            sections = ["Overview", "Key Points", "Connections"]
        if core_concepts is None:
            core_concepts = []
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("planned article 'tags' must be a list of strings")
        if not isinstance(sections, list) or not all(isinstance(section, str) for section in sections):
            raise ValueError("planned article 'sections' must be a list of strings")
        if not isinstance(core_concepts, list) or not all(isinstance(term, str) for term in core_concepts):
            raise ValueError("planned article 'core_concepts' must be a list of strings")

        normalized_articles.append({
            "path": path,
            "action": action,
            "title": title,
            "summary": sanitize_content(str(article.get("summary", "") or "").strip(), max_length=500),
            "tags": [sanitize_content(tag.strip(), max_length=100) for tag in tags if tag.strip()],
            "sections": [sanitize_content(section.strip(), max_length=100) for section in sections if section.strip()],
            "core_concepts": [sanitize_content(term.strip(), max_length=100) for term in core_concepts if term.strip()],
        })

    return {
        "articles": normalized_articles,
        "skipped_reason": skipped_reason,
    }


def validate_article_content(content: str) -> str:
    """Validate article markdown before writing it to disk."""
    if not isinstance(content, str):
        raise ValueError("article content must be a string")

    cleaned = sanitize_content(content.strip(), max_length=50000)
    if not cleaned:
        raise ValueError("article content is empty")
    if not cleaned.startswith("---"):
        raise ValueError("article content must start with YAML frontmatter")

    fm, body = split_frontmatter(cleaned)
    if not fm:
        raise ValueError("article content is missing valid YAML frontmatter")
    if not body.strip():
        raise ValueError("article content body is empty")

    return cleaned


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

    locked_write_text(idx_file, "\n".join(lines))
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
            def _updater(fm: dict, body: str) -> tuple[dict, str, bool]:
                existing_links = fm.get("links", [])
                if not isinstance(existing_links, list):
                    existing_links = []
                if any(l.get("target") == source_slug for l in existing_links if isinstance(l, dict)):
                    return fm, body, False
                fm["links"] = existing_links + [{"target": source_slug, "type": inverse_type}]
                return fm, body, True

            update_article(target_path, _updater)
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
