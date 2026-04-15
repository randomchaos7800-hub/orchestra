#!/usr/bin/env python3
"""
Wiki Q&A -- tools/query.py

Answers natural language questions by reading the wiki index and relevant articles.

LLM configuration loaded from config/config.json.

Usage:
  python3 tools/query.py "What are the latest findings on local inference?"
  python3 tools/query.py --output outputs/reports/summary.md "..."
  python3 tools/query.py --slides outputs/slides/deck.md "..."   # Marp format
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

KB_ROOT = Path(__file__).parent.parent
WIKI_DIR = KB_ROOT / "wiki"
CONFIG_DIR = KB_ROOT / "config"
INDEX_FILE = WIKI_DIR / "_index.md"


def _load_llm_config() -> dict:
    """Load LLM settings from config/config.json."""
    config_path = CONFIG_DIR / "config.json"
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("llm", {})


def _make_client():
    """Return (client, model) using config/config.json settings."""
    import httpx
    llm_cfg = _load_llm_config()

    local_url = llm_cfg.get("local_url", "http://127.0.0.1:8081/v1")
    local_model = llm_cfg.get("local_model", "gemma4")

    try:
        r = httpx.get(local_url.replace("/v1", "/health"), timeout=2)
        if r.status_code == 200:
            return OpenAI(base_url=local_url, api_key="local"), local_model
    except Exception:
        pass

    fallback_url = llm_cfg.get("fallback_url", "")
    fallback_model = llm_cfg.get("fallback_model", "")
    api_key_env = llm_cfg.get("fallback_api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "")

    if fallback_url and api_key:
        return OpenAI(base_url=fallback_url, api_key=api_key), fallback_model

    print("No LLM available.", file=sys.stderr)
    sys.exit(1)


def _find_relevant_articles(question: str, index_text: str) -> list[Path]:
    """Naive relevance: articles whose slug/title appears in the question or index matches."""
    question_lower = question.lower()
    relevant = []
    for section in ["concepts", "entities", "events", "research"]:
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            continue
        for md_file in section_dir.rglob("*.md"):
            slug = md_file.stem.lower()
            title_words = slug.replace("-", " ")
            if title_words in question_lower or any(w in question_lower for w in slug.split("-") if len(w) > 4):
                relevant.append(md_file)

    # If nothing matched by slug, fall back to keyword search in index
    if not relevant:
        words = [w for w in re.findall(r'\w+', question_lower) if len(w) > 4]
        for section in ["concepts", "entities", "events", "research"]:
            section_dir = WIKI_DIR / section
            if not section_dir.exists():
                continue
            for md_file in section_dir.rglob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8").lower()
                    if sum(1 for w in words if w in text) >= 2:
                        relevant.append(md_file)
                except Exception:
                    pass

    return list(set(relevant))[:6]  # cap at 6 articles to stay within context


def answer_question(question: str, output_format: str = "markdown") -> str:
    """Answer a question using wiki content as context."""
    client, model = _make_client()

    if not INDEX_FILE.exists():
        return "Wiki is empty. Run `tools/compile.py` first."

    index_text = INDEX_FILE.read_text(encoding="utf-8")
    relevant_articles = _find_relevant_articles(question, index_text)

    # Build context
    context_parts = [f"## WIKI INDEX\n{index_text[:3000]}"]
    for article_path in relevant_articles:
        try:
            content = article_path.read_text(encoding="utf-8")
            rel = article_path.relative_to(WIKI_DIR)
            context_parts.append(f"## ARTICLE: {rel}\n{content[:2000]}")
        except Exception:
            pass

    context = "\n\n---\n\n".join(context_parts)

    format_instructions = {
        "markdown": "Answer in clean markdown. Use headers, bullets, and bold for structure. Be specific -- cite article names and dates.",
        "slides": "Answer as a Marp slide deck. Use `---` between slides. First slide: title. Keep each slide to 5 bullets max. End with a summary slide.",
    }.get(output_format, "Answer in clean markdown.")

    system = (
        "You are a research assistant with access to a curated knowledge base wiki. "
        "Answer the question using ONLY the information in the provided wiki articles. "
        "If the wiki doesn't cover the question, say so directly -- do not fill in from general knowledge. "
        "Cite specific articles when making claims."
    )

    user = f"""QUESTION: {question}

{format_instructions}

KNOWLEDGE BASE CONTENT:
{context}"""

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"LLM error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Q&A interface for the wiki")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("--output", type=str, help="Write answer to file instead of stdout")
    parser.add_argument("--slides", type=str, help="Write as Marp slides to file")
    args = parser.parse_args()

    if args.slides:
        answer = answer_question(args.question, output_format="slides")
        Path(args.slides).parent.mkdir(parents=True, exist_ok=True)
        Path(args.slides).write_text(answer, encoding="utf-8")
        print(f"Slides written to {args.slides}")
    elif args.output:
        answer = answer_question(args.question, output_format="markdown")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(answer, encoding="utf-8")
        print(f"Answer written to {args.output}")
    else:
        answer = answer_question(args.question)
        print(answer)


if __name__ == "__main__":
    main()
