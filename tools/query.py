#!/usr/bin/env python3
"""
Wiki Q&A -- tools/query.py

Answers natural language questions by reading the wiki and relevant articles.

Usage:
  python3 tools/query.py "What are the latest findings on local inference?"
  python3 tools/query.py --output summary.md "..."
  python3 tools/query.py --slides deck.md "..."   # Marp format
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import (
    WIKI_DIR, INDEX_FILE, get_wiki_sections, make_llm_client,
    cached_llm_call, locked_write_text,
)


def _find_relevant_articles(question: str) -> list[Path]:
    """Find articles relevant to the question by slug/keyword matching."""
    question_lower = question.lower()
    relevant = []

    for section in get_wiki_sections():
        section_dir = WIKI_DIR / section
        if not section_dir.exists():
            continue
        for md_file in section_dir.rglob("*.md"):
            slug = md_file.stem.lower()
            title_words = slug.replace("-", " ")
            if title_words in question_lower or any(
                w in question_lower for w in slug.split("-") if len(w) > 4
            ):
                relevant.append(md_file)

    # Fallback: keyword search in file content
    if not relevant:
        words = [w for w in re.findall(r'\w+', question_lower) if len(w) > 4]
        for section in get_wiki_sections():
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

    return list(set(relevant))[:6]


def answer_question(question: str, output_format: str = "markdown",
                    use_cache: bool = True) -> str:
    """Answer a question using wiki content as context."""
    client, model, _ = make_llm_client()

    if not INDEX_FILE.exists():
        return "Wiki is empty. Run `tools/compile.py` first."

    index_text = INDEX_FILE.read_text(encoding="utf-8")
    relevant = _find_relevant_articles(question)

    context_parts = [f"## WIKI INDEX\n{index_text[:3000]}"]
    for path in relevant:
        try:
            content = path.read_text(encoding="utf-8")
            context_parts.append(f"## ARTICLE: {path.relative_to(WIKI_DIR)}\n{content[:2000]}")
        except Exception:
            pass

    format_instructions = {
        "markdown": "Answer in clean markdown. Cite article names and dates.",
        "slides": "Answer as a Marp slide deck. Use `---` between slides. Keep each slide to 5 bullets max.",
    }.get(output_format, "Answer in clean markdown.")

    system = (
        "You are a research assistant with access to a curated knowledge base wiki. "
        "Answer using ONLY the provided wiki articles. "
        "If the wiki doesn't cover the question, say so directly."
    )

    user = f"QUESTION: {question}\n\n{format_instructions}\n\nKNOWLEDGE BASE:\n{chr(10).join(context_parts)}"

    try:
        return cached_llm_call(
            client,
            model,
            system,
            user,
            max_tokens=4000,
            cache_namespace="query-answer",
            prompt_version="v1",
            use_cache=use_cache,
        )
    except Exception as e:
        return f"LLM error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Q&A interface for the wiki")
    parser.add_argument("question")
    parser.add_argument("--output", type=str, help="Write answer to file")
    parser.add_argument("--slides", type=str, help="Write as Marp slides to file")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the on-disk LLM cache for this run")
    args = parser.parse_args()

    if args.slides:
        answer = answer_question(args.question, output_format="slides", use_cache=not args.no_cache)
        Path(args.slides).parent.mkdir(parents=True, exist_ok=True)
        locked_write_text(Path(args.slides), answer)
        print(f"Slides written to {args.slides}")
    elif args.output:
        answer = answer_question(args.question, use_cache=not args.no_cache)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        locked_write_text(Path(args.output), answer)
        print(f"Answer written to {args.output}")
    else:
        print(answer_question(args.question, use_cache=not args.no_cache))


if __name__ == "__main__":
    main()
