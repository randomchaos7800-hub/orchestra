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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import (
    WIKI_DIR, INDEX_FILE, make_llm_client,
    cached_llm_call, locked_write_text,
)
from lib.retrieval import search as retrieve_articles


def _resolve_article_from_result_id(result_id: str) -> Path:
    """Return the wiki path for a retrieval result id."""
    return WIKI_DIR / result_id


def _append_source_paths(answer: str, source_paths: list[Path], output_format: str) -> str:
    """Append deterministic source paths to the generated answer."""
    if not source_paths:
        return answer

    rel_paths = [str(path.relative_to(WIKI_DIR)) for path in source_paths]
    if output_format == "slides":
        appendix = ["---", "## Sources", *[f"- `{path}`" for path in rel_paths]]
    else:
        appendix = ["", "## Sources", *[f"- `{path}`" for path in rel_paths]]
    return answer.rstrip() + "\n" + "\n".join(appendix) + "\n"


def _find_relevant_articles(question: str, mode: str = "lexical") -> list[Path]:
    """Find articles relevant to the question through shared retrieval."""
    hits = retrieve_articles(question, wiki_dir=WIKI_DIR, mode=mode, top_k=6)
    resolved = []
    for hit in hits:
        path = _resolve_article_from_result_id(hit["id"])
        if path.exists():
            resolved.append(path)
    return resolved


def answer_question(question: str, output_format: str = "markdown",
                    use_cache: bool = True, mode: str = "lexical") -> str:
    """Answer a question using wiki content as context."""
    client, model, _ = make_llm_client()

    if not INDEX_FILE.exists():
        return "Wiki is empty. Run `tools/compile.py` first."

    index_text = INDEX_FILE.read_text(encoding="utf-8")
    relevant = _find_relevant_articles(question, mode=mode)

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
        answer = cached_llm_call(
            client,
            model,
            system,
            user,
            max_tokens=4000,
            cache_namespace="query-answer",
            prompt_version="v1",
            use_cache=use_cache,
        )
        return _append_source_paths(answer, relevant, output_format)
    except Exception as e:
        return f"LLM error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Q&A interface for the wiki")
    parser.add_argument("question")
    parser.add_argument("--output", type=str, help="Write answer to file")
    parser.add_argument("--slides", type=str, help="Write as Marp slides to file")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the on-disk LLM cache for this run")
    parser.add_argument("--mode", choices=["lexical", "hybrid"], default="lexical",
                        help="Retrieval mode: lexical BM25 or hybrid BM25+vector RRF")
    args = parser.parse_args()

    if args.slides:
        answer = answer_question(
            args.question,
            output_format="slides",
            use_cache=not args.no_cache,
            mode=args.mode,
        )
        Path(args.slides).parent.mkdir(parents=True, exist_ok=True)
        locked_write_text(Path(args.slides), answer)
        print(f"Slides written to {args.slides}")
    elif args.output:
        answer = answer_question(args.question, use_cache=not args.no_cache, mode=args.mode)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        locked_write_text(Path(args.output), answer)
        print(f"Answer written to {args.output}")
    else:
        print(answer_question(args.question, use_cache=not args.no_cache, mode=args.mode))


if __name__ == "__main__":
    main()
