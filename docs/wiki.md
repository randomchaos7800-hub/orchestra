# Wiki

Wiki compiles raw research notes, article summaries, and extracted knowledge into a structured, self-maintaining knowledge base. Drop inputs into `raw/`, run the compile step, and get organized wiki articles with cross-references, an index, and health monitoring.

The compilation pattern is adapted from [Andrej Karpathy's LLM Knowledge Bases](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Raw sources go in, the LLM compiles structured articles, and the wiki compounds in value over time.

## What It Does

The compile step reads everything in `raw/`, runs it through an LLM, and produces structured wiki articles in `wiki/`. The LLM decides whether each input updates an existing article or creates a new one, writes the article, generates backlinks to related concepts, and updates the master index.

The result: a growing, organized knowledge base that maintains itself.

## Structure

```
wiki/
├── config/
│   ├── compile-rules.md          # System prompt for the compile LLM
│   └── wiki-style.md             # Style and formatting guide
├── raw/                          # Inputs (you write here)
│   ├── manual/                   # Hand-dropped notes, articles, papers
│   └── [agent-name]/             # Automated drops (if using agents)
├── wiki/                         # Compiled output (LLM writes here)
│   ├── _index.md                 # Master index of all articles
│   ├── _sources.json             # Source tracking (prevents reprocessing)
│   ├── concepts/                 # Concept articles
│   ├── entities/                 # Technology/tool/person profiles
│   ├── events/                   # Time-based entries
│   ├── tools/                    # Tool evaluations and guides
│   └── meta/                     # Self-monitoring
│       ├── suggestions.md        # Compile LLM's improvement ideas
│       ├── orphans.md            # Articles with no inbound links
│       └── stale.md              # Articles flagged as potentially outdated
└── tools/
    ├── compile.py                # Main compile script
    ├── query.py                  # Search interface
    ├── search.py                 # Full-text search
    └── healthcheck.py            # Wiki health monitoring
```

## Raw Inputs

The `raw/` directory is where knowledge enters the system. Two input modes:

**Manual drops:** Put any markdown file in `raw/manual/`. Research notes, article summaries, paper analyses, meeting notes, whatever you want compiled into the wiki.

**Automated drops:** If you have agents or scripts that produce research briefs, point them at a subdirectory under `raw/`. The compile step processes everything in `raw/` regardless of which subdirectory it's in.

### Input Format

The compile step works best with structured markdown that includes YAML frontmatter:

```markdown
---
type: research-brief
date: 2026-04-07
source: arXiv
---

**Topic** — What it is → why it matters → Actionable? Yes/No

**Another topic** — Description → significance → Actionable? Yes/No
```

But it also handles plain markdown, unstructured notes, and raw article text. The LLM adapts to whatever it receives. Structured input produces more consistent output.

For full paper ingestion, the raw file can include the complete article text in markdown. The compile step will extract key concepts, create or update relevant wiki articles, and link them to the source.

## Compilation

The compile script (`compile.py`) does the following:

1. **Reads `_sources.json`** to determine which raw files have already been processed
2. **Reads all new files** in `raw/`
3. **Sends each file to the LLM** with the system prompt from `config/compile-rules.md`
4. **The LLM classifies** the input and determines whether it updates an existing article or creates a new one
5. **Writes articles** to the appropriate directory under `wiki/`
6. **Updates `_index.md`** with new article entries
7. **Updates `_sources.json`** to mark processed files
8. **Writes to `meta/`** — flags stale articles, orphaned pages, and improvement suggestions

### LLM Configuration

Same two-tier strategy as Capture:

1. **Local first:** Tries `127.0.0.1:8081/v1`
2. **API fallback:** Uses OpenRouter or any OpenAI-compatible endpoint via API key

The compile task requires more reasoning than classification — the LLM needs to read existing articles, decide how new information relates, and write coherent updates. A 26B local model handles this adequately for most inputs. Larger or commercial models produce slightly better cross-referencing and more nuanced article updates.

## Self-Monitoring

The `meta/` directory is what keeps the wiki alive long-term:

**`stale.md`** — Articles that reference information the compile LLM suspects may be outdated. Flagged during compilation when new inputs contradict or supersede existing article content.

**`orphans.md`** — Articles with no inbound links from other articles. An orphan isn't necessarily a problem — it might be a new topic that hasn't been connected yet. But a growing orphan count suggests the wiki is fragmenting.

**`suggestions.md`** — The compile LLM's ideas for structural improvements. New articles that should exist, existing articles that should be merged, cross-references that are missing.

This is the layer that separates a wiki from a pile of files. Without self-monitoring, a knowledge base dies slowly — articles go stale, new articles duplicate existing ones, connections get missed. The meta layer makes the decay visible.

## Automation

Run the compile step on a schedule:

```bash
# Cron: 9 PM daily
0 21 * * * cd /path/to/orchestra && python wiki/compile.py
```

Why 9 PM: if you're dropping notes throughout the day, or have agents generating research briefs in the morning, the evening compile step sees all of the day's inputs at once. Batching lets the LLM make better classification decisions than processing files one at a time.

## Search

Four search tools included, at different complexity levels:

**`tools/search.py`** — full-text regex search, no extra dependencies:
```bash
python tools/search.py "knowledge graph"
python tools/search.py --tag agents "memory"
python tools/search.py --section concepts "attention"
```

**`tools/search_hybrid.py`** — hybrid BM25 + vector search with Reciprocal Rank Fusion. Finds semantically similar articles that don't share exact keywords. Incremental ChromaDB index, updates only changed files on each run.
```bash
python tools/search_hybrid.py "transformer attention"
python tools/search_hybrid.py --reindex   # force full reindex
python tools/search_hybrid.py --stats     # show index state
```

**`tools/suggest_links.py`** — finds articles that should be linked but aren't, by querying stored embeddings for similarity above a threshold. Writes `wiki/meta/suggestions.md`.
```bash
python tools/suggest_links.py
python tools/suggest_links.py --threshold 0.7   # stricter
python tools/suggest_links.py --dry-run         # preview only
```

**`tools/query.py`** — natural language Q&A against the wiki using LLM:
```bash
python tools/query.py "what do we know about transformer attention?"
python tools/query.py --slides "overview of agentic architectures"
```

For most uses, grep and ripgrep work fine. `search.py` is the zero-dependency option. `search_hybrid.py` and `suggest_links.py` are worth the extra setup once the wiki is large enough that you're missing connections.

### Search dependency tradeoff

`search_hybrid.py` requires `chromadb` and `sentence-transformers`. The problem: `sentence-transformers` depends on PyTorch, which is a 1-2GB install. That contradicts the "boring technology, no managed services" premise.

The known lighter alternative is [`fastembed`](https://github.com/qdrant/fastembed) (by Qdrant), which uses ONNX Runtime instead of PyTorch. Total install is ~100MB instead of ~2GB, and ChromaDB supports it natively via `FastEmbedEmbeddingFunction`. We haven't validated the switch yet — changing the embedding function invalidates any existing index and requires a full reindex, and we want to verify model quality is comparable before making it the default. The direction is clear, the timeline isn't. Once validated, the change is two lines in `_get_collection()` and a `requirements.txt` update.

If the PyTorch weight is a problem today, `tools/search.py` with `--tag` and `--section` filtering covers most use cases.

## Raw Data Preservation

Raw inputs in `raw/` are never modified by the compile step. The compiled articles in `wiki/` are the only output. If an article is wrong — if the LLM misclassified something or merged two concepts that should be separate — the raw input is always available for reprocessing.

`_sources.json` tracks which raw files have been compiled. To reprocess a file, remove its entry from `_sources.json` and run compile again.

## What Wiki Doesn't Do

**It doesn't capture conversations.** That's Capture's job. Wiki compiles research and notes into structured knowledge. Use Capture for chat history, Wiki for reference material. Use both together for a complete knowledge pipeline.

**It doesn't do retrieval augmentation.** Wiki doesn't inject articles into AI prompts or provide a retrieval API. It builds a human-readable reference. If you want to build RAG on top of it, the articles are markdown files — easy to embed and index.

**It doesn't require Capture.** Wiki runs independently. You can use Wiki without Capture — just drop files into `raw/` manually. The two components are fully decoupled.

## The Fusion

When you run both Capture and Wiki together, you get a complete knowledge pipeline:

1. **Conversations** are captured and classified by Capture into project files
2. **Research notes** are compiled by Wiki into structured articles
3. **Both** preserve raw sources for verification
4. **Both** run on the same LLM endpoint (local or API)
5. **Both** automate via cron on the same schedule

The project files from Capture and the wiki articles from Wiki serve different purposes — Capture is the institutional memory (what was discussed and decided), Wiki is the reference library (what is known and curated). Together they cover the full knowledge lifecycle: from conversation to capture to curation.
