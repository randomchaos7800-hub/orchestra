# Orchestra

Point it at your conversations. Get a knowledge base.

Orchestra turns your AI chat history into structured, searchable knowledge that maintains itself. Two components, no subscriptions, no cloud, no vector database. Markdown all the way down.

**Capture** ingests conversation exports from Claude, ChatGPT, or any JSON chat format, classifies them through an LLM, and appends structured knowledge to project files.

**Wiki** compiles research notes and extracted knowledge into a self-maintaining wiki with cross-references, an index, and health monitoring that flags stale or orphaned articles.

Both run on local inference or a cheap API key. BYO LLM — local llama-server, OpenRouter, any OpenAI-compatible endpoint. The scripts don't care what's answering as long as it speaks the API.

## What You Get

Export your chats. Drop them in a directory. Run Capture. You get structured project files — your discussions organized by domain, append-only, human-readable, searchable with grep.

Drop research notes or articles into `raw/`. Run Wiki compile. You get a knowledge base — structured articles with backlinks, a master index, source tracking, and a meta layer that tells you what's stale, what's orphaned, and what needs attention.

Set up two cron jobs and it runs itself. Every night your knowledge base grows, reorganizes, and self-monitors without you touching it.

## Why This Exists

Every AI conversation you've ever had is gone. The context you built, the decisions you made, the research you synthesized — it disappeared when you closed the tab. Chat history exists as a raw log, but logs aren't knowledge. Nobody's going back through 500 conversations to find that one architectural decision from February.

The existing solutions are either expensive (managed memory APIs, vector database subscriptions) or theoretical (research papers benchmarking against synthetic datasets). Orchestra is neither. It's two Python scripts that turn your actual conversation history into a knowledge base you own, running on hardware you control, for the cost of electricity.

This started as a personal system — built because the architect couldn't afford Obsidian subscriptions or managed knowledge services, and needed a way to capture what was being discussed across dozens of daily AI conversations. It's been running in production for months. The wiki compilation pattern is directly inspired by [Karpathy's LLM Knowledge Bases](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) (April 2026) — full credit for the architecture. The conversation capture layer and the fusion between the two are original.

## How It Works

```
Your conversations (Claude, ChatGPT, any export)
     │
     ▼
  Capture (extract.py)
  - Reads conversation exports
  - Classifies each segment via LLM
  - Appends to structured project files
  - Raw exports preserved immutably
     │
     ▼
  Project Files (PROJECTS.md, RESEARCH.md, STRATEGY.md, ...)
  - Organized by domain
  - Append-only
  - Human-readable markdown
  - Searchable with grep, ripgrep, or any text tool

Your research notes, articles, papers
     │
     ▼
  Wiki (compile.py)
  - Reads raw/ directory
  - Classifies and compiles via LLM
  - Writes structured wiki articles
  - Maintains index + cross-references
  - Flags stale/orphaned articles in meta/
     │
     ▼
  Wiki (wiki/)
  - Structured articles by concept, entity, event, tool
  - Master index (_index.md)
  - Source tracking (_sources.json)
  - Self-monitoring (meta/stale.md, meta/orphans.md, meta/suggestions.md)
```

## What You Need

- Python 3.10+ with `openai` and `httpx`
- One of:
  - A local llama-server (or any OpenAI-compatible local endpoint)
  - An OpenRouter API key
  - Any OpenAI-compatible API endpoint
- Conversation exports from your AI tools
- A directory to put research notes in (optional, for Wiki)
- Two cron jobs (optional, for automation)

That's it. No databases. No Docker. No Kubernetes. No managed services.

## Quick Start

```bash
# Clone
git clone https://github.com/vitale-dynamics/orchestra.git
cd orchestra

# Install dependencies
pip install openai httpx

# Configure your LLM endpoint
cp config/config.example.json config/config.json
# Edit config.json: set your local endpoint or API key

# Export your conversations
# Claude: Settings → Account → Export Data
# ChatGPT: Settings → Data Controls → Export Data

# Run Capture on your exports
python capture/extract.py --input /path/to/your/export/

# Drop research notes into raw/
cp your-notes/*.md wiki/raw/manual/

# Compile the wiki
python wiki/compile.py

# (Optional) Automate with cron
# 2 AM — process new conversation exports
# 0 2 * * * cd /path/to/orchestra && python capture/extract.py --input /path/to/exports/
# 9 PM — compile wiki from new raw inputs
# 0 21 * * * cd /path/to/orchestra && python wiki/compile.py
```

## Documentation

| Document | What It Covers |
|----------|---------------|
| [Capture](docs/capture.md) | Conversation ingestion — input formats, classification, project file structure, raw data preservation |
| [Wiki](docs/wiki.md) | Knowledge compilation — raw input format, compile process, article structure, self-monitoring |
| [Local Inference](docs/local-inference.md) | Running on commodity hardware — real benchmarks, configuration, what works and what doesn't |
| [Lessons Learned](docs/lessons-learned.md) | What breaks, what survives, and what we learned from months of production operation |

## Design Principles

**No subscriptions.** Everything runs on hardware you own or API calls you control. If the money stops, your knowledge base still exists as readable files on your disk.

**No databases.** Markdown files, searchable with any text tool. No schema migrations, no query languages, no managed services. Copy the directory and you've backed up everything.

**Raw data is sacred.** Conversation exports are never modified. Project files are append-only. Wiki articles are the only layer that gets rewritten, and the raw inputs that fed them are preserved. If any derived output drifts from reality, regenerate it from source.

**Local inference for background work.** Classification and compilation are structured tasks with clear success criteria. A quantized 26B model on a $300 mini PC handles them fine. Save commercial API tokens for conversations where quality is visible.

**Boring technology.** Python. Markdown. Cron. Grep. SQLite if you need it, but you probably don't. The boring choice is the one that works at 3 AM when nobody's watching and still works in five years when the trendy tool has been abandoned.

## On Economics

This system was built under real financial constraints. Not "startup budget" constraints — single income, family to support, no venture capital constraints. Every architectural decision reflects that.

Think of this less as a guide to the best possible way to build a knowledge system with unlimited resources, and more as a hitchhiker's guide — how to see the universe on a dollar a day. The patterns that survive aren't the clever ones. They're the ones you can afford to run every day.

If you're building AI tools on your own budget, the constraint-shaped decisions are probably the most useful thing here.

## On Raw Data and Drift

Every stage in this pipeline — extraction, classification, compilation — is lossy. Each step compresses, interprets, and can lose nuance. This is an inherent property of any summarization chain, and it's the central problem of AI-generated knowledge.

The safeguard: **raw data is never deleted.** Conversation exports sit in immutable batch files. Project files are append-only. Wiki raw inputs are preserved alongside compiled articles. Every derived layer is a view on top of source material that still exists. If a view drifts, regenerate it from source.

The entire reason this system was built is that AI-generated summaries and knowledge representations are unreliable. Preserving raw source material is the architectural response to that unreliability. A full re-extraction from raw to structured knowledge can be run at any time, and the results compared against what the pipeline produced incrementally.

## Supported Export Formats

**Currently supported:**
- Claude.ai conversation exports (JSON)

**Planned / community contributions welcome:**
- ChatGPT export format
- Slack export format
- Telegram export format
- Generic JSON conversation format
- Plain text / markdown conversation logs

The extraction script is format-agnostic at its core — it needs a conversation broken into messages with roles and content. Adding a new format means writing a parser that produces that structure.

## What This Is Not

This is not a memory system for AI agents. It doesn't inject knowledge into prompts, manage context windows, or provide retrieval APIs. If you need that, look at [Mem0](https://github.com/mem0ai/mem0), [Hermes Agent](https://github.com/NousResearch/hermes-agent), or [MemPalace](https://github.com/milla-jovovich/mempalace).

This is a knowledge capture and curation system for humans who use AI tools. It turns your conversation history into a reference you can search, browse, and trust — because the raw sources are always there to check against.

## Credit

The wiki compilation pattern is directly adapted from [Andrej Karpathy's LLM Knowledge Bases](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Raw sources in, LLM-compiled wiki out, structured articles with backlinks and health monitoring. Credit where it's due — we saw his architecture, built it, and extended it with automated conversation capture.

## License

MIT — take what you want.
