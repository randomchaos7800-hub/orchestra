# Capture

Capture turns conversation exports into structured, classified knowledge files. It reads your AI chat history, runs each conversation segment through an LLM for classification, and appends the result to the appropriate project file.

## What It Does

You export your conversations from Claude.ai (or any supported format). Capture reads the export, identifies distinct conversations, classifies each one by domain, and appends a structured summary to the matching project file.

The result: your months of AI conversations, organized by topic, in human-readable markdown files you can search with grep.

## Input

Capture expects a directory containing conversation exports. Currently supported:

**Claude.ai exports:**
Drop your export into a batch directory:
```
daily/
└── batch-20260407/
    ├── conversations.json    # The main export file
    ├── projects.json         # Project metadata (if present)
    ├── memories.json         # Memory data (if present)
    └── users.json            # User data (if present)
```

The `conversations.json` file contains the full conversation history. Capture reads it, processes each conversation, and classifies it.

## LLM Configuration

Capture uses a two-tier inference strategy:

1. **Local first:** Tries to connect to a local llama-server at `127.0.0.1:8081/v1`. If it responds, all classification runs locally at zero cost.
2. **API fallback:** If local inference is unavailable, falls back to OpenRouter (or any OpenAI-compatible endpoint) using the API key from your environment or config.

The classification task is structured and well-defined — it doesn't need a frontier model. A quantized 26B model running locally handles it as well as a commercial API for this specific task.

**Environment variables:**
```bash
OPENROUTER_API_KEY=your-key-here  # Fallback API key
```

## Output

Capture writes to a `projects/` directory. Each domain gets its own file:

```
projects/
├── PROJECTS.md      # Project-specific knowledge
├── GENERAL.md       # Cross-cutting insights
├── STRATEGY.md      # Strategic decisions and rationale
├── RESEARCH.md      # Research findings and analysis
├── ETHICS.md        # Ethics discussions
├── SPECULATIVE.md   # Exploratory and theoretical work
└── ...              # Additional domain files as needed
```

Each file is append-only. Capture classifies a conversation segment, determines which file it belongs in, and appends it. Nothing is overwritten. Nothing is deleted.

Files grow over time. That's intentional. A project file is a living document — the full record of everything discussed in that domain, in chronological order. When files get large (100K+), they're still searchable with standard text tools. Grep doesn't care about file size.

## Raw Data Preservation

Conversation exports are never modified. The original `conversations.json` files stay in their batch directories untouched. If Capture's classification was wrong — if it put a research discussion into STRATEGY.md instead of RESEARCH.md — the original conversation is always available for reprocessing.

This is the ground truth guarantee: every derived file can be regenerated from the raw exports.

## Classification

The LLM classifies each conversation by reading its content and determining the best-fit domain. Classification categories map directly to output files:

- Project work → PROJECTS.md
- Research findings → RESEARCH.md
- Strategic discussions → STRATEGY.md
- General insights → GENERAL.md
- Ethics topics → ETHICS.md
- Speculative/theoretical → SPECULATIVE.md

The classification prompt is in the extraction script. It's tunable — if you need different categories for your use case, modify the prompt and add the corresponding output files.

## Automation

Run Capture on a schedule to process new exports automatically:

```bash
# Cron: 2 AM daily
0 2 * * * cd /path/to/orchestra && python capture/extract.py --input /path/to/exports/latest/
```

For continuous capture from messaging platforms (Telegram, Slack), you'd write a relay script that forwards messages to the same project file format. The extraction script itself is batch-oriented — it processes completed conversations, not live streams.

## Sync

The project files can be synced to cloud storage for backup and cross-device access:

```bash
# Example: rclone to Google Drive every 30 minutes
*/30 * * * * rclone sync /path/to/orchestra/projects/ gdrive:orchestra/projects/
```

This is optional. The files are local by default. Sync is your choice, your provider, your schedule.

## What Capture Doesn't Do

**It doesn't summarize.** It classifies and appends. The full content of each conversation segment goes into the project file, not a summary. Summarization is lossy — Capture preserves the detail.

**It doesn't deduplicate.** If you run Capture twice on the same export, you'll get duplicate entries. Track which batches have been processed (the script handles this) or deduplicate manually.

**It doesn't inject into agent prompts.** Capture is a knowledge capture tool, not a retrieval system. The project files are for humans to search and reference. If you want to build retrieval on top of them, that's a separate layer.

**It doesn't require the Wiki.** Capture runs independently. You can use Capture without Wiki — you get organized project files without the compiled knowledge base. The two components are fully decoupled.
