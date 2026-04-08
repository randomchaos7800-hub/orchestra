# Lessons Learned

Operational knowledge from running a conversation-to-knowledge pipeline continuously for months. Everything here cost time, money, or both.

## The Lossy Pipeline Problem

Every stage in a knowledge pipeline compresses information. Raw conversation → classified segment → project file → wiki article. Each step interprets, summarizes, and potentially loses nuance.

The failure mode isn't dramatic — it's gradual. Over weeks, the derived knowledge drifts from what was actually discussed. Summaries lose context. Classifications put things in the wrong bucket. The wiki article says something slightly different from what the conversation actually established.

**The fix:** Raw data is never deleted. Conversation exports sit in immutable batch directories. Project files are append-only. Wiki raw inputs are preserved alongside compiled articles. Every derived layer is a view on top of source material that still exists. When you suspect drift, compare the derived output against the raw source. If they diverge, regenerate from source.

This isn't a theoretical safeguard — it's been used. A project file classification was wrong, putting architecture discussions into STRATEGY.md instead of PROJECTS.md. The raw export was reprocessed with corrected classification and the project file was updated. No information was lost because the source was intact.

## LLM Classification Is Good Enough, Not Perfect

The classification step — "is this conversation about research, strategy, projects, or something else?" — works well on clear-cut cases and struggles on conversations that span multiple domains. A conversation that starts as project work, pivots to strategy, and ends with speculative research might get classified as any of the three.

**What helps:** The LLM is better at classification when conversations are shorter and more focused. Long, rambling, multi-topic conversations are harder to classify cleanly. This isn't fixable with a better model — it's a property of the input.

**What doesn't matter as much as you'd think:** A misclassified conversation still exists in a project file. It's searchable. Grep finds it regardless of which file it's in. Perfect classification would be nice; imperfect classification is survivable because the data is still there.

## Nightly Batching Beats Real-Time Processing

Processing raw inputs once per day produces better results than processing them as they arrive. Two reasons:

1. **Context:** The compile step sees all of today's inputs at once. A single paper summary in isolation is less valuable than a brief that can be compared against three other briefs from the same day and matched against existing wiki articles.

2. **Cost:** One LLM session processing 10 inputs is cheaper than 10 separate sessions processing one input each, especially on local inference where startup and context-loading overhead is significant.

The tradeoff is latency. Something discussed at 9 AM doesn't appear in the wiki until the 9 PM compile step. For a personal knowledge base, 12-hour latency is fine. If you need real-time knowledge updates, this architecture isn't the right fit.

## Markdown Scales Better Than You'd Expect

The project files grow large. One is 138K. Another is 70K. The natural assumption is that these will become unwieldy.

In practice, large markdown files work fine for years:
- Grep and ripgrep don't care about file size
- Text editors handle 100K+ files without issues
- The files are append-only, so you're always writing to the end — no random-access patterns that would benefit from a database
- Version control (git) handles text diffs natively

The point where markdown stops working is if you need structured queries — "find all conversations about X from before March" requires parsing, not grep. If that becomes a need, add a lightweight index. Don't start with a database because you think you might need one later.

## The Wiki Needs Health Monitoring

Without self-monitoring, a wiki dies slowly. Articles go stale. New articles duplicate existing concepts. Cross-references get missed. Nobody notices because each individual article looks fine in isolation.

The `meta/` directory — `stale.md`, `orphans.md`, `suggestions.md` — makes the decay visible. The compile LLM flags problems it notices during compilation. This isn't perfect (the LLM can miss things or flag false positives), but it's dramatically better than no monitoring at all.

The most useful monitor is `orphans.md`. An orphaned article — one with no inbound links — is either a new concept that hasn't been connected yet or a classification error that created a redundant article. Either way, it needs human attention. A growing orphan count is the earliest signal that the wiki is fragmenting.

## Export Formats Are Annoying

Every AI platform exports conversations differently. Claude.ai uses one JSON format. ChatGPT uses another. Slack exports are different again. Writing parsers for each format is tedious but necessary work.

The extraction script is designed to be format-agnostic at its core — it needs messages with roles and content. The format-specific parsers sit in front of that core and normalize everything into the same structure. Adding a new export format means writing one new parser function.

**The frustration:** Export formats change without warning. An AI platform updates their export structure and your parser breaks silently — it still runs, but the extracted content is wrong or incomplete. Validate exports manually after platform updates.

## Cron Is the Best Orchestrator

For a personal knowledge pipeline, cron (or systemd timers) is the right orchestration tool. Not Airflow. Not Prefect. Not a custom job scheduler. Cron.

Reasons:
- It's been running reliably on Unix systems for decades
- It survives reboots (with systemd timers)
- It has zero dependencies
- It's inspectable with one command (`crontab -l`)
- If it breaks, you know within 24 hours because the output doesn't appear

The pipeline runs two jobs: extraction at 2 AM, compilation at 9 PM. Each job is a single Python script invocation. If either fails, the other still runs. If both fail, the raw data is still there for the next successful run.

Don't add orchestration complexity until cron can't do what you need. For most personal knowledge systems, that day never comes.

## AI Coding Tools Build Things That Don't Work

The knowledge pipeline was built by an architect who can't write code, using AI coding tools for implementation. This works — but it fails in specific, dangerous ways.

AI coding tools optimize for "it runs without errors," not "it does the right thing." A script that parses a conversation export, classifies it into the wrong category, and appends the wrong content to the wrong file will run without errors. It's silently wrong.

**Verification method:** Observe the output. After every change to the extraction or compilation scripts, manually check the results. Read the project files. Read the wiki articles. Compare them against the source conversations. If the output doesn't match what you know the conversations contained, something is wrong in the implementation — regardless of what the coding tool says about its own work.

**The hardest bug to find:** The coding tool that changes something you didn't ask it to change. A fix to the classification logic that also modified the output format. A compile step improvement that silently changed how source tracking works. Always diff the changes, even if you can't read the code — the diff shows you what files were touched, which is often enough to know if something unexpected happened.
