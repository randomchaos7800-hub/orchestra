# Wiki Style Guide

Formatting conventions for all articles in `wiki/`.

---

## Files

- Filenames are lowercase, hyphenated slugs: `ai-emergence.md`, `theory-of-mind.md`
- No spaces, no uppercase, no underscores in filenames
- Keep slugs short but unambiguous: `anthropic.md` not `anthropic-the-company.md`

## Sections

Standard section order:
1. Frontmatter (YAML)
2. `# Title` (H1, matches frontmatter title)
3. **Bold one-sentence lead** immediately after H1
4. `## Overview`
5. `## Key Claims`
6. `## Connections`
7. `## Sources`

Optional sections (insert before Connections if used):
- `## Recent Developments` — for fast-moving topics
- `## Open Questions` — for research threads where the answer is not yet known
- `## Against` — for topics where significant counterarguments exist

## Prose Style

- Dense, not verbose. Every sentence carries information.
- Present tense for standing claims, past tense for specific events.
- No hedging filler: not "it could be argued that..." — just state the claim.
- Attribute claims with source and date where known.
- Dates always ISO format: `2026-04-03`

## Backlinks

Use typed wikilinks: `[[type:slug]]`

| Type | Use when |
|------|----------|
| `references` | General mention or citation (default for bare `[[slug]]`) |
| `depends_on` | This article's subject functionally requires the target |
| `extends` | Builds directly on or evolves from the target |
| `contradicts` | Claims in this article conflict with the target |
| `related` | Thematically connected, no directional relationship |

- Only link on first meaningful use in a section, not every occurrence
- The Connections section lists all intentional cross-references with brief rationale
- Bare `[[slug]]` is valid and treated as `references`

## Code and Technical Content

- Inline code: `backticks`
- Blocks: fenced with language tag
- Model names exactly as they appear in official sources: `gemma-4-26B-A4B-it-Q4_K_M`

## Numbers and Units

- Token counts: `32K`, `128K`, `4B`, `26B` (no spaces before unit)
- Prices: `$0.003/1K tokens`
- Dates: `2026-04-03` (ISO), never "April 3rd"

## What Not to Do

- No H2 headers that are just one word ("Background", "Summary")
- No bullet lists of 10+ items — group into subsections instead
- No "As an AI language model..." self-references in compiled content
- No placeholder text like "[TO BE FILLED]" — if you don't have the content, skip the section
