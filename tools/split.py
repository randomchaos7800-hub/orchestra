#!/usr/bin/env python3
"""
Orchestra File Splitter — tools/split.py

Splits large project files by year to prevent bloat.
Files above the size threshold are split into yearly archives.

Usage:
  python3 tools/split.py                    # split all files above threshold
  python3 tools/split.py --dry-run          # show what would change
  python3 tools/split.py --threshold 256    # custom threshold in KB (default: 512)
  python3 tools/split.py --file PROJECTS    # split a specific project file only
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.common import KB_ROOT

PROJECTS_DIR = KB_ROOT / "projects"
DEFAULT_THRESHOLD_KB = 512

# Entry header pattern: ## YYYY-MM-DD - Title
ENTRY_RE = re.compile(r"^## (\d{4})-\d{2}-\d{2}\s", re.MULTILINE)


def _parse_entries(content: str) -> list[dict]:
    """Split content into entries by ## date headers. Returns [{year, start, end}]."""
    matches = list(ENTRY_RE.finditer(content))
    if not matches:
        return []

    entries = []
    for i, m in enumerate(matches):
        year = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        entries.append({"year": year, "start": start, "end": end})
    return entries


def _get_preamble(content: str) -> str:
    """Get content before the first entry (title header, description, etc.)."""
    first = ENTRY_RE.search(content)
    if first:
        return content[:first.start()]
    return content


def split_file(project_file: Path, threshold_kb: int, dry_run: bool = False) -> dict:
    """Split a project file by year if above threshold.

    Returns stats dict with counts.
    """
    stats = {"file": project_file.name, "original_kb": 0, "years": {}, "split": False}

    if not project_file.exists():
        return stats

    content = project_file.read_text(encoding="utf-8")
    size_kb = len(content.encode("utf-8")) / 1024
    stats["original_kb"] = round(size_kb, 1)

    if size_kb < threshold_kb:
        return stats

    entries = _parse_entries(content)
    if not entries:
        return stats

    # Group entries by year
    years: dict[str, list[str]] = {}
    for entry in entries:
        year = entry["year"]
        chunk = content[entry["start"]:entry["end"]]
        years.setdefault(year, []).append(chunk)

    if len(years) <= 1:
        # All entries are same year, no point splitting
        return stats

    preamble = _get_preamble(content)
    project_name = project_file.stem
    stats["split"] = True

    # Determine current year — entries from this year stay in the main file
    current_year = str(datetime.now().year)
    archive_years = sorted(y for y in years if y != current_year)

    if not archive_years:
        # Everything is current year
        return stats

    for year in archive_years:
        year_dir = PROJECTS_DIR / "archive" / year
        year_file = year_dir / f"{project_name}.md"
        year_content = preamble + "".join(years[year])
        entry_count = len(years[year])
        stats["years"][year] = entry_count

        if dry_run:
            print(f"  [DRY RUN] Would write {year_file} ({entry_count} entries)")
        else:
            year_dir.mkdir(parents=True, exist_ok=True)
            # Append if archive file already exists (idempotent over multiple runs)
            if year_file.exists():
                existing = year_file.read_text(encoding="utf-8")
                # Only write new entries not already present
                new_entries = []
                for chunk in years[year]:
                    # Use first line as identity check
                    first_line = chunk.strip().split("\n")[0]
                    if first_line not in existing:
                        new_entries.append(chunk)
                if new_entries:
                    with open(year_file, "a", encoding="utf-8") as f:
                        f.write("".join(new_entries))
                    print(f"  Appended {len(new_entries)} new entries to {year_file}")
                else:
                    print(f"  {year_file} already up to date")
            else:
                year_file.write_text(year_content, encoding="utf-8")
                print(f"  Created {year_file} ({entry_count} entries)")

    # Rewrite main file with only current year entries
    if not dry_run:
        current_entries = years.get(current_year, [])
        if current_entries:
            new_main = preamble + "".join(current_entries)
            project_file.write_text(new_main, encoding="utf-8")
            new_kb = round(len(new_main.encode("utf-8")) / 1024, 1)
            stats["years"][current_year] = len(current_entries)
            print(f"  {project_file.name}: {stats['original_kb']}K -> {new_kb}K ({len(current_entries)} entries kept)")
        else:
            # No current year entries — keep preamble only
            project_file.write_text(preamble, encoding="utf-8")
            print(f"  {project_file.name}: all entries archived")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Split large project files by year")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD_KB,
                        help=f"Size threshold in KB (default: {DEFAULT_THRESHOLD_KB})")
    parser.add_argument("--file", type=str, help="Split a specific project file (e.g., PROJECTS)")
    args = parser.parse_args()

    if not PROJECTS_DIR.exists():
        print(f"No projects/ directory found at {PROJECTS_DIR}")
        sys.exit(1)

    dry_label = " [DRY RUN]" if args.dry_run else ""
    print(f"Orchestra file splitter{dry_label}")
    print(f"Threshold: {args.threshold} KB")
    print("=" * 50)

    if args.file:
        target = PROJECTS_DIR / f"{args.file}.md"
        if not target.exists():
            print(f"File not found: {target}")
            sys.exit(1)
        files = [target]
    else:
        files = sorted(PROJECTS_DIR.glob("*.md"))

    if not files:
        print("No project files found.")
        return

    total_split = 0
    for f in files:
        size_kb = round(f.stat().st_size / 1024, 1)
        if size_kb < args.threshold:
            print(f"\n{f.name}: {size_kb}K (below threshold, skipping)")
            continue

        print(f"\n{f.name}: {size_kb}K (above {args.threshold}K threshold)")
        stats = split_file(f, args.threshold, args.dry_run)
        if stats["split"]:
            total_split += 1

    print(f"\n{'=' * 50}")
    print(f"Files split: {total_split}")


if __name__ == "__main__":
    main()
