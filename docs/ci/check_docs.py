"""Sanity checks for docs/README.md."""

from __future__ import annotations

from pathlib import Path

REQUIRED_HEADINGS = [
    "## Quickstart",
    "## Examples & Notebooks",
    "## Base Tutorials",
    "## API Reference",
    "## Methodology Snapshot",
    "## Automation via GitHub Actions",
]


def main() -> None:
    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text(encoding="utf8")

    missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required sections in docs/README.md: {joined}")

    print("Documentation headings check passed.")


if __name__ == "__main__":
    main()
