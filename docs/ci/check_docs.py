"""Sanity checks for docs/index.md."""

from __future__ import annotations

from pathlib import Path

REQUIRED_HEADINGS = [
    "## Quickstart",
    "## Base Tutorials",
    "## Automation via GitHub Actions",
]


def main() -> None:
    index = Path(__file__).resolve().parent.parent / "index.md"
    text = index.read_text(encoding="utf8")

    missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required sections in docs/index.md: {joined}")

    print("Documentation headings check passed.")


if __name__ == "__main__":
    main()
