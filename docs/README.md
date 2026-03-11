# Documentation Source Guide

The public-facing docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). All Markdown pages live in this `docs/` folder and are rendered to GitHub Pages via the workflows under `.github/workflows/`.

## Local preview

```bash
pip install -e .[dev]
mkdocs serve
```

MkDocs will watch for file changes and refresh the browser automatically.

## Adding pages

1. Create a new Markdown file under `docs/`.
2. Update `mkdocs.yml` to include the page in the navigation.
3. Run `mkdocs serve` (or `mkdocs build --strict`) to ensure the site renders.
4. Commit both the new page and the `mkdocs.yml` change so the GH Actions deployment picks it up.

## Assets

- Images: `docs/assets/images/`
- LaTeX/TikZ snippets: `docs/assets/Tex/`

Reference them with relative paths (e.g., `![caption](assets/images/foo.png)`), and commit the source assets so the static site builder can bundle them.
