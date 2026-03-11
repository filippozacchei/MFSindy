# MF-SINDy Documentation

Welcome to the documentation hub for MF-SINDy. The content is authored in Markdown under `docs/` and rendered via [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Start here for installation, tutorials, and automation notes.

## Quickstart

1. **Clone & install**
   ```bash
   git clone https://github.com/filippozacchei/2025_visiting.git
   cd 2025_visiting
   python -m venv .venv && source .venv/bin/activate
   pip install -e .[dev]
   ```
2. **Launch notebooks** – run `jupyter lab examples/<case>/part1.ipynb` (multi-trajectory GLS) or `part2.ipynb` (heteroscedastic GLS).
3. **Cache outputs** – figures/video exports land in `examples/<case>/results/` (git-ignored) so experiments remain reproducible without polluting the repo.

## Base Tutorials (single-shot Part 1/Part 2)

Use the base notebooks for a guided walkthrough that skips the heavy Monte Carlo loops:

- `examples/base/part1.ipynb` – multi-trajectory GLS with trajectory-wise weights.
- `examples/base/part2.ipynb` – heteroscedastic GLS along a single trajectory.

Both cover trajectory generation (`examples/base/utils.py`), weak regression assembly, and ensemble forecasting hooks.

## Automation via GitHub Actions

- `.github/workflows/docs.yml` builds the MkDocs site (strict mode) and runs sanity checks on pull requests.
- `.github/workflows/mkdocs-deploy.yml` publishes the static site to GitHub Pages on pushes to `main`.

To preview locally:

```bash
pip install -e .[dev]
mkdocs serve
```

MkDocs will watch files and provide a hot-reload server at `http://127.0.0.1:8000/`.
