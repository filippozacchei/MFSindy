# WLS-SINDy

Weighted Least Squares Sparse Identification of Nonlinear Dynamics (WLS-SINDy) extends the weak-form SINDy framework with heteroscedastic noise models, multi-fidelity training data, and GLS-style whitening. This repository houses the research code, documentation, and media that accompany the ongoing paper stored in `paper/main.tex` and `paper/main.pdf`.

The project follows the US-RSE recommendations for research software: a `src/` package installable with `pip`, reproducible experiments in `examples/`, versioned documentation, and automated quality gates via pre-commit + nox.

## Repository Layout

```
.
├── src/mfsindy          # installable Python package (pip/pyproject)
├── examples               # GLS / WLS experiments & notebooks
├── docs                   # documentation entry point + assets
├── paper                  # LaTeX sources for the manuscript
├── presentations          # slide decks, figures, and scripts
├── videos                 # rendered animations and raw movie assets
├── pyproject.toml         # packaging metadata
├── requirements.txt       # pinned runtime dependencies (mirrors pyproject)
└── noxfile.py, .pre-commit-config.yaml (added below)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]   # installs mfsindy plus dev tooling
# or: pip install -r requirements.txt for a minimal runtime env
```

The package exposes reusable case modules (`mfsindy.cases.*`), plotting helpers, and the custom `WeightedWeakPDELibrary` implementation.

## Documentation & Research Assets

- `docs/README.md` is the canonical documentation hub (quickstart, notebook index, tutorials, API reference, and methodology snapshot). A dedicated GitHub Action (`.github/workflows/docs.yml`) checks that those sections stay present on every push/PR.
- `paper/` contains the LaTeX sources (`main.tex`, `abstract.tex`, `result.tex`, etc.). Use `latexmk -pdf main.tex` from inside `paper/` to build the manuscript.
- `presentations/` stores slide decks, Manim scripts, and supporting figures.
- `videos/` centralises all rendered animations (presentations, generated scenes, and raw partials) so large binaries stay out of the core package.

## Examples

Each case now lives under `examples/<case>/` with exactly three touch points (`part1.ipynb`, `part2.ipynb`, `utils.py`). Typical workflow:

```bash
cd examples/lorenz
jupyter lab part1.ipynb   # multi-trajectory weighting scenario
jupyter lab part2.ipynb   # heteroskedastic GLS scenario
```

Use `part1.ipynb` for the multi-trajectory weighting scenario and `part2.ipynb` for the heteroskedastic run (Burgers, Hopf, Lorenz, pendulum, isothermal flow, and the base diffusion tutorial).

## Development Workflow

1. **Automation**: `nox` sessions (`lint`, `tests`, `docs`, etc.) encapsulate repeatable checks. Run `nox -s lint` before pushing.
2. **Pre-commit**: Install hooks via `pre-commit install` to lint staged files (ruff, black, end-of-file fixes, YAML formatting).
3. **Coding style**: follow PEP 8/pyproject formatting, keep notebooks in `docs/notebooks` or `examples/*/*.ipynb`, and store figures/videos under `docs/assets` or `videos/`.
4. **Data**: large simulation outputs belong in `examples/**/results/` (git-ignored) or external storage. Only commit configuration + lightweight references.

For additional context on the scientific motivation, see `paper/main.pdf` once compiled.
