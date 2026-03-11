# Documentation

This folder centralises written guidance, assets, and notebook pointers for the WLS-SINDy project.

## Assets

- `assets/images/` – publication-quality figures reused across the paper and presentations.
- `assets/Tex/` – snippets (TikZ/LaTeX) shared between the paper and slides.
- `../videos/` – rendered animations referenced in the docs, grouped by source (generated experiments, presentation scenes, etc.).

## Notebooks

The table below lists the canonical notebooks for each experiment. Launch them with `jupyter lab <path>` after installing the package.

| Notebook | Description | Location |
| --- | --- | --- |
| Base Tutorial | Walkthrough of weighted weak-form regression and library construction. | `examples/base/part1.ipynb` |
| Lorenz Multi-Fidelity Forecasting | Ensemble MF workflow and comparison plots. | `docs/notebooks/lorenz/forecastingMF.ipynb` |
| Lorenz Double Pendulum Forecast | Coupled MF run mirroring the presentation figures. | `docs/notebooks/lorenz/forecastingDoublePendulum.ipynb` |
| Double Pendulum Deep Dive | Diagnostics + R² heatmaps for pendulum datasets. | `docs/notebooks/pendulum/double_pendulum.ipynb` |
| Hopf System | Heteroscedastic GLS benchmarking on the Hopf oscillator. | `docs/notebooks/hopf/hopf.ipynb` |
| Isothermal Flow | Reproduces Navier–Stokes datasets & visualisations. | `docs/notebooks/isothermal_flow/itflow.ipynb` |
| Presentation Intro | Narrative version of the talk’s introduction. | `presentations/intro.ipynb` |
| Presentation Methodology | Notebook that drives most of the talk’s figures. | `presentations/methodology.ipynb` |

## Paper & Presentations

- Build the manuscript from `paper/` with `latexmk -pdf main.tex`.
- Render slides/animations under `presentations/` using the provided Manim scripts. Video artefacts are relocated to `videos/` to keep the tree tidy.

If you add a new experiment, drop its notebooks under `examples/<domain>/` and reference them here to keep the documentation index up to date.
