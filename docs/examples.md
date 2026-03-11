# Examples & Notebooks

Every case lives in `examples/<case>/` with three artefacts: `part1.ipynb`, `part2.ipynb`, and `utils.py`. The table below lists the canonical entry points.

| Case | Part 1 (multi-trajectory GLS) | Part 2 (heteroscedastic GLS) |
| --- | --- | --- |
| Base diffusion tutorial | `examples/base/part1.ipynb` | `examples/base/part2.ipynb` |
| Lorenz | `examples/lorenz/part1.ipynb` | `examples/lorenz/part2.ipynb` |
| Burgers | `examples/burgers/part1.ipynb` | `examples/burgers/part2.ipynb` |
| Hopf | `examples/hopf/part1.ipynb` | `examples/hopf/part2.ipynb` |
| Pendulum | `examples/pendulum/part1.ipynb` | `examples/pendulum/part2.ipynb` |
| Isothermal flow | `examples/isothermal_flow/part1.ipynb` | `examples/isothermal_flow/part2.ipynb` |

Each notebook follows the same storyline:

1. Introduce the governing equations and reference trajectories.
2. Fit HF/LF/MF/MF\_w ensemble models via the shared experiment utilities.
3. Visualise coefficient MAE/support errors with `mfsindy.plots.bubble_hist`.

> **Tip:** keep large outputs under `examples/<case>/results/`; those folders are git-ignored so repeated runs stay clean.
