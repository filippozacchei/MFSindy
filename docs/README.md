# MF-SINDy Documentation

This README is the single entry point for the project documentation. Everything lives next to the code (no external site yet) and will be rendered by a lightweight GitHub Action so we can keep the workflow simple.

## Quickstart

1. **Clone + install**
   ```bash
   git clone <repo> && cd 2025_visiting
   python -m venv .venv && source .venv/bin/activate
   pip install -e .[dev]
   ```
2. **Launch notebooks** – run `jupyter lab examples/<case>/part1.ipynb` (multi-trajectory GLS) or `part2.ipynb` (heteroscedastic GLS).
3. **Cache assets** – figures/video exports land in `examples/<case>/results/` (git-ignored) so you can re-run experiments locally without touching tracked files.

## Examples & Notebooks

| Case | Part 1 (multi-trajectory GLS) | Part 2 (heteroscedastic GLS) |
| --- | --- | --- |
| Base diffusion tutorial | `examples/base/part1.ipynb` | `examples/base/part2.ipynb` |
| Lorenz | `examples/lorenz/part1.ipynb` | `examples/lorenz/part2.ipynb` |
| Burgers | `examples/burgers/part1.ipynb` | `examples/burgers/part2.ipynb` |
| Hopf | `examples/hopf/part1.ipynb` | `examples/hopf/part2.ipynb` |
| Pendulum | `examples/pendulum/part1.ipynb` | `examples/pendulum/part2.ipynb` |
| Isothermal flow | `examples/isothermal_flow/part1.ipynb` | `examples/isothermal_flow/part2.ipynb` |

Each notebook follows the same pattern: (i) show the governing equations and reference trajectories, (ii) fit the HF/LF/MF models, (iii) visualise MAE and support errors via the shared `mfsindy.plots.bubble_hist` helper.

## Base Tutorials (single-shot Part 1/Part 2)

`examples/base/part1.ipynb` (multi-trajectory) and `examples/base/part2.ipynb` (heteroscedastic) act as step-by-step tutorials. They instantiate a single trajectory set instead of the full Monte Carlo experiment so you can inspect:

- HF/LF trajectory generation (`examples/base/utils.py`)
- Weighted weak regression assembly (`mfsindy.experiments.multi_trajectory`)
- Forecasting hooks (ensemble rollout plots)

Use these two notebooks when writing the README tutorial or presenting the workflow.

## API Reference (selected modules)

- **Experiments** (`mfsindy.experiments`): `run_multi_trajectory_gls_experiment`, `run_intra_trajectory_gls_experiment`, and the accompanying dataclasses (`MultiTrajectoryGLSData`, `IntraTrajectoryGLSData`).
- **Plots** (`mfsindy.plots`): currently exposes `bubble_hist` for consistent bubblegrams across all Part 1 notebooks; more shared visuals will migrate here.
- **Weighted weak library** (`mfsindy.weighted_weak_pde_library.WeightedWeakPDELibrary`): builds GLS-aware weak libraries compatible with PySINDy feature stacks.

Additional helper modules (case generators, training, etc.) will be documented once they stabilise.

## Methodology Snapshot

MF–SINDy augments weak-form + ensemble SINDy with explicit fidelity modeling:

1. **Goal & overview** – treat fidelity as an input by estimating noise variance per trajectory/time step, then propagate it through covariance-aware whitening before sparse regression.
2. **Stage 1 (fidelity annotation)** – collect LF/HF trajectories and estimate either trajectory-wise noise levels (Regime I) or time-varying variances (Regime II). These yield heteroscedastic variance profiles.
3. **Stage 2 (weak GLS weighting)** – assemble the weak system \((\mathbf{b}, \mathbf{G})\) using test functions, model the induced weak residual covariance, and whiten the system with \(W^\top W = \Sigma^{-1}\) so sequential thresholding works with GLS weights.
4. **Stage 3 (ensemble & forecasting)** – bag rows of the whitened system, fit sparse ensembles, and propagate them to obtain forecast bands.
5. **Key novelty** – the weak-space covariance model captures heteroscedastic noise and makes whitening part of the solver rather than a preprocessing hack.

Two regimes are especially relevant:

- **Trajectory-wise homogeneous noise** – each trajectory \(k\) has variance \(\sigma_k^2\). The weak covariance is block diagonal with blocks \(\sigma_k^2 \Sigma_0\) (\(\Sigma_0 = V'(V')^\top\)), leading to a simple block whitening matrix \(W=\mathrm{diag}(\sigma_k^{-1}\Sigma_0^{-1/2})\).
- **Trajectory-wise heterogeneous noise** – variance varies over time; covariance becomes \(V' D_\sigma (V')^\top\), generally dense. Whitening uses any \(W\) with \(W^\top W = \Sigma^{-1}\) (e.g., Cholesky of \(\Sigma\)). Stacked trajectories keep the block structure but each block uses its own \(D_{\sigma^{(k)}}\).

Variance can be estimated by comparing noisy trajectories to smoothed versions and squaring the residuals (Savitzky–Golay or local polynomial smoothers). Use absolute estimates when calibrated or relative weights otherwise.

![MF–SINDy methodology schematic](assets/images/method.png)

## Automation via GitHub Actions

- Workflow: `.github/workflows/docs.yml` (see repository) installs the package and runs `python docs/ci/check_docs.py` to ensure this README keeps the required sections.
- Future work: extend the job to execute lightweight notebook runs (single-shot tutorials) once execution time is under control.

## Assets

- `docs/assets/` – images + LaTeX snippets.
- `videos/` – rendered animations referenced in the README/tutorial.

If you add an experiment or figure, update the table above and drop assets into the folders listed here so the GitHub Action has everything it needs.
