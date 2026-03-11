# API Reference

The package surface is intentionally small. The modules below are the ones most notebooks should import directly.

## Experiments (`mfsindy.experiments`)

- `run_multi_trajectory_gls_experiment`
- `run_intra_trajectory_gls_experiment`
- `MultiTrajectoryGLSData`
- `IntraTrajectoryGLSData`
- `MonteCarloConfig` and `EnsembleConfigMixin`

These utilities orchestrate Monte Carlo loops, compute coefficient errors, and store ensemble hyperparameters.

## Plots (`mfsindy.plots`)

- `bubble_hist(errors_dict, ...)`

Generates compact bubble histograms for Part 1 coefficient/support comparisons. Additional shared visuals (trajectory snapshots, residual bands) will live here in the future.

## Weighted weak library (`mfsindy.weighted_weak_pde_library`)

- `WeightedWeakPDELibrary`

Builds GLS-aware weak libraries compatible with PySINDy optimizers. It handles covariance-aware weighting, multi-fidelity scaling, and sample-wise variance propagation.

## Case generators (`mfsindy.cases.*`)

Each case module (Lorenz, Burgers, Hopf, pendulum, isothermal flow) exposes:

- Trajectory generators (`generate_<case>_dataset`)
- Config dataclasses (`<Case>MultiTrajectoryGLSConfig`, `<Case>IntraTrajectoryGLSConfig`)
- Convenience functions (`run_<case>_multi_trajectory_gls_experiment`, etc.)

The notebooks import from these modules rather than re-implementing problem-specific glue code.
