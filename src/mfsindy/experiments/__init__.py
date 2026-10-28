"""Reusable experiment primitives."""

from .base import (
    REFERENCE_SEED_OFFSET,
    EnsembleConfigMixin,
    MonteCarloConfig,
    coefficient_errors,
    paired_comparison,
    run_monte_carlo_experiment,
)
from .intra_trajectory import (
    IntraTrajectoryGLSData,
    fit_intra_trajectory_gls_model_objects,
    run_intra_trajectory_gls_experiment,
)
from .multi_trajectory import (
    MultiTrajectoryGLSData,
    PART1_METHODS,
    FULL_COVARIANCE_RUNGS,
    RUNG_BLOCKS,
    WHITENING_RUNGS,
    assemble_weak_rungs,
    fit_multi_trajectory_weak_gls_models,
    run_multi_trajectory_gls_experiment,
)
from .selection import (
    BICEnsembleSTLSQ,
    load_selection,
    selection_path,
)
from .artifacts import (
    build_pde_intra_trajectory_artifacts,
    build_polynomial_intra_trajectory_artifacts,
    get_library_feature_names,
    pde_coverage_test_function_count,
)

__all__ = [
    "REFERENCE_SEED_OFFSET",
    "BICEnsembleSTLSQ",
    "load_selection",
    "selection_path",
    "coefficient_errors",
    "paired_comparison",
    "run_monte_carlo_experiment",
    "MonteCarloConfig",
    "EnsembleConfigMixin",
    "MultiTrajectoryGLSData",
    "assemble_weak_rungs",
    "RUNG_BLOCKS",
    "WHITENING_RUNGS",
    "FULL_COVARIANCE_RUNGS",
    "PART1_METHODS",
    "fit_multi_trajectory_weak_gls_models",
    "run_multi_trajectory_gls_experiment",
    "IntraTrajectoryGLSData",
    "fit_intra_trajectory_gls_model_objects",
    "run_intra_trajectory_gls_experiment",
    "build_pde_intra_trajectory_artifacts",
    "build_polynomial_intra_trajectory_artifacts",
    "get_library_feature_names",
    "pde_coverage_test_function_count",
]
