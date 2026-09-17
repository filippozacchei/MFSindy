"""Reusable experiment primitives."""

from .base import (
    EnsembleConfigMixin,
    MonteCarloConfig,
    coefficient_errors,
    run_monte_carlo_experiment,
)
from .hyperparameters import (
    HyperparameterSearchResult,
    MetricTerm,
    make_combined_metric_scorer,
    make_metric_scorer,
    optimize_hyperparams,
)
from .intra_trajectory import (
    IntraTrajectoryGLSData,
    fit_intra_trajectory_gls_model_objects,
    fit_intra_trajectory_gls_models,
    run_intra_trajectory_gls_experiment,
)
from .multi_trajectory import (
    MultiTrajectoryGLSData,
    fit_multi_trajectory_gls_models,
    fit_multi_trajectory_weak_gls_models,
    run_multi_trajectory_gls_experiment,
)
from .tuning import (
    RungTuning,
    TUNING_SEED_OFFSET,
    run_with_tuned_configs,
    save_tuning,
    tune_rungs,
)
from .rollout import (
    PolynomialRolloutModel,
    build_polynomial_intra_trajectory_artifacts,
    build_polynomial_rollout_models,
    evaluate_rollout_models,
    evaluate_windowed_rollout_models,
    format_model_equations,
    rollout_r2_score,
    split_single_trajectory,
    split_trajectory_list,
)
from .weak_validation import (
    WeakValidationBlock,
    evaluate_weak_form_models,
    format_coefficient_equations,
    get_library_feature_names,
    split_spatiotemporal_trajectory,
    weak_r2_score,
)

__all__ = [
    "RungTuning",
    "TUNING_SEED_OFFSET",
    "run_with_tuned_configs",
    "save_tuning",
    "tune_rungs",
    "coefficient_errors",
    "run_monte_carlo_experiment",
    "MonteCarloConfig",
    "EnsembleConfigMixin",
    "HyperparameterSearchResult",
    "MetricTerm",
    "make_combined_metric_scorer",
    "make_metric_scorer",
    "optimize_hyperparams",
    "MultiTrajectoryGLSData",
    "fit_multi_trajectory_gls_models",
    "fit_multi_trajectory_weak_gls_models",
    "run_multi_trajectory_gls_experiment",
    "IntraTrajectoryGLSData",
    "fit_intra_trajectory_gls_model_objects",
    "fit_intra_trajectory_gls_models",
    "run_intra_trajectory_gls_experiment",
    "PolynomialRolloutModel",
    "build_polynomial_intra_trajectory_artifacts",
    "build_polynomial_rollout_models",
    "evaluate_rollout_models",
    "evaluate_windowed_rollout_models",
    "format_model_equations",
    "rollout_r2_score",
    "split_single_trajectory",
    "split_trajectory_list",
    "WeakValidationBlock",
    "evaluate_weak_form_models",
    "format_coefficient_equations",
    "get_library_feature_names",
    "split_spatiotemporal_trajectory",
    "weak_r2_score",
]
