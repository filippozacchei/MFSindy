import pysindy as ps
import numpy as np


def run_ensemble_sindy(
    X_list,
    t_list,
    threshold: float = 0.1,
    library=None,
    weights=None,
    n_models: int = 20,
):
    """
    Train an Ensemble SINDy (Weak-SINDy) model.
    """
    optimizer = ps.EnsembleOptimizer(
        ps.STLSQ(threshold=threshold, alpha=1e-12),
        bagging=True,
        n_models=n_models,
    )
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    model.fit(X_list, t=t_list, sample_weight=weights)
    return model, optimizer


def copy_sindy(source_model, X_list, t_list):
    """
    Create a plain SINDy model that copies coefficients
    from a trained weak/ensemble SINDy model.
    """
    optimizer_copy = ps.STLSQ()
    model_copy = ps.SINDy(
        feature_library=source_model.feature_library,
        optimizer=optimizer_copy,
    )

    model_copy.fit(X_list, t=t_list)
    optimizer_copy.coef_ = np.copy(source_model.optimizer.coef_)

    return model_copy, optimizer_copy
