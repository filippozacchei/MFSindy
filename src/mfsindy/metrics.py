import numpy as np

def absolute_deviation(C_hat, C_ref, tau=1e-6):
    return np.abs(C_hat - C_ref) / np.maximum(np.abs(C_ref), tau)

def ensemble_disagreement(optimizer):
    coefs = np.stack(optimizer.coef_list, axis=0)
    median_coef = np.median(coefs, axis=0)
    abs_diff = np.abs(coefs - median_coef)
    return np.median(abs_diff, axis=0)
