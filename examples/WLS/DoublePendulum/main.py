import sys
sys.path.append("../../../")

from pathlib import Path
from utils.part1 import evaluate_mf_sindy  # updated kwargs version
from generator import generate_dataset     # your double pendulum generator
import numpy as np
import pysindy as ps


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # System setup
    # ---------------------------------------------------------------------
    system_name = "double-pendulum"
    out_dir = "./Results"
    seed_val = 1

    # Grid definitions
    n_lf_vals = np.arange(1, 11, 1)
    n_hf_vals = np.arange(1, 11, 1)
    runs = 1
    dt = 0.001
    degree = 3
    threshold = 0.25

    # ---------------------------------------------------------------------
    # Custom feature library for SINDy
    # ---------------------------------------------------------------------
    custom_library = ps.PolynomialLibrary(degree=degree, include_bias=False)

    # ---------------------------------------------------------------------
    # Reference dataset for scaling
    # ---------------------------------------------------------------------
    X_test, _, _ = generate_dataset(noise=0.0, T=15, n_traj=5)

    std_test = float(np.std(X_test))
    print(f"Reference data standard deviation: {std_test:.4f}")

    # ---------------------------------------------------------------------
    # Run full multi-fidelity evaluation
    # ---------------------------------------------------------------------
    evaluate_mf_sindy(
        generator=generate_dataset,
        system_name=system_name,
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        runs=runs,
        noise_hf=0.01,
        noise_lf=0.25,
        dt=dt,
        threshold=threshold,
        out_dir=out_dir,
        seed=seed_val,
        lib=custom_library,
        distinct_ic=False,

        # ---- kwargs passthroughs ----
        gen_kwargs={
            "dt": dt,
        },
        gen_hf_kwargs={
            "T": 5.0,
            "n_per_trajectory": 1,
            "noise": 0.01 * std_test,
        },
        gen_lf_kwargs={
            "T": 5.0,
            "noise": 0.25 * std_test,
            "n_per_trajectory": 10,
        },
        gen_test_kwargs={
            "noise": 0.0,
            "T": 5.0,
            "n_per_trajectory": 1,
            "n_traj": 25
        },
        lib_kwargs={
            "p": 2,
        }
    )

    print(f"\n✅ Multi-fidelity evaluation for {system_name} completed. Results saved in {out_dir}\n")
