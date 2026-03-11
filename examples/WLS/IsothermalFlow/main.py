import sys
sys.path.append("../../../")

from pathlib import Path
from utils.part1 import evaluate_mf_sindy  # updated kwargs version
from generator import generate_compressible_flow, animate_field
import numpy as np
import pysindy as ps


if __name__ == "__main__":
    # System setup
    system_name = "isothermal-flow"
    out_dir = "./Results"
    seed_val = 1

    # Grid definitions
    n_lf_vals = np.arange(10, 101, 10)
    n_hf_vals = np.arange(1, 11, 1)
    runs = 1
    dt = 0.001
    threshold = 0.5
    L = 5

    # ---------------------------------------------------------------------
    # Custom feature library for SINDy
    # ---------------------------------------------------------------------
    library_functions = [
        lambda x: x,
        lambda x: 1 / (1e-6 + np.abs(x))
    ]
    library_function_names = [
        lambda x: x,
        lambda x: x + "^-1"
    ]

    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names
    )

    X_test, _, _ = generate_compressible_flow(n_traj=1,T=1.0,seed=999,noise_level=0.0)
    std_test = float(np.std(X_test))  # scale for noise
    
    # ---------------------------------------------------------------------
    # Run full multi-fidelity evaluation with kwargs passthrough
    # ---------------------------------------------------------------------
    evaluate_mf_sindy(
        generator=generate_compressible_flow,
        system_name=system_name,
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        runs=runs,
        dt=dt,
        threshold=threshold,
        out_dir=out_dir,
        seed=seed_val,
        lib=custom_library,

        # ---- kwargs passthroughs ----
        gen_kwargs={
            "N": 64,
            "L": L,
            "mu": 1.0,
            "RT": 1.0,
            "initial_condition": "taylor-green",
            "noise_0": 0.01,
        },
        gen_hf_kwargs={"noise_level": 0.01*std_test, "T":0.1, "Nt":100},   # high-fidelity noise
        gen_lf_kwargs={"noise_level": 0.25*std_test, "T":0.1, "Nt":100},   # low-fidelity noise
        gen_test_kwargs={"noise_level": 0.0, "T": 1.0, "n_traj":1, "seed":999, "Nt":1000},
        lib_kwargs={
            "derivative_order": 2,
            "p": 2,
        },
    )

    print(f"\nMulti-fidelity evaluation for {system_name} completed. Results saved in {out_dir}\n")
