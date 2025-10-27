import sys
sys.path.append("../../../")
from pathlib import Path
from utils.part1 import evaluate_mf_sindy
from generator import generate_compressible_flow, animate_field
import numpy as np
import pysindy as ps

# We need this signature
# X_hf, grid_hf, t_hf = generator(n_hf, 
#                                 noise_level=noise_level_hf * std_per_dim, 
#                                 T=T, 
#                                 seed=run*seed)
# def generate_compressible_flow(
#     n_traj=1,
#     N=64,
#     Nt=200,
#     L=0.1,
#     T=2,
#     mu=1.0,
#     RT=1.0,
#     noise_level=0.0,
#     seed=42,
#     initial_condition="taylor-green",
#     noise_0 = 0.01
# )

    
if __name__ == "__main__":
    system_name = "isothermal-flow"
    out_dir = "./Results_final"
    seedd = 1
    # Define grid and parameters
    n_lf_vals = np.arange(10, 101, 10)
    n_hf_vals = np.arange(1, 11, 1)
    runs = 25
    dt=0.001
    threshold = 0.5
    degree = 2
    L=5
    K=200
    T=0.1

    # Simple library for now (polynomial)
    library_functions = [
        lambda x: x,
        lambda x: 1 / (1e-6 + np.abs(x))
    ]
    library_function_names = [
        lambda x: x,
        lambda x: x + "^-1"
    ]

    # Create a custom feature library
    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names
    )
                
    x0, grid, t0 = generate_compressible_flow(T=T, noise_level=0.01)
    
    library = ps.feature_library.WeakPDELibrary(
        custom_library,
        derivative_order=2,
        spatiotemporal_grid=grid,
        p=2, 
        # K=K,
        # H_xt = [L/20, L/20, T/20]
    )
    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-12,)
    
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    
    model.fit(x=x0,t=t0)
    
    model.print()
    # u_dot = ps.FiniteDifference(d=1, axis=2)._differentiate(x0[0], t0[0])
    # animate_field(x0[0], t0[0], L=5, save_path="./flow1.gif")
    
    # x0, grid, t0 = generate_compressible_flow(T=T, noise_level=0.25)
    # library = ps.feature_library.WeakPDELibrary(
    #     custom_library,
    #     derivative_order=2,
    #     spatiotemporal_grid=grid,
    #     p=2, 
    #     K=K,
    #     H_xt = [L/10, L/10, T/10]
    # )
    # print(t0)
    # print(x0)
    # optimizer = ps.STLSQ(threshold=threshold, alpha=1e-12,)
    
    # model = ps.SINDy(feature_library=library, optimizer=optimizer)
    
    # model.fit(x=x0,t=t0)
    
    # model.print()
    # u_dot = ps.FiniteDifference(d=1, axis=2)._differentiate(x0[0], t0[0])
    # animate_field(x0[0], t0[0], L=5, save_path="./flow2.gif")
    
    # Run the unified evaluation routine
    evaluate_mf_sindy(
        generator=generate_compressible_flow,
        system_name=system_name,
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        noise_level_hf=0.01,
        noise_level_lf=0.25,
        runs=runs,
        dt=dt,
        threshold=threshold,
        degree=degree,
        out_dir=out_dir,
        seed=seedd,
        T=T,
        T_test=1.0,
        d_order=2,
        # K=K,
        # H_xt=[L/20, L/20, T/20],
        lib=custom_library,
    )
