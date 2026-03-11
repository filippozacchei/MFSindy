from pathlib import Path
import numpy as np

import sys
sys.path.append("../../../")
from utils.plot import plot_heatmap

system_name = "lorenz"
in_dir = "./Results"
out_dir = "./Figures"

# Define grid and parameters
n_lf_vals = np.arange(10, 101, 10)
n_hf_vals = np.arange(1, 11, 1)
data = np.load(Path(in_dir) / system_name / f"{system_name}_results.npz")

# Example plots
plot_heatmap(np.clip(data["mf_score"], 0, None), n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_mf.png")
plot_heatmap(np.clip(data["lf_score"], 0, None), n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_lf.png")
plot_heatmap(np.clip(data["hf_score"], 0, None), n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_hf.png")
plot_heatmap(np.clip(data["dlf_score"], None, 1), n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_mf_minus_lf.png")
plot_heatmap(np.clip(data["dhf_score"], None, 1), n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_mf_minus_hf.png")

plot_heatmap(data["mf_mad"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_mf.png", label="MAD")
plot_heatmap(data["lf_mad"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_lf.png", label="ΔMAD")
plot_heatmap(data["hf_mad"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_hf.png", label="ΔMAD")
plot_heatmap(data["dlf_mad"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_mf_minus_lf.png", label="ΔMAD")
plot_heatmap(data["dhf_mad"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_mf_minus_hf.png", label="ΔMAD")

plot_heatmap(data["mf_dis"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_mf.png", label="Disagreement")
plot_heatmap(data["lf_dis"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_lf.png", label="ΔDisagreement")
plot_heatmap(data["hf_dis"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_hf.png", label="ΔDisagreement")
plot_heatmap(data["dlf_dis"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_mf_minus_lf.png", label="ΔDisagreement")
plot_heatmap(data["dhf_dis"], n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_mf_minus_hf.png", label="ΔDisagreement")

print(f"\nLorenz evaluation complete. Results and figures saved to {out_dir}")
