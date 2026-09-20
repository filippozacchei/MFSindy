# weak_validity_ratio — the computation is right, the paper's formula isn't
The structure is correct. From (A.1), the feature-driven row-k contribution is Σᵢ Vₖᵢ ∇F(uᵢ)εᵢ and the retained one is Σᵢ V̇ₖᵢ εᵢ, so comparing the ℓ₂ norms of row k of V̇ against row k of V ⊙ |∇F| is exactly the diagnostic (A.2) asks for. fulltweights[k] really is the row of V̇ and fullweights0[k] the row of V, and pysindy's scaling (∏H^(1-deriv) vs ∏H) leaves the two consistently normalised, so this is the "assembled weak system" ratio, not a proxy. I confirmed the scaling empirically on Lorenz — κ goes 0.0023 → 0.0101 → 0.0443 → 0.177 → 0.758 → 2.44 as H doubles from 0.00625 to 0.2, i.e. ~4× per doubling, clean O(h²).

That O(h²) is the problem. (A.3) as printed is max_k ‖φₖ|∇F|‖₂ / ‖φ̇ₖ‖₂ — unsquared. With ‖V‖₂² = O(h) and ‖V̇‖₂² = O(h⁻¹), that ratio is O(h), not O(h²). The two statements in that same paragraph contradict each other. The code uses squared norms and matches the O(h²) claim, so the display in (A.3) and (A.2) is what needs the ². The docstring in weighted_weak_pde_library.py:80 asserts "the appendix writes [it] as a ratio of squared norms" — it doesn't, currently.

σ is dropped. The covariance model is V̇ D_σ V̇ᵀ, not σ²V̇V̇ᵀ. The faithful diagonal ratio is Σᵢ σᵢ²(Vₖᵢ|∇Fᵢ|)² / Σᵢ σᵢ²V̇ₖᵢ². This is harmless in Part I because the variance field is constant per trajectory (np.full(traj.shape[:-1], noise_abs**2)), so σ² cancels exactly — worth one sentence in the appendix. It is not harmless in Part II, where the variance is ∝ α²‖U‖² — and Part II computes no κ at all.

Minor: indices = inds_k[k][0] takes only the first grid axis while phi.ravel() is the whole tensor, so the function is silently ODE-only with no guard. And |∇F| is evaluated on the noisy trajectory; I measured this as negligible (0.7585 vs 0.7582 at 1% HF noise), so ignore it.

The numbers are the real finding. At the supports the tuner actually selects, current code gives κ_median = 0.76 at H = T/10 and 2.44 at H = T/5. Six of eight Lorenz rungs selected H ∈ {0.1, 0.2}. κ ≳ 1 means the term the covariance model discards is as large as the one it keeps — condition (A.2) is violated at the operating point the paper reports. That's not a bug, it's a result, and it needs to be stated rather than buried in a table column.

# pde_scale_separation_ratio is dimensional and shouldn't share a column with κ
h_t²·min(h_x)^(-2m) carries units of time²/length^(2m). Burgers (L=8, T=1) yields 0.00038–1.56 across its grid; NS (L=5, T=0.1) yields 2.5e-5–0.10. Those aren't comparable to each other, aren't comparable to the ODE κ, and would shift by 10⁸ if you measured x in different units. It's also the bare monomial from an O(·) statement with the constant set to 1, it ignores |∇F| entirely, and it moves in the opposite direction from the ODE κ (wider spatial support ⇒ smaller). Put it in the same table column labelled κ and a referee will read it as the same quantity. It needs a different name and a non-dimensionalisation.

# cfg export — yes, there are bugs
Lorenz's K_rule/K_selected are wrong. The JSON says "K_rule": "int(5 * (t1 - t0) / H_xt)", K_selected 50/25 — while weak_design in the same file says K_requested 10/5. The 5× rule is copied from LorenzIntraTrajectoryGLSConfig (lorenz.py:520, Part II); Part I uses max(2, round(extent/H)). Only Lorenz carries this block.

Every committed tuning JSON is stale. Only Lorenz's has any κ at all — hopf, pendulum, burgers and NS have weak_design without kappa keys, predating commit 0447ac6. And Lorenz's κ (1.228 / 4.081) doesn't match what the code produces today (0.759 / 2.440) — it predates the coverage-2 and dedup commits. Hopf and pendulum JSONs also record a 4-point H grid [0.02, 0.05, 0.1, 0.2] against notebooks that now define 3 points ending at 0.1.

fixed is a hand-maintained duplicate of the config and has drifted. Unexported everywhere: n_runs, T_true, deduplicate, whitener_mode, stlsq_alpha. p is exported for lorenz/hopf/pendulum but not burgers/NS; poly_degree/derivative_order for burgers but not NS. dataclasses.asdict(rung_cfg) would make this correct by construction. (Serialization itself is fine — I checked, floats and H_xt lists survive default=str.)

Lorenz drops the boundary-flag print every other notebook has. Its JSON flags stlsq_threshold: "low" on 5 of 8 rungs and H_xt: "high" on 3 — the grid is too narrow at both ends and the notebook never says so.

Dead imports in both lorenz and isothermal cell 0 (make_metric_scorer, optimize_hyperparams, load/save_ns_part1_tuned_hyperparams), and PLOT_ORDER is defined and never used.

# Validation bugs
- Pendulum tunes at 5.4× the Monte Carlo noise level. Cell 4's X_ref calls generate_pendulum_dataset(T=10.0, seed=cfg.seed_base+10) without g/L/c, so it runs at the generator default c=0.1 while cfg.c=0.5. Measured: notebook tune_std = 1.2168, MC state_std = 0.2262. Every pendulum hyperparameter is selected on data 5.4× noisier than it's evaluated on. Hopf has the same shape of bug (hardcoded T=100.0 vs cfg.T_true=10.0) but it's benign — ratio 1.006, the limit cycle dominates.

- NS tunes on a different flow. The tuning cell draws its reference at seed_base + TUNING_SEED_OFFSET, and make_initial_condition randomises the Taylor–Green wavenumbers from that seed — which the code's own comment notes changes the derivative magnitudes. The MC's state_std comes from seed_base. Different SNR at tune time than at eval time. Burgers uses a third reference again.

- The selection objective disagrees with the reported metric. The tuner maximises full-horizon rollout R²; the paper reports coefficient MAE/L0. For HF these disagree materially: at the tuned point (H=0.1) coefficient MAE is 1.99, while H=0.05 — also in the grid — gives 0.29 under plain STLSQ. The tuner chose the 7×-worse point because its rollout R² was less negative.

- The HF/VHF baselines are broken, and I'd fix this before anything else. Committed best scores: HF −0.356, VHF −0.195 — worse than predicting the mean, across all 18 grid points. It isn't chaos: I ran the true Lorenz coefficients through the same rollout on the same noisy validation set and got 0.99. The cause is the weak design. At H=0.1 a single HF trajectory gives K=10 rows for 9 unknowns with cond(Θ) ≈ 1.75e7; at H=0.2 it's K=5 rows for 9 unknowns — underdetermined on clean data (MAE 2.28). LF gets 10× the rows from its 10 trajectories. Ablations (bagging off, normalize_columns=True) leave HF at MAE 2–6 at every H. So an HF baseline at 1% noise losing to an LF baseline at 25% noise is a weak-equation-count artifact, not a noise-weighting result — and that is precisely the comparison the paper's headline rests on.

- Silent -1e12. Both evaluate_rollout_models and evaluate_weak_form_models swallow every exception into score = -1e12. If a wiring bug makes all grid points fail, tune_rungs still succeeds and idxmax returns the first row — an arbitrary selection, reported as tuned. Commit 9a106f1 was exactly such a wiring fix in the NS tuning call, so this has already bitten once silently.

- STLSQ thresholds on unnormalised weak columns. Measured column norms of the Lorenz weak Θ: [0.69, 0.71, 2.14, 6.08, 6.30, 18.3, 6.83, 18.7, 56.8] — 83× spread. One stlsq_threshold prunes the x/y columns far harder than z². This is likely why the threshold pins to the grid's lower edge on 5 of 8 rungs.

- weak_r2_score pools all state components into one SS_tot, so for NS whichever of (u, v, ρ) has the largest weak-RHS magnitude decides the score. Compounding it, the NS noise level is derived from channel 2 (ρ) alone and then applied to all three channels.

- NS uses n_bins=25 with n_runs=25 — about one sample per bubble.


