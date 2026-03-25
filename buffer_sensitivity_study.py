"""
Buffer sensitivity study: sweeps the buffer margin parameter across a range of values
for a given parameterization, measuring how it affects convergence.

Usage:
    python buffer_sensitivity_study.py                  # runs all parameterizations
    python buffer_sensitivity_study.py harmonic_mean    # runs one parameterization
"""
import csv
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from main_experiment import (
    PARAMETERIZATIONS,
    IterativeSolution, run_adaptive_discovery, print_distance_to_truth,
    n_mcmc_samples, UNCERTAINTY_MULTIPLIER,
)
from simulator import true_process
from mcmc_sampler import get_true_logprob, run_mcmc_blackjax

# ============================================================
# STUDY CONFIGURATION
# ============================================================
BUFFER_SCALES = np.linspace(0, 2, 10)
MODEL_TYPE = "gp"  # emulator to use for the sweep
# ============================================================


def run_study_single(name, cfg, key):
    """Run the buffer sensitivity sweep for one parameterization."""
    sim_func = cfg["sim_func"]
    true_theta = cfg["true_theta"]
    theta_bounds = cfg["theta_bounds"]
    x_bounds = cfg["x_bounds"]
    n_params = len(true_theta)
    all_bounds = theta_bounds + [x_bounds]
    initial_mcmc_position = jnp.array([(lo + hi) / 2.0 for lo, hi in theta_bounds])

    n_ppe_r1 = 50 * n_params ** 2
    n_ppe_r2 = n_ppe_r1 // 10
    n_refinement_rounds = n_params
    refinement_range = list(range(2, 2 + n_refinement_rounds))

    print(f"\n{'#'*70}")
    print(f"# BUFFER SENSITIVITY: {name} ({n_params} params)")
    print(f"# True theta: {true_theta}")
    print(f"# n_ppe_r1: {n_ppe_r1}, n_ppe_r2: {n_ppe_r2}")
    print(f"# Buffer scales: {[f'{s:.0%}' for s in BUFFER_SCALES]}")
    print(f"{'#'*70}")

    out_dir = f"posteriors/buffer_sst/{name}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Generate observations ---
    x_obs = jnp.geomspace(x_bounds[0] if x_bounds[0] > 0 else 0.1, x_bounds[1], 50)
    key, subkey = jax.random.split(key)
    clean_y = true_process(x_obs, true_theta, sim_func)
    obs_error = 0.05 * jnp.abs(clean_y)
    y_obs = clean_y + obs_error * jax.random.normal(subkey, x_obs.shape)

    # --- True MCMC baseline ---
    print("\n--- True MCMC baseline ---")
    true_lp = get_true_logprob(x_obs, y_obs, obs_error, theta_bounds, sim_func)
    key, subkey = jax.random.split(key)
    true_smp = np.array(run_mcmc_blackjax(true_lp, initial_mcmc_position, 3000, subkey)[1500:])
    true_theta_dist, true_sim_rmse = print_distance_to_truth(
        true_smp, true_theta, "True Simulator", x_obs, y_obs, sim_func
    )
    np.savez(f"{out_dir}/True_Simulator_samples.npz", samples=true_smp)

    # --- Shared discovery round ---
    print("\n--- Discovery (shared across all buffer scales) ---")
    key, subkey = jax.random.split(key)
    X_R1, y_R1, samples_R1, disc_time = run_adaptive_discovery(
        f"{MODEL_TYPE.upper()}-Shared", MODEL_TYPE, subkey, all_bounds,
        x_obs, y_obs, obs_error, true_theta, theta_bounds, sim_func,
        n_ppe_override=n_ppe_r1,
    )
    _, r1_sim_rmse = print_distance_to_truth(
        samples_R1, true_theta, "Discovery R1", x_obs, y_obs, sim_func
    )

    # --- One IterativeSolution per buffer scale ---
    solutions = {}
    for scale in BUFFER_SCALES:
        sol = IterativeSolution(
            name=f"{MODEL_TYPE.upper()} Buffered {scale:.0%}",
            strategy_type="buffered",
            model_type=MODEL_TYPE,
            X_R1=X_R1, y_R1=y_R1, samples_R1=samples_R1.copy(),
            x_obs=x_obs, y_obs=y_obs, obs_error=obs_error,
            theta_bounds=theta_bounds, x_bounds=x_bounds,
            sim_func=sim_func,
            true_theta=true_theta,
            buffer_margin=scale,
        )
        sol.timings["Discovery"] = disc_time
        solutions[scale] = sol

    # --- Refinement rounds ---
    for round_idx in refinement_range:
        print(f"\n{'='*50}")
        print(f"--- ROUND {round_idx} ---")
        print(f"{'='*50}")
        for scale, sol in solutions.items():
            print(f"\n--- {sol.name} (Round {round_idx}) ---")
            sol.update_priors()
            key, subkey = jax.random.split(key)
            sol.add_round(n_ppe_r2, subkey, f"Round {round_idx}")
            key, subkey = jax.random.split(key)
            sol.train(subkey)
            sol.n_train_history[round_idx] = len(sol.y_train)
            key, subkey = jax.random.split(key)
            sol.run_mcmc(subkey, n_steps=3000, uncertainty_multiplier=2.0)
            sol.samples_history[round_idx] = sol.samples.copy()
            np.savez(f"{out_dir}/Scale_{scale:.2f}_Round_{round_idx}_samples.npz",
                     samples=sol.samples)

    # --- Collect results ---
    all_rounds = [1] + refinement_range
    final_results = {}
    round_results = {}  # scale -> list of per-round dicts

    for scale, sol in solutions.items():
        means = np.mean(sol.samples, axis=0)
        stds = np.std(sol.samples, axis=0)
        dist = float(jnp.sqrt(jnp.sum((means - true_theta)**2)))
        y_pred = true_process(x_obs, means, sim_func)
        sim_rmse = float(jnp.sqrt(jnp.mean((y_pred - y_obs)**2)))
        final_results[scale] = {"means": means, "stds": stds, "dist": dist, "sim_rmse": sim_rmse}

        per_round = []
        for ridx in all_rounds:
            smp = sol.samples_history.get(ridx)
            if smp is None:
                continue
            m = jnp.mean(smp, axis=0)
            y_p = true_process(x_obs, m, sim_func)
            per_round.append({
                "round": ridx,
                "n_train": sol.n_train_history.get(ridx, int(0.8 * n_ppe_r1)),
                "sim_rmse": float(jnp.sqrt(jnp.mean((y_p - y_obs)**2))),
                "dist": float(jnp.sqrt(jnp.sum((m - true_theta)**2))),
                "mean": np.array(m),
            })
        round_results[scale] = per_round

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"--- Buffer Sensitivity Summary: {name} ---")
    header = f"{'Scale':>10} | " + " | ".join([f"Th_{i}" for i in range(n_params)]) + " | DistToTh | Sim RMSE"
    print(header)
    print("-" * len(header))
    row = f"{'True Sim':>10} | " + " | ".join(
        [f"{m:.2f}±{s:.2f}" for m, s in zip(np.mean(true_smp, axis=0), np.std(true_smp, axis=0))]
    ) + f" | {true_theta_dist:8.4f} | {true_sim_rmse:8.4f}"
    print(row)
    print("-" * len(header))
    for scale in sorted(final_results.keys()):
        r = final_results[scale]
        row = f"{scale:8.0%} | " + " | ".join(
            [f"{m:.2f}±{s:.2f}" for m, s in zip(r["means"], r["stds"])]
        ) + f" | {r['dist']:8.4f} | {r['sim_rmse']:8.4f}"
        print(row)

    # --- CSV output ---
    csv_path = f"results/{name}_buffer_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        theta_cols = [f"mean_theta_{i}" for i in range(n_params)]
        writer.writerow(["buffer_scale", "round", "n_train", "dist_to_truth", "sim_rmse"] + theta_cols)
        for scale in sorted(round_results.keys()):
            for m in round_results[scale]:
                row = [f"{scale:.4f}", m["round"], m["n_train"],
                       f"{m['dist']:.6f}", f"{m['sim_rmse']:.6f}"]
                row += [f"{v:.6f}" for v in m["mean"]]
                writer.writerow(row)
    print(f"  Saved {csv_path}")

    # --- Plot: convergence trajectories ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.plasma
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(BUFFER_SCALES), vmax=max(BUFFER_SCALES)))

    ax = axes[0]
    for scale in sorted(round_results.keys()):
        rmses = [m["sim_rmse"] for m in round_results[scale]]
        rounds = [m["round"] for m in round_results[scale]]
        ax.plot(rounds, rmses, marker='o', color=cmap(sm.norm(scale)))
    ax.axhline(true_sim_rmse, color='black', linestyle='--', alpha=0.5, label="True Sim")
    ax.set_xlabel("Round")
    ax.set_ylabel("Simulator RMSE (at Posterior Mean)")
    ax.set_title(f"{name}: Performance Trajectories by Buffer Scale")
    ax.set_xticks(all_rounds)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sm, ax=ax, label="Buffer Scale")

    ax = axes[1]
    scales_list = sorted(final_results.keys())
    sim_rmses_list = [final_results[s]["sim_rmse"] for s in scales_list]
    colors = [cmap(sm.norm(s)) for s in scales_list]
    ax.bar([f"{s:.0%}" for s in scales_list], sim_rmses_list, color=colors, alpha=0.8, edgecolor='k')
    ax.axhline(true_sim_rmse, color='black', linestyle='-', linewidth=2, label=f"True Sim: {true_sim_rmse:.3f}")
    ax.axhline(min(sim_rmses_list), color='tab:red', linestyle='--', label=f"Best: {min(sim_rmses_list):.3f}")
    ax.set_xlabel("Buffer Scale")
    ax.set_ylabel("Final Simulator RMSE")
    ax.set_title(f"{name}: Final Error (Round {refinement_range[-1]})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"plots/{name}_buffer_sensitivity.pdf")
    plt.close()

    # --- Plot: per-parameter posteriors ---
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
    if n_params == 1:
        axes = [axes]
    for i in range(n_params):
        ax = axes[i]
        for scale in sorted(solutions.keys()):
            sol = solutions[scale]
            ax.hist(sol.samples[:, i], bins=50, histtype='step', density=True,
                    color=cmap(sm.norm(scale)), linewidth=1.5)
        ax.axvline(float(true_theta[i]), color='black', linestyle='--', linewidth=2)
        ax.set_title(f"$\\theta_{i}$")
        ax.set_xlabel(f"$\\theta_{i}$")
    plt.suptitle(f"{name}: Posterior by Buffer Scale (Round {refinement_range[-1]})")
    plt.tight_layout()
    plt.savefig(f"plots/{name}_buffer_posteriors.pdf")
    plt.close()

    print(f"\n  Saved plots/{name}_buffer_sensitivity.pdf, plots/{name}_buffer_posteriors.pdf")


def main():
    key = jax.random.PRNGKey(42)

    if len(sys.argv) > 1:
        names_to_run = sys.argv[1:]
        for n in names_to_run:
            if n not in PARAMETERIZATIONS:
                print(f"Unknown parameterization: {n}. Options: {list(PARAMETERIZATIONS.keys())}")
                sys.exit(1)
    else:
        names_to_run = list(PARAMETERIZATIONS.keys())

    for pname in names_to_run:
        cfg = PARAMETERIZATIONS[pname]
        key, subkey = jax.random.split(key)
        run_study_single(pname, cfg, subkey)

    print("\nDone.")


if __name__ == "__main__":
    main()
