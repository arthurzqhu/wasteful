import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the shared infrastructure directly from main_experiment.py
from main_experiment import (
    IterativeSolution, run_adaptive_discovery, print_distance_to_truth,
    power_law_func, TRUE_THETA, bounds_R1, initial_mcmc_position,
    n_samples_R2,
)
from simulator import true_process
from mcmc_sampler import get_true_logprob, run_mcmc_blackjax

SIM_FUNC = power_law_func  # convenience alias


# ============================================================
# STUDY CONFIGURATION
# ============================================================

# Fractional padding on each side of the 3-sigma interval.
# Extend this list to test more values.
BUFFER_SCALES = np.linspace(0, 2, 10)

# Rounds to run after discovery. Extend for more complex models.
REFINEMENT_ROUNDS = [2, 3, 4]

# ============================================================


def run_study():
    key = jax.random.PRNGKey(42)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('posteriors/buffer_sst', exist_ok=True)

    # 1. OBSERVATION DATA (identical setup to main_experiment.py)
    x_obs = jnp.geomspace(bounds_R1[-1][0], bounds_R1[-1][1], 50)
    key, subkey = jax.random.split(key)
    clean_y = true_process(x_obs, TRUE_THETA, SIM_FUNC)
    obs_error = 0.05 * clean_y
    y_obs = clean_y + obs_error * jax.random.normal(subkey, x_obs.shape)

    mcmc_prior_bounds = bounds_R1[:-1]
    
    # 2. RUN TRUE MCMC (Physical Simulator benchmark)
    # This establishes the theoretical limit of parameter recovery given the noise and model structure.
    print("=" * 60)
    print("--- Running True MCMC (Physical Simulator Baseline) ---")
    print("=" * 60)
    true_lp = get_true_logprob(x_obs, y_obs, obs_error, mcmc_prior_bounds, SIM_FUNC)
    key, subkey = jax.random.split(key)
    # Use 3000 steps to match the refinement depth
    true_smp = np.array(run_mcmc_blackjax(true_lp, initial_mcmc_position, 3000, subkey)[1500:])
    
    true_theta_dist, true_sim_rmse = print_distance_to_truth(
        true_smp, TRUE_THETA, "True Simulator", x_obs, y_obs, SIM_FUNC
    )
    print(f"  True Simulator baseline RMSE: {true_sim_rmse:.4f}")
    np.savez('posteriors/buffer_sst/True_Simulator_samples.npz', samples=true_smp)

    # 3. ROUND 1 DISCOVERY — shared across all buffer scales
    print("=" * 60)
    print("--- Running Round 1 Discovery (Shared) ---")
    print("=" * 60)
    
    disc_data_fname = 'posteriors/buffer_sst/Discovery_data.npz'
    if os.path.exists(disc_data_fname):
        print("  [Persistence] Loading Discovery data...")
        data = np.load(disc_data_fname)
        X_R1, y_R1, samples_R1, disc_time = data['X'], data['y'], data['samples'], float(data['time'])
    else:
        key, subkey = jax.random.split(key)
        X_R1, y_R1, samples_R1, disc_time = run_adaptive_discovery(
            "GP-Shared", "gp", subkey, bounds_R1,
            x_obs, y_obs, obs_error, TRUE_THETA, mcmc_prior_bounds, SIM_FUNC
        )
        np.savez(disc_data_fname, X=X_R1, y=y_R1, samples=samples_R1, time=disc_time)
        print(f"  Discovery completed in {disc_time:.1f}s")

    # Initial performance
    _, r1_sim_rmse = print_distance_to_truth(samples_R1, TRUE_THETA, "Discovery R1", x_obs, y_obs, SIM_FUNC)
    np.savez('posteriors/buffer_sst/Discovery_samples.npz', samples=samples_R1)

    # 3. INITIALIZE ONE IterativeSolution PER BUFFER SCALE
    #    Only 'buffered' strategy is swept; buffer_margin controls the padding.
    solutions = {}
    for scale in BUFFER_SCALES:
        sol = IterativeSolution(
            name=f"GP Buffered {scale:.0%}",
            strategy_type="buffered",
            model_type="gp",
            X_R1=X_R1, y_R1=y_R1, samples_R1=samples_R1.copy(),
            x_obs=x_obs, y_obs=y_obs, obs_error=obs_error,
            initial_bounds=bounds_R1,
            sim_func=SIM_FUNC,
            true_theta=TRUE_THETA,
            buffer_margin=scale,           # <-- the only thing that differs
        )
        sol.timings["Discovery"] = disc_time
        solutions[scale] = sol

    # 4. ITERATIVE REFINEMENT ROUNDS
    #    Mirrors the main_experiment.py loop exactly — add rounds by extending REFINEMENT_ROUNDS.
    for round_idx in REFINEMENT_ROUNDS:
        print(f"\n{'='*60}")
        print(f"--- RUNNING ROUND {round_idx} ---")
        print("=" * 60)
        for scale, sol in solutions.items():
            print(f"\n--- {sol.name} (Round {round_idx}) ---")
            
            fname = f'posteriors/buffer_sst/Scale_{scale:.2f}_Round_{round_idx}_samples.npz'
            if os.path.exists(fname):
                print(f"  [Persistence] Loading samples for {sol.name} (Round {round_idx})...")
                sol.samples = np.load(fname)['samples']
                # Re-populate history for plotting
                means = jnp.mean(sol.samples, axis=0)
                y_pred = true_process(x_obs, means, SIM_FUNC)
                cur_sim_rmse = float(jnp.sqrt(jnp.mean((y_pred - y_obs)**2)))
                sol.samples_history[round_idx] = {"samples": sol.samples.copy(), "sim_rmse": cur_sim_rmse}
                continue

            sol.update_priors()
            key, subkey = jax.random.split(key)
            sol.add_round(n_samples_R2, subkey, f"Round {round_idx}")
            key, subkey = jax.random.split(key)
            sol.train(subkey)
            print(f"  -> Sampling posterior (Scale: 2.0, Steps: 3000)...")
            key, subkey = jax.random.split(key)
            sol.run_mcmc(subkey, n_steps=3000, uncertainty_multiplier=2.0)
            
            # Calculate Simulator RMSE for history
            means = jnp.mean(sol.samples, axis=0)
            y_pred = true_process(x_obs, means, SIM_FUNC)
            cur_sim_rmse = float(jnp.sqrt(jnp.mean((y_pred - y_obs)**2)))
            
            sol.samples_history[round_idx] = {
                "samples": sol.samples.copy(),
                "sim_rmse": cur_sim_rmse
            }
            # Save per-round per-scale samples
            np.savez(f'posteriors/buffer_sst/Scale_{scale:.2f}_Round_{round_idx}_samples.npz', samples=sol.samples)

    # 5. SUMMARY TABLE
    N_PARAMS = len(mcmc_prior_bounds)
    print(f"\n{'='*60}")
    print("--- Final Buffer Sensitivity Summary ---")
    header = f"{'Scale':>10} | " + " | ".join([f"Th_{i}" for i in range(N_PARAMS)]) + " | DistToTh | Sim RMSE"
    print(header)
    print("-" * len(header))
    
    # True Simulator Reference Row
    row = f"{'True Sim':>10} | " + " | ".join([f"{m:.2f}±{s:.2f}" for m, s in zip(np.mean(true_smp, axis=0), np.std(true_smp, axis=0))]) + f" | {true_theta_dist:8.4f} | {true_sim_rmse:8.4f}"
    print(row)
    print("-" * len(header))

    final_results = {}
    for scale, sol in solutions.items():
        means = np.mean(sol.samples, axis=0)
        stds = np.std(sol.samples, axis=0)
        dist = float(jnp.sqrt(jnp.sum((means - TRUE_THETA)**2)))
        y_pred = true_process(x_obs, means, SIM_FUNC)
        sim_rmse = float(jnp.sqrt(jnp.mean((y_pred - y_obs)**2)))
        
        final_results[scale] = {"means": means, "stds": stds, "dist": dist, "sim_rmse": sim_rmse}
        row = f"{scale:8.0%} | " + " | ".join([f"{m:.2f}±{s:.2f}" for m, s in zip(means, stds)]) + f" | {dist:8.4f} | {sim_rmse:8.4f}"
        print(row)

    # 6. PLOT: Convergence trajectories across rounds
    all_rounds = [1] + REFINEMENT_ROUNDS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Continuous colormap for scales
    cmap = plt.cm.plasma
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(BUFFER_SCALES), vmax=max(BUFFER_SCALES)))

    ax = axes[0]
    for i, (scale, sol) in enumerate(solutions.items()):
        rmse_history = [r1_sim_rmse]
        for r in REFINEMENT_ROUNDS:
            h = sol.samples_history.get(r)
            if h is not None:
                rmse_history.append(h["sim_rmse"])
        ax.plot(all_rounds[:len(rmse_history)], rmse_history, marker='o', 
                color=cmap(sm.norm(scale)), label=f"Scale {scale:.0%}")
    ax.axhline(true_sim_rmse, color='black', linestyle='--', alpha=0.5, label="True Sim")
    ax.set_xlabel("Round")
    ax.set_ylabel("Simulator RMSE (at Posterior Mean)")
    ax.set_title("Performance Trajectories by Buffer Scale")
    ax.set_xticks(all_rounds)
    # ax.legend(fontsize=9) # Removing legend to emphasize colormap trend
    ax.grid(True, alpha=0.3)
    plt.colorbar(sm, ax=ax, label="Buffer Scale")

    ax = axes[1]
    scales_list = sorted(final_results.keys())
    sim_rmses_list = [final_results[s]["sim_rmse"] for s in scales_list]
    colors = [cmap(sm.norm(s)) for s in scales_list]
    ax.bar([f"{s:.0%}" for s in scales_list], sim_rmses_list, color=colors, alpha=0.8, edgecolor='k')
    ax.axhline(true_sim_rmse, color='black', linestyle='-', linewidth=2, label=f"True Sim: {true_sim_rmse:.3f}")
    ax.axhline(min(sim_rmses_list), color='tab:red', linestyle='--', label=f"Best PPE: {min(sim_rmses_list):.3f}")
    ax.set_xlabel("Buffer Scale")
    ax.set_ylabel("Final Simulator RMSE")
    ax.set_title(f"Final Prediction Error (After Round {REFINEMENT_ROUNDS[-1]})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('plots/buffer_sensitivity_results.pdf')
    plt.close()

    # 7. PLOT: Per-parameter posterior distributions (final round)
    fig, axes = plt.subplots(1, N_PARAMS, figsize=(6 * N_PARAMS, 5))
    if N_PARAMS == 1:
        axes = [axes]
    for i in range(N_PARAMS):
        ax = axes[i]
        for scale in sorted(solutions.keys()):
            sol = solutions[scale]
            ax.hist(sol.samples[:, i], bins=50, histtype='step', density=True,
                    color=cmap(sm.norm(scale)), label=f"Scale {scale:.0%}", linewidth=1.5)
        ax.axvline(float(TRUE_THETA[i]), color='black', linestyle='--', label="Truth", linewidth=2)
        ax.set_title(f"Parameter $\\theta_{i}$ Posterior")
        ax.set_xlabel(f"$\\theta_{i}$")
        # ax.legend(fontsize=9) # Colorbar handles it
    plt.suptitle(f"Posterior Comparison by Buffer Scale (Round {REFINEMENT_ROUNDS[-1]})")
    plt.tight_layout()
    plt.savefig('plots/buffer_posterior_comparison.pdf')
    plt.close()

    print("\nSaved: plots/buffer_sensitivity_results.pdf, plots/buffer_posterior_comparison.pdf")


if __name__ == "__main__":
    run_study()
