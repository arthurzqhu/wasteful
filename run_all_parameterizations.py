"""
Runner script: evaluates all three parameterizations (power_law, harmonic_mean, hill_growth)
with both GP and NN emulators across all three strategies (global, buffered, local).

Number of refinement rounds = number of parameters for each parameterization.
"""
import csv
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import corner

jax.config.update("jax_enable_x64", True)

from simulator import simulator, true_process
from emulators import train_gp_jax, predict_gp_jax, tune_and_train_nn_jax, predict_nn_jax
from mcmc_sampler import get_true_logprob, get_gp_logprob, get_nn_logprob, run_mcmc_blackjax

# Import pieces from main_experiment
from main_experiment import (
    power_law, harmonic_mean, harmonic_mean_3t, hill_growth,
    save_npz, is_posterior_wide_enough, generate_ppes,
    print_distance_to_truth, run_adaptive_discovery, IterativeSolution,
    n_samples_R1, n_samples_R2, num_steps, UNCERTAINTY_MULTIPLIER,
)

# ---------------------------------------------------------------------------
# Parameterization configs
# ---------------------------------------------------------------------------
PARAMETERIZATIONS = {
    "power_law": {
        "sim_func": power_law,
        "true_theta": jnp.array([1.6, 2.4]),
        "bounds": [(0.1, 5.0), (0.1, 5.0), (0.1, 10.0)],  # last is x
    },
    "harmonic_mean": {
        "sim_func": harmonic_mean,
        "true_theta": jnp.array([3.0, 2.0, 16.0]),
        "bounds": [(0.1, 5.0), (0.1, 5.0), (1.0, 20.0), (0.1, 150.0)],
    },
    "harmonic_mean_3t": {
        "sim_func": harmonic_mean_3t,
        "true_theta": jnp.array([3.0, 2.0, 5.5, 16.0]),
        "bounds": [(0.1, 5.0), (0.1, 5.0), (1.0, 20.0), (1.0, 20.0), (0.1, 150.0)],
    },
    "hill_growth": {
        "sim_func": hill_growth,
        "true_theta": jnp.array([1.6, 2.4, 1.0, 5.0]),
        "bounds": [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0), (0.1, 7.0), (0.1, 150.0)],
    },
}

MODELS_TO_RUN = ["gp", "nn"]
STRATEGIES_TO_RUN = ["global", "buffered", "local"]


def normalized_param_distance(posterior_mean, true_theta, param_bounds):
    """sqrt(sum(((mean_i - true_i) / width_i)^2)) using initial bounds."""
    widths = jnp.array([b[1] - b[0] for b in param_bounds])
    return float(jnp.sqrt(jnp.sum(((posterior_mean - true_theta) / widths) ** 2)))


def sim_rmse(posterior_mean, x_obs, y_obs, sim_func):
    """RMSE of simulator predictions at posterior mean vs synthetic observations."""
    y_pred = true_process(x_obs, posterior_mean, sim_func)
    return float(jnp.sqrt(jnp.mean((y_pred - y_obs) ** 2)))


def run_single_parameterization(name, cfg, key):
    """Run the full experiment for one parameterization, return structured results."""
    sim_func = cfg["sim_func"]
    true_theta = cfg["true_theta"]
    bounds = cfg["bounds"]
    n_params = len(true_theta)
    param_bounds = bounds[:-1]  # exclude x
    initial_mcmc_position = jnp.array([(lo + hi) / 2.0 for lo, hi in param_bounds])
    param_labels = [f"$\\theta_{i}$" for i in range(n_params)]

    n_refinement_rounds = n_params
    refinement_range = list(range(2, 2 + n_refinement_rounds))

    print(f"\n{'#'*70}")
    print(f"# PARAMETERIZATION: {name} ({n_params} params, {n_refinement_rounds} refinement rounds)")
    print(f"# True theta: {true_theta}")
    print(f"# Bounds: {bounds}")
    print(f"{'#'*70}")

    # --- Generate observations ---
    x_obs = jnp.geomspace(bounds[-1][0] if bounds[-1][0] > 0 else 0.1, bounds[-1][1], 50)
    key, subkey = jax.random.split(key)
    clean_y = true_process(x_obs, true_theta, sim_func)
    obs_error = 0.05 * jnp.abs(clean_y)
    y_obs = clean_y + obs_error * jax.random.normal(subkey, x_obs.shape)

    # --- Discovery (Round 1) per model type ---
    discovery = {}
    for mtype in MODELS_TO_RUN:
        key, subkey = jax.random.split(key)
        X_R1, y_R1, samples_R1, dur = run_adaptive_discovery(
            f"{mtype.upper()}-Shared", mtype, subkey, bounds,
            x_obs, y_obs, obs_error, true_theta, param_bounds, sim_func,
        )
        discovery[mtype] = {
            "X_R1": X_R1, "y_R1": y_R1, "samples_R1": samples_R1, "duration": dur,
        }
        save_npz(f"{name}_{mtype.upper()}_Discovery_Round_1", samples_R1)

    # --- Initialize solutions ---
    solutions = []
    for mtype in MODELS_TO_RUN:
        d = discovery[mtype]
        for strat in STRATEGIES_TO_RUN:
            sol = IterativeSolution(
                f"{mtype.upper()} {strat.capitalize()}", strat, mtype,
                d["X_R1"], d["y_R1"], d["samples_R1"],
                x_obs, y_obs, obs_error, bounds, sim_func, true_theta,
            )
            sol.timings["Discovery"] = d["duration"]
            solutions.append(sol)

    # --- Refinement rounds ---
    for round_idx in refinement_range:
        print(f"\n{'='*50}")
        print(f"--- ROUND {round_idx} ---")
        print(f"{'='*50}")
        for sol in solutions:
            print(f"\n--- {sol.name} (Round {round_idx}) ---")
            sol.update_priors()
            key, subkey = jax.random.split(key)
            sol.add_round(n_samples_R2, subkey, f"Round {round_idx}")
            key, subkey = jax.random.split(key)
            sol.train(subkey)
            key, subkey = jax.random.split(key)
            sol.run_mcmc(subkey, n_steps=3000, uncertainty_multiplier=2.0)
            sol.samples_history[round_idx] = sol.samples.copy()
            sol.save_samples("posteriors", suffix=f"_{name}_Round_{round_idx}")

    # --- True simulator baseline ---
    true_lp = get_true_logprob(x_obs, y_obs, obs_error, param_bounds, sim_func)
    key, subkey = jax.random.split(key)
    true_smp = np.array(run_mcmc_blackjax(true_lp, initial_mcmc_position, num_steps, subkey)[num_steps // 2:])

    # --- Collect per-round metrics for every solution ---
    all_rounds = [1] + refinement_range
    round_metrics = {}  # key: sol.name, value: list of dicts per round
    for sol in solutions:
        metrics_list = []
        for ridx in all_rounds:
            smp = sol.samples_history.get(ridx)
            if smp is None:
                continue
            mean_smp = jnp.mean(smp, axis=0)
            metrics_list.append({
                "round": ridx,
                "norm_dist": normalized_param_distance(mean_smp, true_theta, param_bounds),
                "sim_rmse": sim_rmse(mean_smp, x_obs, y_obs, sim_func),
                "mean": np.array(mean_smp),
            })
        round_metrics[sol.name] = metrics_list

    # --- Timing ---
    timing_info = {}
    for sol in solutions:
        t = sol.timings
        timing_info[sol.name] = {
            "discovery": t["Discovery"],
            "train": t["Train"],
            "mcmc": t["MCMC"],
            "refinement": t["Refinement"],
            "total": t["Discovery"] + t["Train"] + t["MCMC"] + t["Refinement"],
        }

    # --- Corner plots (per solution, named by parameterization) ---
    os.makedirs("plots", exist_ok=True)
    for sol in solutions:
        fig = corner.corner(
            true_smp, color="k", labels=param_labels,
            show_titles=True, title_fmt=".2f",
            plot_datapoints=False, plot_density=False,
            no_fill_contours=True, smooth=1.0,
            truths=true_theta,
            truth_color=mcolors.to_rgba("tab:red", alpha=0.5),
        )
        cmap = plt.get_cmap("viridis")
        disc_smp = discovery[sol.model_type]["samples_R1"]
        total_plots = 1 + len(refinement_range)  # discovery + refinement rounds
        handles = [
            mlines.Line2D([], [], color="k", label="True Simulator"),
            mlines.Line2D([], [], color=mcolors.to_rgba("tab:red", alpha=0.5), label="Truth"),
        ]

        # Discovery
        c = cmap(0.0 / max(total_plots - 1, 1))
        corner.corner(disc_smp, fig=fig, color=c,
                      hist_kwargs={"ls": "--"}, contour_kwargs={"linestyles": "--"},
                      plot_datapoints=False, plot_density=False,
                      no_fill_contours=True, smooth=1.0)
        d_nd = normalized_param_distance(jnp.mean(disc_smp, axis=0), true_theta, param_bounds)
        handles.append(mlines.Line2D([], [], color=c, ls="--", label=f"R1 Discovery (ND={d_nd:.3f})"))

        # Refinement rounds
        for ci, ridx in enumerate(refinement_range, start=1):
            c = cmap(ci / max(total_plots - 1, 1))
            smp = sol.samples_history[ridx]
            corner.corner(smp, fig=fig, color=c,
                          plot_datapoints=False, plot_density=False,
                          no_fill_contours=True, smooth=1.0)
            nd = normalized_param_distance(jnp.mean(smp, axis=0), true_theta, param_bounds)
            handles.append(mlines.Line2D([], [], color=c, label=f"R{ridx} (ND={nd:.3f})"))

        fig.legend(handles=handles, loc=(0.65, 0.75), fontsize=9)
        plt.suptitle(f"{name}: {sol.name}", fontsize=14, y=1.02)
        fname = f"plots/{name}_{sol.name.replace(' ', '_')}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved {fname}")

    return {
        "name": name,
        "n_params": n_params,
        "true_theta": true_theta,
        "param_bounds": param_bounds,
        "round_metrics": round_metrics,
        "timing": timing_info,
    }


def print_summary(all_results):
    """Print a consolidated summary table across all parameterizations."""
    print("\n" + "=" * 100)
    print("CONSOLIDATED RESULTS")
    print("=" * 100)

    for res in all_results:
        name = res["name"]
        n_params = res["n_params"]
        print(f"\n{'─'*100}")
        print(f"  {name.upper()} ({n_params} params, true={np.array(res['true_theta'])})")
        print(f"{'─'*100}")

        # Round-by-round progression
        print(f"\n  {'Solution':<20} | {'Round':>5} | {'Norm Dist':>10} | {'Sim RMSE':>10} | {'Mean Theta'}")
        print(f"  {'-'*85}")
        for sol_name, metrics_list in res["round_metrics"].items():
            for m in metrics_list:
                mean_str = ", ".join(f"{v:.3f}" for v in m["mean"])
                print(f"  {sol_name:<20} | {m['round']:>5} | {m['norm_dist']:>10.4f} | {m['sim_rmse']:>10.4f} | [{mean_str}]")

        # Timing
        print(f"\n  {'Solution':<20} | {'Discovery':>10} | {'Train':>10} | {'MCMC':>10} | {'Refine':>10} | {'Total':>10}")
        print(f"  {'-'*85}")
        for sol_name, t in res["timing"].items():
            print(f"  {sol_name:<20} | {t['discovery']:>10.1f} | {t['train']:>10.1f} | {t['mcmc']:>10.1f} | {t['refinement']:>10.1f} | {t['total']:>10.1f}")

    # Final comparison: best per parameterization
    print(f"\n{'='*100}")
    print("BEST FINAL ROUND (lowest normalized distance)")
    print(f"{'='*100}")
    print(f"  {'Parameterization':<20} | {'Best Solution':<20} | {'Norm Dist':>10} | {'Sim RMSE':>10} | {'Time (s)':>10}")
    print(f"  {'-'*85}")
    for res in all_results:
        best_name, best_nd, best_rmse = None, float("inf"), None
        for sol_name, metrics_list in res["round_metrics"].items():
            final = metrics_list[-1]
            if final["norm_dist"] < best_nd:
                best_name = sol_name
                best_nd = final["norm_dist"]
                best_rmse = final["sim_rmse"]
        best_time = res["timing"][best_name]["total"]
        print(f"  {res['name']:<20} | {best_name:<20} | {best_nd:>10.4f} | {best_rmse:>10.4f} | {best_time:>10.1f}")

    # --- Write CSV files per parameterization ---
    os.makedirs("results", exist_ok=True)
    for res in all_results:
        pname = res["name"]
        n_params = res["n_params"]

        # Round metrics CSV
        metrics_path = f"results/{pname}_metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            theta_cols = [f"mean_theta_{i}" for i in range(n_params)]
            writer.writerow(["solution", "model_type", "strategy", "round", "norm_dist", "sim_rmse"] + theta_cols)
            for sol_name, metrics_list in res["round_metrics"].items():
                parts = sol_name.split(" ", 1)
                mtype = parts[0].lower()
                strat = parts[1].lower() if len(parts) > 1 else ""
                for m in metrics_list:
                    row = [sol_name, mtype, strat, m["round"], f"{m['norm_dist']:.6f}", f"{m['sim_rmse']:.6f}"]
                    row += [f"{v:.6f}" for v in m["mean"]]
                    writer.writerow(row)
        print(f"  Saved {metrics_path}")

        # Timing CSV
        timing_path = f"results/{pname}_timing.csv"
        with open(timing_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["solution", "model_type", "strategy", "discovery_s", "train_s", "mcmc_s", "refinement_s", "total_s"])
            for sol_name, t in res["timing"].items():
                parts = sol_name.split(" ", 1)
                mtype = parts[0].lower()
                strat = parts[1].lower() if len(parts) > 1 else ""
                writer.writerow([sol_name, mtype, strat,
                                 f"{t['discovery']:.1f}", f"{t['train']:.1f}",
                                 f"{t['mcmc']:.1f}", f"{t['refinement']:.1f}",
                                 f"{t['total']:.1f}"])
        print(f"  Saved {timing_path}")


def main():
    import sys
    key = jax.random.PRNGKey(42)

    # Allow running a single parameterization via CLI arg
    if len(sys.argv) > 1:
        names_to_run = sys.argv[1:]
        for n in names_to_run:
            if n not in PARAMETERIZATIONS:
                print(f"Unknown parameterization: {n}. Options: {list(PARAMETERIZATIONS.keys())}")
                sys.exit(1)
    else:
        names_to_run = list(PARAMETERIZATIONS.keys())

    all_results = []
    for pname in names_to_run:
        cfg = PARAMETERIZATIONS[pname]
        key, subkey = jax.random.split(key)
        result = run_single_parameterization(pname, cfg, subkey)
        all_results.append(result)

    print_summary(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
