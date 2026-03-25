"""
Runner script: evaluates multiple parameterizations (power_law, harmonic_mean, harmonic_mean_3t, hill_growth).

Usage:
    python run_all_parameterizations.py                          # runs all parameterizations
    python run_all_parameterizations.py power_law harmonic_mean  # runs a subset
"""
import sys
from main_experiment import PARAMETERIZATIONS, run_experiment


def print_consolidated_summary(all_results):
    """Print best-per-parameterization comparison across all runs."""
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


def main():
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
        result = run_experiment(param_name=pname)
        all_results.append(result)

    if len(all_results) > 1:
        print_consolidated_summary(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
