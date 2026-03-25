import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import corner
from simulator import simulator, true_process
from emulators import train_gp_jax, predict_gp_jax, tune_and_train_nn_jax, predict_nn_jax
from mcmc_sampler import get_true_logprob, get_gp_logprob, get_nn_logprob, run_mcmc_blackjax
 
# Experiment configuration
# Set to None to auto-compute based on n_params:
#   n_ppe_r1 = 50 * n_params², n_ppe_r2 = n_ppe_r1 // 10,
#   n_refinement_rounds = n_params
n_ppe_r1 = None
n_ppe_r2 = None
n_refinement_rounds = None
n_mcmc_samples = 2500
UNCERTAINTY_MULTIPLIER = 3.0  # Inflate emulator uncertainty to prevent over-tightening

# -------------------------------------------------------------
# Simulator functions
# -------------------------------------------------------------
def power_law(theta, x):
    """ y = theta_0 * x^theta_1 """
    return theta[:, 0] * jnp.power(x, theta[:, 1])

def harmonic_mean(theta, x):
    """ vt = a * (v1**(-k) + v2**(-k))**(-1/k) """
    a = theta[:, 0]
    k = theta[:, 1]
    v1 = jnp.power(x, 2/3.)
    v2 = jnp.power(x, 1/6.) * jnp.power(theta[:, 2], 1/2.)
    vt = a * (v1**(-k) + v2**(-k))**(-1/k)
    return vt

def harmonic_mean_3t(theta, x):
    """ vt = a * (v1**(-k) + v2**(-k) + v3**(-k))**(-1/k) """
    a = theta[:, 0]
    k = theta[:, 1]
    v1 = jnp.power(x, 2/3.)
    v2 = jnp.power(x, 1/3.) * jnp.power(theta[:, 2], 1/3.)
    v3 = jnp.power(x, 1/6.) * jnp.power(theta[:, 3], 1/2.)
    vt = a * (v1**(-k) + v2**(-k) + v3**(-k))**(-1/k)
    return vt

def hill_growth(theta, x):
    """ y = theta_0 * (x^theta_1/(x^theta_2 + theta_3)) """
    return theta[:, 0] * (jnp.power(x, theta[:, 1]) / (jnp.power(x, theta[:, 2]) + theta[:, 3]))

# ---------------------------------------------------------------------------
# Parameterization configs
# ---------------------------------------------------------------------------
PARAMETERIZATIONS = {
    "power_law": {
        "sim_func": power_law,
        "true_theta": jnp.array([1.6, 2.4]),
        "theta_bounds": [(0.1, 5.0), (0.1, 5.0)],
        "x_bounds": (0.1, 10.0),
    },
    "harmonic_mean": {
        "sim_func": harmonic_mean,
        "true_theta": jnp.array([3.0, 2.0, 16.0]),
        "theta_bounds": [(0.1, 5.0), (0.1, 5.0), (1.0, 20.0)],
        "x_bounds": (0.1, 150.0),
    },
    "harmonic_mean_3t": {
        "sim_func": harmonic_mean_3t,
        "true_theta": jnp.array([3.0, 2.0, 5.5, 16.0]),
        "theta_bounds": [(0.1, 5.0), (0.1, 5.0), (1.0, 20.0), (1.0, 20.0)],
        "x_bounds": (0.1, 250.0),
    },
    "hill_growth": {
        "sim_func": hill_growth,
        "true_theta": jnp.array([1.6, 2.4, 1.0, 5.0]),
        "theta_bounds": [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0), (0.1, 7.0)],
        "x_bounds": (0.1, 150.0),
    },
}

def save_npz(name, samples, directory='posteriors'):
    """Helper to save MCMC samples to a permanent record."""
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{name.replace(' ', '_')}_samples.npz")
    np.savez(filename, samples=samples, labels=np.array([f"theta_{i}" for i in range(samples.shape[1])]))
    print(f"  [Persistence] Saved {filename}")

def is_posterior_wide_enough(samples, prior_bounds, threshold=0.15):
    """
    Realistic heuristic: Broaden until the average marginal std is at least 
    target fraction of the initial prior range. This ensures exploration.
    """
    stds = np.std(samples, axis=0)
    widths = np.array([b[1] - b[0] for b in prior_bounds])
    fraction = np.mean(stds / widths)
    return fraction >= threshold, fraction

def generate_ppes(n_ppe, bounds, key, sim_func):
    """Uniformly samples theta and x within given bounds, then runs the simulator."""
    columns = []
    for b in bounds:
        key, subkey = jax.random.split(key)
        columns.append(jax.random.uniform(subkey, shape=(n_ppe, 1), minval=b[0], maxval=b[1]))
    X = jnp.column_stack(columns)
    y = simulator(X, sim_func)
    return X, y

# Real-time Convergence Diagnostic
def print_distance_to_truth(samples, true_theta, context_name, x_obs, y_obs, sim_func):
    mean_smp = jnp.mean(samples, axis=0)
    theta_dist = jnp.sqrt(jnp.sum((mean_smp - true_theta)**2))
    
    # Calculate Simulator RMSE at the mean of the posterior
    y_pred = true_process(x_obs, mean_smp, sim_func)
    sim_rmse = jnp.sqrt(jnp.mean((y_pred - y_obs)**2))
    
    print(f"  [{context_name}] Dist: {theta_dist:.4f} | Sim RMSE: {sim_rmse:.4f} | Mean: {mean_smp}")
    return theta_dist, sim_rmse

def run_adaptive_discovery(name, model_type, key, bounds_R1, x_obs, y_obs, obs_error, true_theta, mcmc_prior_bounds, sim_func, n_ppe_override=None):
    """Performs the first adaptive discovery round for a given solution type."""
    start_t = time.time()
    print(f"\n--- Running Round 1 Discovery for {name} ({model_type.upper()}) ---")
    if n_ppe_override is not None:
        n_ppe = n_ppe_override
    elif n_ppe_r1 is not None:
        n_ppe = n_ppe_r1
    else:
        n_params = len(mcmc_prior_bounds)
        n_ppe = 50 * n_params ** 2
    key, subkey = jax.random.split(key)
    X_R1, y_R1 = generate_ppes(n_ppe, bounds_R1, subkey, sim_func)
    
    # Subsample for Discovery (Reduced to 10 to allow effective broadening)
    n_sub = 10
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, len(x_obs), (n_sub,), replace=False)
    x_sub = x_obs[indices]
    y_sub = y_obs[indices]
    err_sub = obs_error[indices] if not isinstance(obs_error, (int, float)) else obs_error
    
    broadening = 0.0
    max_broadening = 4.0
    step = 0.5

    print(f"  [{name}] Training Round 1 Discovery emulator ({model_type.upper()})...")
    if model_type == 'gp':
        key, subkey = jax.random.split(key)
        emulator_params = train_gp_jax(X_R1, y_R1, key=subkey)
    elif model_type == 'nn':
        emulator_params = tune_and_train_nn_jax(X_R1, y_R1, key, num_searches=10) # Faster discovery
    
    while broadening <= max_broadening:
        # Use the single uncertainty_multiplier knob (broadened by 1 + broadening)
        # and scaled by sqrt(N) in the sampler.
        current_mult = UNCERTAINTY_MULTIPLIER * (1.0 + broadening)
        print(f"  [{name}] Testing Discovery MCMC with {current_mult:.1f} base multiplier (n_ppe={n_ppe})...")

        if model_type == 'gp':
            lp_fn = get_gp_logprob(predict_gp_jax, emulator_params[0], X_R1, y_R1,
                                  emulator_params[1], emulator_params[2], emulator_params[3], emulator_params[4],
                                  x_sub, y_sub, err_sub, mcmc_prior_bounds,
                                  uncertainty_multiplier=current_mult)
        elif model_type == 'nn':
            lp_fn = get_nn_logprob(predict_nn_jax, emulator_params, x_sub, y_sub, err_sub,
                                   mcmc_prior_bounds, uncertainty_multiplier=current_mult)
            
        key, subkey = jax.random.split(key)
        init_pos = jnp.array([(lo + hi) / 2.0 for lo, hi in mcmc_prior_bounds])
        samples = run_mcmc_blackjax(lp_fn, init_pos, 3000, subkey)
        samples = np.array(samples[1000:])
        
        l_wide, fraction = is_posterior_wide_enough(samples, mcmc_prior_bounds, threshold=0.15)
        if l_wide:
            print(f"  [{name}] [SUCCESS] Round 1 achieves discovery width at multiplier {current_mult:.1f}.")
            print_distance_to_truth(samples, true_theta, f"{name} R1 Final", x_obs, y_obs, sim_func)
            save_npz(f"{name}_Round_1", samples)
            duration = time.time() - start_t
            return X_R1, y_R1, samples, duration
        else:
            print(f"  [{name}] [FAILURE] Posterior is too tight, fraction: {fraction:0.2f} vs. threshold = 0.15. Increasing broadening...")
            broadening += step
            
    duration = time.time() - start_t
    return X_R1, y_R1, samples, duration

class IterativeSolution:
    """Encapsulates a specific emulation strategy (e.g. GP Buffered) and its 3-round journey."""
    def __init__(self, name, strategy_type, model_type, X_R1, y_R1, samples_R1, x_obs, y_obs, obs_error, theta_bounds, x_bounds, sim_func, true_theta, buffer_margin=0.2):
        self.name = name
        self.strategy_type = strategy_type
        self.model_type = model_type
        self.theta_bounds = theta_bounds
        self.mcmc_prior_bounds = list(theta_bounds)
        self.x_range = (x_bounds[0], x_bounds[1])
        self.history = [(X_R1, y_R1)] # List of (X, y) per round
        self.samples = samples_R1
        self.samples_history = {1: samples_R1.copy()}
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.obs_error = obs_error
        self.sim_func = sim_func
        self.true_theta = true_theta
        
        # Domain-Specific Data (Managed in self.train)
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        
        # Performance Tracking
        self.timings = {"Discovery": 0.0, "Train": 0.0, "MCMC": 0.0, "Refinement": 0.0}
        self.n_train_history = {}  # round_idx -> n_train used
        self.buffer_margin = buffer_margin
        self.params = None

    def update_priors(self):
        """Updates MCMC prior bounds based on the samples of the previous round."""
        if self.strategy_type == 'global' or self.samples is None:
            return
            
        if self.strategy_type == 'local':
            # Dynamic Shrinking: strict [min, max]
            lower = jnp.min(self.samples, axis=0)
            upper = jnp.max(self.samples, axis=0)
            self.mcmc_prior_bounds = [(float(l), float(u)) for l, u in zip(lower, upper)]
            print(f"  [{self.name}] Priors shrunk to min/max of previous samples.")
        
        elif self.strategy_type == 'buffered':
            # Buffered: 3*sigma + 20% total margin
            means = jnp.mean(self.samples, axis=0)
            stds = jnp.std(self.samples, axis=0)
            prior_lower = jnp.array([b[0] for b in self.theta_bounds])
            prior_upper = jnp.array([b[1] for b in self.theta_bounds])
            prior_width = prior_upper - prior_lower
            
            # (3 sigma) + (buffer_margin * total prior width) on each side
            lower_B = jnp.maximum(means - 3 * stds - self.buffer_margin * prior_width, prior_lower)
            upper_B = jnp.minimum(means + 3 * stds + self.buffer_margin * prior_width, prior_upper)
            self.mcmc_prior_bounds = [(float(l), float(u)) for l, u in zip(lower_B, upper_B)]
            print(f"  [{self.name}] Priors set to 3-sigma + {self.buffer_margin:.0%} margin.")
            print(f"  [{self.name}] DIAG: posterior mean={[f'{v:.3f}' for v in means]}, std={[f'{v:.3f}' for v in stds]}")
            print(f"  [{self.name}] DIAG: new bounds={[(f'{float(l):.3f}', f'{float(u):.3f}') for l, u in zip(lower_B, upper_B)]}")

    @property
    def X_all(self):
        return jnp.vstack([h[0] for h in self.history])
    @property
    def y_all(self):
        return jnp.concatenate([h[1] for h in self.history])

    def train(self, key):
        """Constructs the strategy-specific training set and trains the model with a val split."""
        start_t = time.time()
        # 1. Source all points available to this strategy
        if self.strategy_type == 'global':
            X_all, y_all = self.X_all, self.y_all
        elif self.strategy_type == 'local':
            # All accumulated PPEs strictly within current local bounds
            X_accum = self.X_all
            y_accum = self.y_all
            lower = jnp.array([b[0] for b in self.mcmc_prior_bounds])
            upper = jnp.array([b[1] for b in self.mcmc_prior_bounds])
            in_local = jnp.all((X_accum[:, :len(lower)] >= lower) &
                               (X_accum[:, :len(lower)] <= upper), axis=1)
            X_all = X_accum[in_local]
            y_all = y_accum[in_local]
            print(f"  [{self.name}] DIAG: Bounds lower={[f'{v:.3f}' for v in lower]}, upper={[f'{v:.3f}' for v in upper]}")
            print(f"  [{self.name}] DIAG: {int(in_local.sum())}/{len(y_accum)} points in local bounds")
        elif self.strategy_type == 'buffered':
            X_accum = self.X_all
            y_accum = self.y_all
            lower = jnp.array([b[0] for b in self.mcmc_prior_bounds])
            upper = jnp.array([b[1] for b in self.mcmc_prior_bounds])
            in_buffer = jnp.all((X_accum[:, :len(lower)] >= lower) &
                                (X_accum[:, :len(lower)] <= upper), axis=1)
            X_all = X_accum[in_buffer]
            y_all = y_accum[in_buffer]
            print(f"  [{self.name}] DIAG: Bounds lower={[f'{v:.3f}' for v in lower]}, upper={[f'{v:.3f}' for v in upper]}")
            print(f"  [{self.name}] DIAG: {int(in_buffer.sum())}/{len(y_accum)} points in buffer, y range=[{float(y_all.min()):.3f}, {float(y_all.max()):.3f}]")
            for dim in range(lower.shape[0]):
                print(f"  [{self.name}] DIAG:   dim{dim}: X range=[{float(X_all[:,dim].min()):.3f}, {float(X_all[:,dim].max()):.3f}], width={float(upper[dim]-lower[dim]):.3f}")
        else:
            X_all, y_all = self.X_all, self.y_all
            
        # 2. 80/20 Shuffled Train/Val Split
        n = len(y_all)
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, n)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]
        
        self.X_train, self.y_train = X_all[train_idx], y_all[train_idx]
        self.X_val, self.y_val = X_all[val_idx], y_all[val_idx]
        
        # 3. Fit
        if self.model_type == 'gp':
            key, subkey = jax.random.split(key)
            self.params = train_gp_jax(self.X_train, self.y_train, key=subkey)
            gp_p = self.params[0]
            print(f"  [{self.name}] DIAG GP: log_scale={float(gp_p['log_scale']):.3f}, log_noise={float(gp_p['log_noise']):.3f}, log_length={[f'{v:.3f}' for v in gp_p['log_length']]}")
            # Test GP accuracy on validation set
            if self.X_val is not None and len(self.y_val) > 0:
                yp, ystd = predict_gp_jax(self.X_val, self.params[0], self.X_train, self.y_train, *self.params[1:])
                rmse = float(jnp.sqrt(jnp.mean((self.y_val - yp)**2)))
                mean_std = float(jnp.mean(ystd))
                print(f"  [{self.name}] DIAG GP val: RMSE={rmse:.4f}, mean_pred_std={mean_std:.4f}, n_train={len(self.y_train)}, n_val={len(self.y_val)}")
        elif self.model_type == 'nn':
            key, subkey = jax.random.split(key)
            self.params = tune_and_train_nn_jax(self.X_train, self.y_train, subkey, num_searches=40)

        print(f"  [{self.name}] n_train={len(self.y_train)}, n_val={len(self.y_val)}")
        self.timings["Train"] += time.time() - start_t

    def get_logprob_fn(self, uncertainty_multiplier=UNCERTAINTY_MULTIPLIER):
        if self.model_type == 'gp':
            return get_gp_logprob(predict_gp_jax, self.params[0], self.X_train, self.y_train,
                                 self.params[1], self.params[2], self.params[3], self.params[4],
                                 self.x_obs, self.y_obs, self.obs_error, self.mcmc_prior_bounds,
                                 uncertainty_multiplier=uncertainty_multiplier)
        elif self.model_type == 'nn':
            return get_nn_logprob(predict_nn_jax, self.params, self.x_obs, self.y_obs,
                                 self.obs_error, self.mcmc_prior_bounds,
                                 uncertainty_multiplier=uncertainty_multiplier)

    def predict(self, X_new):
        """Unified prediction interface that handles model-specific signatures."""
        if self.model_type == 'gp':
            return predict_gp_jax(X_new, self.params[0], self.X_train, self.y_train, *self.params[1:5])
        elif self.model_type == 'nn':
            return predict_nn_jax(X_new, *self.params)

    def run_mcmc(self, key, initial_pos=None, n_steps=3000, uncertainty_multiplier=1.0):
        start_t = time.time()
        if initial_pos is None:
            # Sequential initialization based on previous posterior mean
            initial_pos = jnp.mean(self.samples, axis=0)
            
        lp_fn = self.get_logprob_fn(uncertainty_multiplier=uncertainty_multiplier)
        self.samples = np.array(run_mcmc_blackjax(lp_fn, initial_pos, n_steps, key)[n_steps//2:])
        print_distance_to_truth(self.samples, self.true_theta, f"{self.name} Current", self.x_obs, self.y_obs, self.sim_func)
        self.timings["MCMC"] += time.time() - start_t
        return self.samples

    def add_round(self, n_ppe, key, round_name):
        start_t = time.time()
        print(f"  [{self.name}] Adding {round_name} PPE ({n_ppe} points)...")
        key, subkey = jax.random.split(key)

        # Posterior-Weighted Sampling for all refining rounds
        # We sample directly from the most recent posterior mass to ensure density
        # where the likelihood is highest, preventing emulator "drift".
        n_available = len(self.samples)
        idx = jax.random.choice(subkey, n_available, shape=(n_ppe,))
        theta_new = self.samples[idx]

        # Diagnostic: check how concentrated the new PPE samples are
        theta_std = jnp.std(theta_new, axis=0)
        theta_range = jnp.ptp(theta_new, axis=0)
        print(f"  [{self.name}] DIAG PPE: new theta std={[f'{v:.4f}' for v in theta_std]}, range={[f'{v:.4f}' for v in theta_range]}")

        x_new = jnp.geomspace(self.x_range[0], self.x_range[1], n_ppe).reshape(-1, 1)
        X_new = jnp.column_stack((theta_new, x_new))
        y_new = simulator(X_new, self.sim_func)
        self.history.append((X_new, y_new))
        self.timings["Refinement"] += time.time() - start_t

    def save_samples(self, directory, suffix=""):
        """Saves final MCMC samples to a numpy file for downstream analysis."""
        os.makedirs(directory, exist_ok=True)
        name_clean = self.name.replace(' ', '_')
        filename = os.path.join(directory, f"{name_clean}{suffix}_samples.npz")
        np.savez(filename, samples=self.samples, labels=np.array([f"theta_{i}" for i in range(self.samples.shape[1])]))
        print(f"  [{self.name}] Samples saved to {filename}")

def check_coverage(samples, truth, margin=0.025):
    """Checks if the truth is within the [margin, 1-margin] percentile bounds of the samples."""
    lower = np.percentile(samples, margin * 100, axis=0)
    upper = np.percentile(samples, (1-margin) * 100, axis=0)
    covered = np.all((truth >= lower) & (truth <= upper))
    return covered, lower, upper

# =====================================================================
# SHARED OUTPUT ROUTINES
# =====================================================================

STYLE_COLORS = {
    "GP Global": "tab:blue", "GP Buffered": "tab:pink", "GP Local": "tab:orange",
    "NN Global": "tab:cyan", "NN Buffered": "tab:green", "NN Local": "tab:purple",
}

def normalized_param_distance(posterior_mean, true_theta, param_bounds):
    """sqrt(sum(((mean_i - true_i) / width_i)^2)) using initial bounds."""
    widths = jnp.array([b[1] - b[0] for b in param_bounds])
    return float(jnp.sqrt(jnp.sum(((posterior_mean - true_theta) / widths) ** 2)))


def sim_rmse(posterior_mean, x_obs, y_obs, sim_func):
    """RMSE of simulator predictions at posterior mean vs synthetic observations."""
    y_pred = true_process(x_obs, posterior_mean, sim_func)
    return float(jnp.sqrt(jnp.mean((y_pred - y_obs) ** 2)))


def collect_round_metrics(solutions, all_rounds, true_theta, param_bounds, x_obs, y_obs, sim_func, n_ppe_r1):
    """Collect per-round metrics for every solution. Returns dict: sol.name -> list of dicts."""
    round_metrics = {}
    for sol in solutions:
        metrics_list = []
        for ridx in all_rounds:
            smp = sol.samples_history.get(ridx)
            if smp is None:
                continue
            mean_smp = jnp.mean(smp, axis=0)
            metrics_list.append({
                "round": ridx,
                "n_train": sol.n_train_history.get(ridx, int(0.8 * n_ppe_r1)),
                "norm_dist": normalized_param_distance(mean_smp, true_theta, param_bounds),
                "sim_rmse": sim_rmse(mean_smp, x_obs, y_obs, sim_func),
                "mean": np.array(mean_smp),
            })
        round_metrics[sol.name] = metrics_list
    return round_metrics


def collect_timing(solutions):
    """Collect timing info for every solution. Returns dict: sol.name -> timing dict."""
    timing_info = {}
    for sol in solutions:
        t = sol.timings
        timing_info[sol.name] = {
            "discovery": t["Discovery"],
            "train": t["Train"],
            "mcmc": t["MCMC"],
            "total": t["Discovery"] + t["Train"] + t["MCMC"],
        }
    return timing_info


def collect_emulator_perf(solutions, round_idx):
    """Capture emulator validation predictions for a given round. Returns dict: sol.name -> perf dict."""
    perf = {}
    for sol in solutions:
        if sol.X_val is not None and len(sol.y_val) > 0:
            yp, _ = sol.predict(sol.X_val)
            yt = sol.y_val
            rmse_val = float(jnp.sqrt(jnp.mean((yp - yt)**2)))
            mas_val = float(1 - jnp.sum((yt - yp)**2) / jnp.sum((yt - jnp.mean(yt))**2))
            perf[sol.name] = {
                "y_true": np.array(yt), "y_pred": np.array(yp),
                "rmse": rmse_val, "mas": mas_val,
            }
    return perf


def plot_corner(sol, true_smp, discovery_smp, true_theta, param_bounds, param_labels,
                refinement_range, prefix_name):
    """Generate corner plot for one solution overlaying all rounds."""
    os.makedirs("plots", exist_ok=True)
    fig = corner.corner(
        true_smp, color="k", labels=param_labels,
        show_titles=True, title_fmt=".2f",
        plot_datapoints=False, plot_density=False,
        no_fill_contours=True, smooth=1.0,
        truths=true_theta,
        truth_color=mcolors.to_rgba("tab:red", alpha=0.5),
    )
    cmap = plt.get_cmap("viridis")
    total_plots = 1 + len(refinement_range)
    handles = [
        mlines.Line2D([], [], color="k", label="True Simulator"),
        mlines.Line2D([], [], color=mcolors.to_rgba("tab:red", alpha=0.5), label="Truth"),
    ]
    # Discovery
    c = cmap(0.0 / max(total_plots - 1, 1))
    corner.corner(discovery_smp, fig=fig, color=c,
                  hist_kwargs={"ls": "--"}, contour_kwargs={"linestyles": "--"},
                  plot_datapoints=False, plot_density=False,
                  no_fill_contours=True, smooth=1.0)
    d_nd = normalized_param_distance(jnp.mean(discovery_smp, axis=0), true_theta, param_bounds)
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
    plt.suptitle(f"{prefix_name}: {sol.name}", fontsize=14, y=1.02)
    fname = f"plots/{prefix_name}_{sol.name.replace(' ', '_')}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {fname}")


def plot_emulator_performance(solutions, emulator_perf, round_idx, prefix_name):
    """Generate emulator parity plot for one round."""
    os.makedirs("plots", exist_ok=True)
    round_sols = [(sol, emulator_perf[sol.name])
                  for sol in solutions if sol.name in emulator_perf]
    if not round_sols:
        return
    n_plots = len(round_sols)
    ncols = min(n_plots, 3)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for i, (sol, perf) in enumerate(round_sols):
        ax = axes_flat[i]
        yt, yp = perf["y_true"], perf["y_pred"]
        color = STYLE_COLORS.get(sol.name, "tab:blue")
        ax.scatter(yt[::2], yp[::2], alpha=0.5, s=15, color=color)
        all_vals = np.concatenate([yt, yp])
        vmin, vmax = max(np.nanmin(all_vals), 1e-1), np.nanmax(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)
        # MAS = Model Accuracy Score, 
        # similar to R² but residual is calculated from "truth", whatever that means
        # technically called Nash–Sutcliffe efficiency but that's a terrible name - ah
        ax.set_title(f"{sol.name}\nRMSE={perf['rmse']:.3f}, MAS={perf['mas']:.4f}", fontsize=11)
        ax.set_xlabel("True $y$")
        ax.set_ylabel("Predicted $y$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"{prefix_name}: Emulator Performance (Round {round_idx})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"plots/{prefix_name}_emulator_performance_R{round_idx}.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  -> Saved {fname}")


def write_csvs(prefix_name, n_params, round_metrics, timing_info):
    """Write metrics and timing CSV files for one parameterization."""
    import csv
    os.makedirs("results", exist_ok=True)

    metrics_path = f"results/{prefix_name}_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        theta_cols = [f"mean_theta_{i}" for i in range(n_params)]
        writer.writerow(["solution", "model_type", "strategy", "round", "n_train", "norm_dist", "sim_rmse"] + theta_cols)
        for sol_name, metrics_list in round_metrics.items():
            parts = sol_name.split(" ", 1)
            mtype = parts[0].lower()
            strat = parts[1].lower() if len(parts) > 1 else ""
            for m in metrics_list:
                row = [sol_name, mtype, strat, m["round"], m["n_train"], f"{m['norm_dist']:.6f}", f"{m['sim_rmse']:.6f}"]
                row += [f"{v:.6f}" for v in m["mean"]]
                writer.writerow(row)
    print(f"  Saved {metrics_path}")

    timing_path = f"results/{prefix_name}_timing.csv"
    with open(timing_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solution", "model_type", "strategy", "discovery_s", "train_s", "mcmc_s", "total_s"])
        for sol_name, t in timing_info.items():
            parts = sol_name.split(" ", 1)
            mtype = parts[0].lower()
            strat = parts[1].lower() if len(parts) > 1 else ""
            writer.writerow([sol_name, mtype, strat,
                             f"{t['discovery']:.1f}", f"{t['train']:.1f}",
                             f"{t['mcmc']:.1f}", f"{t['total']:.1f}"])
    print(f"  Saved {timing_path}")


def print_single_summary(prefix_name, n_params, true_theta, param_bounds, round_metrics, timing_info):
    """Print summary table for one parameterization."""
    print(f"\n{'─'*100}")
    print(f"  {prefix_name.upper()} ({n_params} params, true={np.array(true_theta)})")
    print(f"{'─'*100}")

    print(f"\n  {'Solution':<20} | {'Round':>5} | {'N Train':>7} | {'Norm Dist':>10} | {'Sim RMSE':>10} | {'Mean Theta'}")
    print(f"  {'-'*90}")
    for sol_name, metrics_list in round_metrics.items():
        for m in metrics_list:
            mean_str = ", ".join(f"{v:.3f}" for v in m["mean"])
            print(f"  {sol_name:<20} | {m['round']:>5} | {m['n_train']:>7} | {m['norm_dist']:>10.4f} | {m['sim_rmse']:>10.4f} | [{mean_str}]")

    print(f"\n  {'Solution':<20} | {'Discovery':>10} | {'Train':>10} | {'MCMC':>10} | {'Total':>10}")
    print(f"  {'-'*75}")
    for sol_name, t in timing_info.items():
        print(f"  {sol_name:<20} | {t['discovery']:>10.1f} | {t['train']:>10.1f} | {t['mcmc']:>10.1f} | {t['total']:>10.1f}")

def run_experiment(param_name="harmonic_mean", sim_func=None, true_theta=None,
                   theta_bounds=None, x_bounds=None, prefix_name=None):
    """Run the full experiment pipeline for one parameterization.

    Can be called by name (looks up PARAMETERIZATIONS) or with explicit overrides.
    """
    key = jax.random.PRNGKey(42)

    # Look up parameterization by name, allow explicit overrides
    cfg = PARAMETERIZATIONS.get(param_name, {})
    SIM_FUNC = sim_func if sim_func is not None else cfg["sim_func"]
    TRUE_THETA = true_theta if true_theta is not None else cfg["true_theta"]
    THETA_BOUNDS = theta_bounds if theta_bounds is not None else cfg["theta_bounds"]
    X_BOUNDS = x_bounds if x_bounds is not None else cfg["x_bounds"]
    if prefix_name is None:
        prefix_name = param_name

    n_params = len(TRUE_THETA)
    all_bounds = THETA_BOUNDS + [X_BOUNDS]  # combined for generate_ppes
    initial_mcmc_position = jnp.array([(lo + hi) / 2.0 for lo, hi in THETA_BOUNDS])
    param_labels = [f"$\\theta_{i}$" for i in range(n_params)]

    _n_ppe_r1 = n_ppe_r1 if n_ppe_r1 is not None else 50 * n_params ** 2
    _n_ppe_r2 = n_ppe_r2 if n_ppe_r2 is not None else _n_ppe_r1 // 10
    _n_refinement_rounds = n_refinement_rounds if n_refinement_rounds is not None else n_params
    refinement_range = list(range(2, 2 + _n_refinement_rounds))

    # -------------------------------------------------------------
    # 0. RUN CONFIGURATION
    # -------------------------------------------------------------
    MODELS_TO_RUN = ["gp", "nn"]  # Options: "gp", "nn"
    STRATEGIES_TO_RUN = ["global", "buffered", "local"]

    print(f"\n{'#'*70}")
    print(f"# {prefix_name.upper()} ({n_params} params, {_n_refinement_rounds} refinement rounds)")
    print(f"# True theta: {TRUE_THETA}")
    print(f"# Theta bounds: {THETA_BOUNDS}")
    print(f"# X bounds: {X_BOUNDS}")
    print(f"# n_ppe_r1: {_n_ppe_r1}, n_ppe_r2: {_n_ppe_r2}")
    print(f"# Models: {MODELS_TO_RUN}, Strategies: {STRATEGIES_TO_RUN}")
    print(f"{'#'*70}")

    # 1. GENERATE OBSERVATION DATA
    x_obs = jnp.geomspace(X_BOUNDS[0] if X_BOUNDS[0] > 0 else 0.1, X_BOUNDS[1], 50)
    key, subkey = jax.random.split(key)
    clean_y = true_process(x_obs, TRUE_THETA, SIM_FUNC)
    obs_error = 0.05 * jnp.abs(clean_y)
    y_obs = clean_y + obs_error * jax.random.normal(subkey, x_obs.shape)

    # 2. DISCOVERY (Round 1) per model type
    discovery = {}
    for mtype in MODELS_TO_RUN:
        key, subkey = jax.random.split(key)
        X_R1, y_R1, samples_R1, dur = run_adaptive_discovery(
            f"{mtype.upper()}-Shared", mtype, subkey, all_bounds,
            x_obs, y_obs, obs_error, TRUE_THETA, THETA_BOUNDS, SIM_FUNC,
            n_ppe_override=_n_ppe_r1,
        )
        discovery[mtype] = {
            "X_R1": X_R1, "y_R1": y_R1, "samples_R1": samples_R1, "duration": dur,
        }
        save_npz(f"{prefix_name}_{mtype.upper()}_Discovery_Round_1", samples_R1)

    # 3. INITIALIZE SOLUTIONS
    solutions = []
    for mtype in MODELS_TO_RUN:
        d = discovery[mtype]
        for strat in STRATEGIES_TO_RUN:
            sol = IterativeSolution(
                f"{mtype.upper()} {strat.capitalize()}", strat, mtype,
                d["X_R1"], d["y_R1"], d["samples_R1"],
                x_obs, y_obs, obs_error, THETA_BOUNDS, X_BOUNDS, SIM_FUNC, TRUE_THETA,
            )
            sol.timings["Discovery"] = d["duration"]
            solutions.append(sol)

    # 4. REFINEMENT ROUNDS
    emulator_perf = {}
    for round_idx in refinement_range:
        print(f"\n{'='*50}")
        print(f"--- ROUND {round_idx} ---")
        print(f"{'='*50}")
        for sol in solutions:
            print(f"\n--- {sol.name} (Round {round_idx}) ---")
            sol.update_priors()
            key, subkey = jax.random.split(key)
            sol.add_round(_n_ppe_r2, subkey, f"Round {round_idx}")
            key, subkey = jax.random.split(key)
            sol.train(subkey)
            sol.n_train_history[round_idx] = len(sol.y_train)

            # Capture emulator performance
            perf = collect_emulator_perf([sol], round_idx)
            for k, v in perf.items():
                emulator_perf[(k, round_idx)] = v

            key, subkey = jax.random.split(key)
            sol.run_mcmc(subkey, n_steps=3000, uncertainty_multiplier=2.0)
            sol.samples_history[round_idx] = sol.samples.copy()
            sol.save_samples('posteriors', suffix=f'_{prefix_name}_Round_{round_idx}')

    # 5. TRUE SIMULATOR BASELINE
    true_lp = get_true_logprob(x_obs, y_obs, obs_error, THETA_BOUNDS, SIM_FUNC)
    key, subkey = jax.random.split(key)
    true_smp = np.array(run_mcmc_blackjax(true_lp, initial_mcmc_position, n_mcmc_samples, subkey)[n_mcmc_samples // 2:])
    save_npz(f"{prefix_name}_True_Simulator", true_smp)

    # 6. COLLECT METRICS & OUTPUT
    all_rounds = [1] + refinement_range
    round_metrics = collect_round_metrics(solutions, all_rounds, TRUE_THETA, THETA_BOUNDS,
                                          x_obs, y_obs, SIM_FUNC, _n_ppe_r1)
    timing_info = collect_timing(solutions)

    # Corner plots
    for sol in solutions:
        disc_smp = discovery[sol.model_type]["samples_R1"]
        plot_corner(sol, true_smp, disc_smp, TRUE_THETA, THETA_BOUNDS, param_labels,
                    refinement_range, prefix_name)

    # Emulator performance plots
    for round_idx in refinement_range:
        round_perf = {name: perf for (name, ridx), perf in emulator_perf.items() if ridx == round_idx}
        plot_emulator_performance(solutions, round_perf, round_idx, prefix_name)

    # Summary, CSVs
    print_single_summary(prefix_name, n_params, TRUE_THETA, THETA_BOUNDS, round_metrics, timing_info)
    write_csvs(prefix_name, n_params, round_metrics, timing_info)

    print("\nExperiment Complete.")

    return {
        "name": prefix_name,
        "n_params": n_params,
        "true_theta": TRUE_THETA,
        "theta_bounds": THETA_BOUNDS,
        "round_metrics": round_metrics,
        "timing": timing_info,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python main_experiment.py [parameterization_name]")
        print(f"  Only one parameterization at a time. Use run_all_parameterizations.py for multiple.")
        print(f"  Options: {list(PARAMETERIZATIONS.keys())}")
        sys.exit(1)
    name = sys.argv[1]
    if name not in PARAMETERIZATIONS:
        print(f"Unknown parameterization: {name}. Options: {list(PARAMETERIZATIONS.keys())}")
        sys.exit(1)
    run_experiment(param_name=name)
