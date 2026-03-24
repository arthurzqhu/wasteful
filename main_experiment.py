import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import os
import corner
from simulator import simulator, true_process
from emulators import train_gp_jax, predict_gp_jax, tune_and_train_nn_jax, predict_nn_jax
from mcmc_sampler import get_true_logprob, get_gp_logprob, get_nn_logprob, run_mcmc_blackjax
 
# Experiment configuration
n_samples_R1 = 1000
n_samples_R2 = 500
num_steps = 2500
UNCERTAINTY_MULTIPLIER = 3.0  # Inflate emulator uncertainty to prevent over-tightening

# TRUE_THETA_PL = jnp.array([1.6, 2.4])
# TRUE_THETA_HG = jnp.array([1.6, 2.4, 1.0, 5.0])
TRUE_THETA_HM = jnp.array([3, 2, 16])
# Bounds: [theta_0, theta_1, ..., x]
bounds_R1_HM = [(0.1, 5.0), (0.1, 5.0), (1, 20), (0.1, 150.0)] 
# bounds_R1_HG = [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0), (0.1, 7.0), (0.1, 150.0)] 
# bounds_R1_PL = [(0.1, 5.0), (0.1, 5.0), (0, 10.0)] 
# Derived automatically: midpoint of each parameter bound (excludes x, the last bound)
initial_mcmc_position = jnp.array([(lo + hi) / 2.0 for lo, hi in bounds_R1_HM[:-1]])

# -------------------------------------------------------------
# 0. DEFINE SIMULATOR AND TRUTH
def power_law(theta, x):
    """ y = theta_0 * x^theta_1 """
    return theta[:, 0] * jnp.power(x, theta[:, 1])

def harmonic_mean_3t(theta, x):
    # """ y = theta_0 * x^theta_1 """
    # """ y = theta_0 * (x^theta_1/(x^theta_2 + theta_3)) """
    """ vt = a * (v1**(-k) + v2**(-k))**(-1/k) """
    a = theta[:, 0]
    k = theta[:, 1]
    v1 = jnp.power(x, 2/3.)
    v2 = jnp.power(x, 1/3.) * jnp.power(theta[:, 2], 1/3.)
    v3 = jnp.power(x, 1/6.) * jnp.power(theta[:, 3], 1/2.)
    vt = a * (v1**(-k) + v2**(-k) + v3**(-k))**(-1/k)
    return vt
    # return theta[:, 0] * (jnp.power(x, theta[:, 1]) / (jnp.power(x, theta[:, 2]) + theta[:, 3]))

def harmonic_mean(theta, x):
    # """ y = theta_0 * x^theta_1 """
    # """ y = theta_0 * (x^theta_1/(x^theta_2 + theta_3)) """
    """ vt = a * (v1**(-k) + v2**(-k))**(-1/k) """
    a = theta[:, 0]
    k = theta[:, 1]
    v1 = jnp.power(x, 2/3.)
    v2 = jnp.power(x, 1/6.) * jnp.power(theta[:, 2], 1/2.)
    vt = a * (v1**(-k) + v2**(-k))**(-1/k)
    return vt
    # return theta[:, 0] * (jnp.power(x, theta[:, 1]) / (jnp.power(x, theta[:, 2]) + theta[:, 3]))

def hill_growth(theta, x):
    # """ y = theta_0 * x^theta_1 """
    """ y = theta_0 * (x^theta_1/(x^theta_2 + theta_3)) """
    return theta[:, 0] * (jnp.power(x, theta[:, 1]) / (jnp.power(x, theta[:, 2]) + theta[:, 3]))

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
    return fraction >= threshold

def generate_ppes(n_samples, bounds, key, sim_func):
    """Uniformly samples theta and x within given bounds, then runs the simulator."""
    samples = []
    for b in bounds:
        key, subkey = jax.random.split(key)
        samples.append(jax.random.uniform(subkey, shape=(n_samples, 1), minval=b[0], maxval=b[1]))
    X = jnp.column_stack(samples)
    y = simulator(X, sim_func)
    return X, y

# Task 20: Real-time Convergence Diagnostic
def print_distance_to_truth(samples, true_theta, context_name, x_obs, y_obs, sim_func):
    mean_smp = jnp.mean(samples, axis=0)
    theta_dist = jnp.sqrt(jnp.sum((mean_smp - true_theta)**2))
    
    # Calculate Simulator RMSE at the mean of the posterior
    y_pred = true_process(x_obs, mean_smp, sim_func)
    sim_rmse = jnp.sqrt(jnp.mean((y_pred - y_obs)**2))
    
    print(f"  [{context_name}] Dist: {theta_dist:.4f} | Sim RMSE: {sim_rmse:.4f} | Mean: {mean_smp}")
    return theta_dist, sim_rmse

def run_adaptive_discovery(name, model_type, key, bounds_R1, x_obs, y_obs, obs_error, true_theta, mcmc_prior_bounds, sim_func):
    """Performs the first adaptive discovery round for a given solution type."""
    start_t = time.time()
    print(f"\n--- Running Round 1 Discovery for {name} ({model_type.upper()}) ---")
    n_samples = n_samples_R1
    key, subkey = jax.random.split(key)
    X_R1, y_R1 = generate_ppes(n_samples, bounds_R1, subkey, sim_func)
    
    # Subsample for Discovery (Reduced to 10 to allow effective broadening)
    n_sub = 10
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, len(x_obs), (n_sub,), replace=False)
    x_sub = x_obs[indices]
    y_sub = y_obs[indices]
    err_sub = obs_error[indices] if not isinstance(obs_error, (int, float)) else obs_error
    
    noise_floor = 0.
    max_floor = 20.0 # Allow for much more aggressive broadening
    step = 1.0
    
    print(f"  [{name}] Training Round 1 Discovery emulator ({model_type.upper()})...")
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
        # Task 33: Use the single uncertainty_multiplier knob (broadened by 1 + broadening)
        # and scaled by sqrt(N) in the sampler.
        current_mult = UNCERTAINTY_MULTIPLIER * (1.0 + broadening)
        print(f"  [{name}] Testing Discovery MCMC with {current_mult:.1f} base multiplier (N_samples={n_samples})...")
        
        if model_type == 'gp':
            lp_fn = get_gp_logprob(predict_gp_jax, emulator_params[0], X_R1, y_R1, 
                                  emulator_params[1], emulator_params[2], emulator_params[3], emulator_params[4],
                                  x_sub, y_sub, err_sub, mcmc_prior_bounds, 
                                  uncertainty_multiplier=current_mult, n_samples=n_samples)
        elif model_type == 'nn':
            lp_fn = get_nn_logprob(predict_nn_jax, emulator_params, x_sub, y_sub, err_sub, 
                                   mcmc_prior_bounds, uncertainty_multiplier=current_mult, n_samples=n_samples)
            
        key, subkey = jax.random.split(key)
        init_pos = jnp.array([(lo + hi) / 2.0 for lo, hi in mcmc_prior_bounds])
        samples = run_mcmc_blackjax(lp_fn, init_pos, 3000, subkey)
        samples = np.array(samples[1000:])
        
        if is_posterior_wide_enough(samples, mcmc_prior_bounds, threshold=0.15):
            print(f"  [{name}] [SUCCESS] Round 1 achieves discovery width at multiplier {current_mult:.1f}.")
            print_distance_to_truth(samples, true_theta, f"{name} R1 Final", x_obs, y_obs, sim_func)
            save_npz(f"{name}_Round_1", samples)
            duration = time.time() - start_t
            return X_R1, y_R1, samples, duration
        else:
            print(f"  [{name}] [FAILURE] Posterior is too tight. Increasing broadening...")
            broadening += step
            
    duration = time.time() - start_t
    return X_R1, y_R1, samples, duration

class IterativeSolution:
    """Encapsulates a specific emulation strategy (e.g. GP Buffered) and its 3-round journey."""
    def __init__(self, name, strategy_type, model_type, X_R1, y_R1, samples_R1, x_obs, y_obs, obs_error, initial_bounds, sim_func, true_theta, buffer_margin=0.2):
        self.name = name
        self.strategy_type = strategy_type
        self.model_type = model_type
        self.initial_bounds = initial_bounds
        self.mcmc_prior_bounds = initial_bounds[:-1] # Exclude x
        self.x_range = (initial_bounds[-1][0], initial_bounds[-1][1])
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
            prior_lower = jnp.array([b[0] for b in self.initial_bounds[:-1]])
            prior_upper = jnp.array([b[1] for b in self.initial_bounds[:-1]])
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
            # Use most recent round + 10 global anchors
            X_last, y_last = self.history[-1]
            X_R1, y_R1 = self.history[0]
            key, subkey = jax.random.split(key)
            anc_idx = jax.random.choice(subkey, len(y_R1), shape=(min(10, len(y_R1)),), replace=False)
            X_all = jnp.vstack((X_last, X_R1[anc_idx]))
            y_all = jnp.concatenate((y_last, y_R1[anc_idx]))
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

        self.timings["Train"] += time.time() - start_t

    def get_logprob_fn(self, uncertainty_multiplier=UNCERTAINTY_MULTIPLIER):
        n_samples = len(self.y_train) if self.y_train is not None else 1
        if self.model_type == 'gp':
            return get_gp_logprob(predict_gp_jax, self.params[0], self.X_train, self.y_train, 
                                 self.params[1], self.params[2], self.params[3], self.params[4],
                                 self.x_obs, self.y_obs, self.obs_error, self.mcmc_prior_bounds, 
                                 uncertainty_multiplier=uncertainty_multiplier, n_samples=n_samples)
        elif self.model_type == 'nn':
            return get_nn_logprob(predict_nn_jax, self.params, self.x_obs, self.y_obs, 
                                 self.obs_error, self.mcmc_prior_bounds, 
                                 uncertainty_multiplier=uncertainty_multiplier, n_samples=n_samples)

    def predict(self, X_new):
        """Unified prediction interface that handles model-specific signatures."""
        if self.model_type == 'gp':
            return predict_gp_jax(X_new, self.params[0], self.X_train, self.y_train, *self.params[1:5])
        elif self.model_type == 'nn':
            return predict_nn_jax(X_new, *self.params)

    def run_mcmc(self, key, initial_pos=None, n_steps=3000, uncertainty_multiplier=1.0):
        start_t = time.time()
        if initial_pos is None:
            # Task 7: Sequential initialization based on previous posterior mean
            initial_pos = jnp.mean(self.samples, axis=0)
            
        lp_fn = self.get_logprob_fn(uncertainty_multiplier=uncertainty_multiplier)
        self.samples = np.array(run_mcmc_blackjax(lp_fn, initial_pos, n_steps, key)[n_steps//2:])
        print_distance_to_truth(self.samples, self.true_theta, f"{self.name} Current", self.x_obs, self.y_obs, self.sim_func)
        self.timings["MCMC"] += time.time() - start_t
        return self.samples

    def add_round(self, n_new, key, round_name):
        start_t = time.time()
        print(f"  [{self.name}] Adding {round_name} samples...")
        key, subkey = jax.random.split(key)
        
        # Task 30: Posterior-Weighted Sampling for all refining rounds
        # We sample directly from the most recent posterior mass to ensure density
        # where the likelihood is highest, preventing emulator "drift".
        n_available = len(self.samples)
        idx = jax.random.choice(subkey, n_available, shape=(n_new,))
        theta_new = self.samples[idx]

        # Diagnostic: check how concentrated the new PPE samples are
        theta_std = jnp.std(theta_new, axis=0)
        theta_range = jnp.ptp(theta_new, axis=0)
        print(f"  [{self.name}] DIAG PPE: new theta std={[f'{v:.4f}' for v in theta_std]}, range={[f'{v:.4f}' for v in theta_range]}")

        x_new = jnp.geomspace(self.x_range[0], self.x_range[1], n_new).reshape(-1, 1)
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

def run_experiment(sim_func=None, true_theta=None, bounds=None):
    key = jax.random.PRNGKey(42)

    # Use arguments if provided, otherwise fall back to module-level defaults (harmonic_mean)
    SIM_FUNC = sim_func if sim_func is not None else harmonic_mean
    TRUE_THETA = true_theta if true_theta is not None else TRUE_THETA_HM
    BOUNDS = bounds if bounds is not None else bounds_R1_HM

    mcmc_prior_bounds = BOUNDS[:-1]
    param_labels = [f"Parameter $\\theta_{i}$" for i in range(len(TRUE_THETA))]

    # -------------------------------------------------------------
    # 0. RUN CONFIGURATION
    # -------------------------------------------------------------
    MODELS_TO_RUN = ["nn"]  # Options: "gp", "nn"
    STRATEGIES_TO_RUN = ["global", "buffered", "local"] # Options: "global", "buffered", "local"

    print(f"\n--- PPE RUN CONFIGURATION ---")
    print(f"Models: {MODELS_TO_RUN}")
    print(f"Strategies: {STRATEGIES_TO_RUN}")

    # 1. GENERATE OBSERVATION DATA (Log-distributed)
    x_obs = jnp.geomspace(BOUNDS[-1][0], BOUNDS[-1][1], 50)
    key, subkey = jax.random.split(key)
    # Use 5% relative observation error for stability across 8 orders of magnitude
    clean_y = true_process(x_obs, TRUE_THETA, SIM_FUNC)
    obs_error = 0.05 * clean_y
    y_obs = clean_y + obs_error * jax.random.normal(subkey, x_obs.shape)

    # -------------------------------------------------------------
    # 2. RUN ADAPTIVE DISCOVERY (Round 1)
    # -------------------------------------------------------------
    X_R1_GP, y_R1_GP, samples_R1_GP, dur_GP = None, None, None, 0.0
    if "gp" in MODELS_TO_RUN:
        # GPs share the same Round 1 Discovery
        key, subkey = jax.random.split(key)
        X_R1_GP, y_R1_GP, samples_R1_GP, dur_GP = run_adaptive_discovery("GP-Shared", "gp", subkey, BOUNDS,
                                                                  x_obs, y_obs, obs_error, TRUE_THETA,
                                                                  mcmc_prior_bounds, SIM_FUNC)

    X_R1_NN, y_R1_NN, samples_R1_NN, dur_NN = None, None, None, 0.0
    if "nn" in MODELS_TO_RUN:
        # NN runs independent Round 1 Discovery
        key, subkey = jax.random.split(key)
        X_R1_NN, y_R1_NN, samples_R1_NN, dur_NN = run_adaptive_discovery("NN-Isolated", "nn", subkey, BOUNDS,
                                                                  x_obs, y_obs, obs_error, TRUE_THETA,
                                                                  mcmc_prior_bounds, SIM_FUNC)

    # -------------------------------------------------------------
    # 3. INITIALIZE INDEPENDENT SOLUTION PATHS
    # -------------------------------------------------------------
    solutions = []

    # Define potential strategy combinations
    combos = []
    if "gp" in MODELS_TO_RUN:
        for s in STRATEGIES_TO_RUN:
            combos.append(("GP", s, "gp", X_R1_GP, y_R1_GP, samples_R1_GP))
    if "nn" in MODELS_TO_RUN:
        for s in STRATEGIES_TO_RUN:
            combos.append(("NN", s, "nn", X_R1_NN, y_R1_NN, samples_R1_NN))

    for prefix, strat, mtype, X1, y1, s1 in combos:
        solutions.append(IterativeSolution(f"{prefix} {strat.capitalize()}", strat, mtype, X1, y1, s1,
                          x_obs, y_obs, obs_error, BOUNDS, SIM_FUNC, TRUE_THETA))
    
    # Attribute Discovery Time
    for sol in solutions:
        sol.timings["Discovery"] = dur_GP if sol.model_type == 'gp' else dur_NN

    # -------------------------------------------------------------
    # 4. ITERATIVE PPE - MULTI-MODEL REFINEMENT
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # 4. ITERATIVE PPE - MULTI-MODEL REFINEMENT (Rounds 2-5)
    # -------------------------------------------------------------
    for round_idx in [2, 3, 4, 5]:
        print(f"\n" + "="*50)
        print(f"--- RUNNING ROUND {round_idx} (Iterative Refinement) ---")
        print("="*50)
        for sol in solutions:
            print(f"\n--- Strategy: {sol.name} (Round {round_idx}) ---")
            # 1. Update priors based on previous round results
            sol.update_priors()
            
            # 2. Add round data (Guided by previous posterior/shrunk bounds)
            key, subkey = jax.random.split(key)
            sol.add_round(n_samples_R2, subkey, f"Round {round_idx}")
            
            # 3. Train on accumulated history
            key, subkey = jax.random.split(key)
            sol.train(subkey)
            # 4. Refine Posterior (Guide for next round)
            # Task 14: Use tighter uncertainty (2.0) and more steps (3000) for refinement
            print(f"  -> Sampling from internal {sol.model_type.upper()} posterior (Scale: 2.0, Steps: 3000)...")
            key, subkey = jax.random.split(key)
            sol.run_mcmc(subkey, n_steps=3000, uncertainty_multiplier=2.0)
            sol.samples_history[round_idx] = sol.samples.copy()
            
            # Save intermediate samples
            sol.save_samples('posteriors', suffix=f'_Round_{round_idx}')

    # 5. FINAL METRIC AUDIT
    # -------------------------------------------------------------
    print("\n--- Final Metric Audit ---")
    def get_metrics(yt, yp):
        rmse = jnp.sqrt(jnp.mean((yp - yt)**2))
        std_t = jnp.std(yt)
        nrmse = rmse / std_t if std_t > 0 else 0.0
        r2 = 1 - jnp.sum((yt - yp)**2) / jnp.sum((yt - jnp.mean(yt))**2)
        return float(rmse), float(nrmse), float(r2)

    # Task 21: Exhaustive Likelihood Audit
    print("\n--- Comprehensive Likelihood Audit (Truth Recovery) ---")
    results_table = []
    if samples_R1_GP is not None:
        lp = get_true_logprob(x_obs, y_obs, obs_error, mcmc_prior_bounds, SIM_FUNC)(jnp.mean(samples_R1_GP, axis=0))
        results_table.append(["GP Discovery R1", float(lp)])
    if samples_R1_NN is not None:
        lp = get_true_logprob(x_obs, y_obs, obs_error, mcmc_prior_bounds, SIM_FUNC)(jnp.mean(samples_R1_NN, axis=0))
        results_table.append(["NN Discovery R1", float(lp)])

    for sol in solutions:
        for ridx in sorted(sol.samples_history.keys()):
            smp = sol.samples_history[ridx]
            lp = get_true_logprob(x_obs, y_obs, obs_error, mcmc_prior_bounds, SIM_FUNC)(jnp.mean(smp, axis=0))
            results_table.append([f"{sol.name} R{ridx}", float(lp)])

    print(f"{'Source':30} | {'LogProb(Truth)':>15}")
    print("-" * 50)
    for name, lp in results_table:
        print(f"{name:30} | {lp:15.3f}")

    print("\n--- Final Emulation Metric Audit ---")
    for sol in solutions:
        X_t, y_t = sol.X_val, sol.y_val
        yp, _ = sol.predict(X_t)
        rmse, nrmse, r2 = get_metrics(y_t, yp)
        print(f"{sol.name:12} | RMSE: {rmse:10.3f} | MAS: {r2:7.4f} | Train Data: {len(sol.y_train)}")

    # 6. FINAL POSTERIOR PREPARATION
    print("\n--- Formatting Posteriors for Plotting ---")
    
    # 1. True Simulator Baseline
    true_lp = get_true_logprob(x_obs, y_obs, obs_error, mcmc_prior_bounds, SIM_FUNC)
    print("Sampling True Simulator Baseline...")
    key, subkey = jax.random.split(key)
    init_pos = jnp.array([(lo + hi) / 2.0 for lo, hi in mcmc_prior_bounds])
    true_smp = np.array(run_mcmc_blackjax(true_lp, init_pos, num_steps, subkey)[num_steps//2:])
    save_npz("True_Simulator", true_smp)

    # -------------------------------------------------------------
    # 7. PLOTTING (Task 22: De-cluttered Strategy-Specific Plots)
    # -------------------------------------------------------------
    os.makedirs('plots', exist_ok=True)
    style_config = {
        "True Simulator": {"color": "k"},
        "GP Discovery (Round 1)": {"color": "tab:gray", "ls": "--", "alpha": 0.5},
        "NN Discovery (Round 1)": {"color": "tab:gray", "ls": "--", "alpha": 0.5},
        "GP Global": {"color": "tab:blue"},
        "GP Buffered": {"color": "tab:pink"},
        "GP Local": {"color": "tab:orange"},
        "NN Global": {"color": "tab:cyan"},
        "NN Buffered": {"color": "tab:green"},
        "NN Local": {"color": "tab:purple"},
    }

    def plot_strategy(sol_name, sol_samples_dict, filename):
        """Task 23 & 24: Individual corner plots with colormap-based round progression."""
        fig = None
        
        # 1. Determine total rounds for color mapping
        # Rounds: Discovery (R1) + Refinement Rounds (2, 3, ...)
        discovery_smp = samples_R1_GP if sol_name.startswith("GP") else samples_R1_NN
        rounds_keys = sorted(sol_samples_dict.keys())
        total_rounds = len(rounds_keys) + (1 if discovery_smp is not None else 0)
        
        # Create colormap sequence
        cmap = plt.get_cmap('viridis')
        def get_round_color(idx):
            if total_rounds <= 1: return cmap(0.0) # Purple
            # Divide colormap evenly
            return cmap(idx / (total_rounds - 1))

        # 2. Base: True Simulator
        # Truth transparency set to 0.5 per Task 26
        truth_color = mcolors.to_rgba('tab:red', alpha=0.5)
        
        fig = corner.corner(true_smp, color='k', labels=param_labels, 
                            show_titles=True, title_fmt=".2f",
                            plot_datapoints=False, plot_density=False, 
                            no_fill_contours=True, smooth=1.0,
                            truths=TRUE_THETA, truth_color=truth_color)
        
        curr_idx = 0
        true_mean = jnp.mean(true_smp, axis=0)
        true_dist = jnp.sqrt(jnp.sum((true_mean - TRUE_THETA)**2))
        handles = [mlines.Line2D([], [], color='k', label=f'True Simulator (D={true_dist:.3f})'),
                   mlines.Line2D([], [], color=truth_color, label='Truth Value')]

        # 3. Add Discovery R1 context
        if discovery_smp is not None:
            color = get_round_color(curr_idx)
            corner.corner(discovery_smp, fig=fig, color=color, 
                          hist_kwargs={"ls": "--"}, contour_kwargs={"linestyles": "--"},
                          plot_datapoints=False, plot_density=False, 
                          no_fill_contours=True, smooth=1.0)
            
            d_mean = jnp.mean(discovery_smp, axis=0)
            d_dist = jnp.sqrt(jnp.sum((d_mean - TRUE_THETA)**2))
            handles.append(mlines.Line2D([], [], color=color, ls='--', label=f'Discovery (R1) (D={d_dist:.3f})'))
            curr_idx += 1
            
        # 4. Add refinement rounds
        for ridx in rounds_keys:
            # Task 26: Skip Round 1 if Discovery (R1) is already plotted
            if ridx == 1 and discovery_smp is not None:
                continue
                
            color = get_round_color(curr_idx)
            smp = sol_samples_dict[ridx]
            corner.corner(smp, fig=fig, color=color,
                          plot_datapoints=False, plot_density=False,
                          no_fill_contours=True, smooth=1.0)
            
            r_mean = jnp.mean(smp, axis=0)
            r_dist = jnp.sqrt(jnp.sum((r_mean - TRUE_THETA)**2))
            handles.append(mlines.Line2D([], [], color=color, label=f'Round {ridx} (D={r_dist:.3f})'))
            curr_idx += 1

        fig.legend(handles=handles, loc=(0.65, 0.75), fontsize=10)
        plt.suptitle(f"4D PPE Trajectory: {sol_name}", fontsize=16, y=1.02)
        fig.savefig(f'plots/{filename}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved plots/{filename}.pdf")

    # Final Samples Dictionary for contextual plotting
    samples_dict = {"True Simulator": true_smp}
    if samples_R1_GP is not None: samples_dict["GP Discovery (Round 1)"] = samples_R1_GP
    if samples_R1_NN is not None: samples_dict["NN Discovery (Round 1)"] = samples_R1_NN

    for sol in solutions:
        # Pass only the round index as the key for cleaner plotting logic
        sol_rounds = {ridx: smp for ridx, smp in sol.samples_history.items()}
        plot_strategy(sol.name, sol_rounds, f"mcmc_posteriors_{sol.name.replace(' ', '_')}")

    print("\n--- Quantitative Posterior Analysis ---")
    header = f"{'Model':<20} | " + " | ".join([f"Th_{i}" for i in range(len(TRUE_THETA))]) + " | DistToTh | Sim RMSE"
    print(header)
    print("-" * len(header))
    
    # Quantitative table
    for name, smp in samples_dict.items():
        means = np.mean(smp, axis=0)
        stds = np.std(smp, axis=0)
        dist = jnp.sqrt(jnp.sum((means - TRUE_THETA)**2))
        y_pred = true_process(x_obs, means, SIM_FUNC)
        sim_rmse = jnp.sqrt(jnp.mean((y_pred - y_obs)**2))
        row = f"{name[:20]:<20} | " + " | ".join([f"{m:4.2f}±{s:4.2f}" for m, s in zip(means, stds)]) + f" | {dist:8.4f} | {sim_rmse:8.4f}"
        print(row)
    for sol in solutions:
        means = np.mean(sol.samples, axis=0)
        stds = np.std(sol.samples, axis=0)
        dist = jnp.sqrt(jnp.sum((means - TRUE_THETA)**2))
        y_pred = true_process(x_obs, means, SIM_FUNC)
        sim_rmse = jnp.sqrt(jnp.mean((y_pred - y_obs)**2))
        row = f"{sol.name[:20]:<20} | " + " | ".join([f"{m:4.2f}±{s:4.2f}" for m, s in zip(means, stds)]) + f" | {dist:8.4f} | {sim_rmse:8.4f}"
        print(row)
    print(f"{'Truth':<20} | " + " | ".join([f"{t:4.2f}" for t in TRUE_THETA]))

    # 9. Emulator Performance Comparison Plot (Task 25)
    print("\n--- Generating Emulator Performance Comparison ---")
    fig_rmse, axes_rmse = plt.subplots(2, 3, figsize=(18, 12))
    axes_rmse = axes_rmse.flatten()
    
    for i, sol in enumerate(solutions): # Show all 6 strategies in a 2x3 grid
        ax = axes_rmse[i]
        X_t, y_t = sol.X_val, sol.y_val
        yp, _ = sol.predict(X_t)
        rmse, nrmse, r2 = get_metrics(y_t, yp)
        
        color = style_config.get(sol.name, {"color":"tab:blue"})["color"]
        ax.scatter(y_t[::2], yp[::2], alpha=0.5, s=15, color=color) # sub-sample for speed
        
        all_vals = jnp.concatenate([y_t, yp])
        min_v, max_v = float(jnp.nanmin(all_vals)), float(jnp.nanmax(all_vals))
        min_v = max(min_v, 1e-1)
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label="1:1 Line")
        
        ax.set_title(f"{sol.name}\nRMSE: {rmse:.3f}, MAS: {r2:.4f}", fontsize=12)
        ax.set_xlabel("True Simulator $y$")
        ax.set_ylabel("Emulator Predicted $y$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        if i == 0: ax.legend()

    plt.suptitle("Emulator Performance Comparison: Each Model Evaluated on its Walk", fontsize=18, y=0.98)
    fig_rmse.tight_layout(rect=[0, 0, 1, 0.96])
    fig_rmse.savefig('plots/emulator_performance_comparison.pdf')
    print("Saved parity plots to plots/emulator_performance_comparison.pdf")

    # 10. TIMING SUMMARY (Task 31: Corrected shared discovery attribution)
    print("\n" + "="*80)
    print(f"SHARED DISCOVERY TIME:")
    if samples_R1_GP is not None: print(f"  GP Discovery: {dur_GP:10.2f} s")
    if samples_R1_NN is not None: print(f"  NN Discovery: {dur_NN:10.2f} s")
    print("-" * 80)
    print(f"{'STRATEGY':<15} | {'TRAIN (s)':<10} | {'MCMC (s)':<10} | {'REFINE (s)':<10} | {'TOTAL (s)':<10}")
    print("-" * 80)
    for sol in solutions:
        t = sol.timings
        # Total excludes Discovery as per Task 31 (shared, not strategy-specific)
        total = t['Train'] + t['MCMC'] + t['Refinement']
        print(f"{sol.name:<15} | {t['Train']:10.2f} | {t['MCMC']:10.2f} | {t['Refinement']:10.2f} | {total:10.2f}")
    print("="*80)

    print("\nExperiment Complete.")

if __name__ == "__main__":
    run_experiment()
