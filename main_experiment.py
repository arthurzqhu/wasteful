import os
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b5356651-0d8e-5cd1-bdf3-ccbb8b221031"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import corner
import numpy as np
from simulator import simulator, true_process, generate_ppes
from emulators import train_gp_jax, predict_gp_jax, train_nn_jax, predict_nn_jax
from mcmc_sampler import get_true_logprob, get_gp_logprob, get_nn_logprob, run_mcmc_blackjax

def run_experiment():
    key = jax.random.PRNGKey(42)
    
    # ---------------- N-DIMENSIONAL CONFIGURATION ----------------
    # Define arbitrary internal structure for y
    def power_law_func(theta, x):
        """ y = theta_0 * x^theta_1 """
        return theta[:, 0] * jnp.power(x, theta[:, 1])
        
    SIM_FUNC = power_law_func
    TRUE_THETA = jnp.array([1.6, 2.4])
    
    # Bounds for the parameters: [theta_0_bounds, theta_1_bounds ..., x_bounds]
    bounds_R1 = [(0.1, 5.0), (0.1, 5.0), (1.0, 5.0)] 
    
    # Define a buffer padding size for each parameter EXCEPT the final physical state 'x'
    buffer_deltas = [0.5, 0.5] 
    
    mcmc_prior_bounds = [(0.1, 5.0), (0.1, 5.0)]
    initial_mcmc_position = jnp.array([2.0, 2.0])
    
    param_labels = [f"Parameter $\\theta_{i}$" for i in range(len(TRUE_THETA))]
    # -------------------------------------------------------------
    
    # 1. GENERATE OBSERVATION DATA
    x_obs = jnp.linspace(1.0, 5.0, 10)
    obs_error = 2.0
    key, subkey = jax.random.split(key)
    y_obs = true_process(x_obs, TRUE_THETA, SIM_FUNC) + obs_error * jax.random.normal(subkey, x_obs.shape)
    
    n_samples_R1 = 200
    n_samples_R2 = 200

    print("--- Running Round 1 ---")
    key, subkey = jax.random.split(key)
    X_R1, y_R1 = generate_ppes(n_samples_R1, bounds_R1, subkey, SIM_FUNC)
    
    # Train GP for robust NROY bounds
    gp_params_R1, r1_x_mu, r1_x_sd, r1_y_mu, r1_y_sd = train_gp_jax(X_R1, y_R1)
    
    # Grid search for NROY space in N-1 dimensions (everything except x)
    grid_res = 30
    grids = [jnp.linspace(b[0], b[1], grid_res) for b in bounds_R1[:-1]]
    meshgrids = jnp.meshgrid(*grids, indexing='ij')
    flat_grids = [m.ravel() for m in meshgrids]
    
    def calc_implausibility(*theta_vals):
        """Find max implausibility across all x_obs for a given theta vector."""
        theta_mat = jnp.column_stack([jnp.full_like(x_obs, val) for val in theta_vals])
        X_test = jnp.column_stack((theta_mat, x_obs))
        y_pred, y_std = predict_gp_jax(X_test, gp_params_R1, X_R1, y_R1, r1_x_mu, r1_x_sd, r1_y_mu, r1_y_sd)
        impl = jnp.abs(y_pred - y_obs) / jnp.sqrt(obs_error**2 + y_std**2)
        return jnp.max(impl)
    
    # Dynamically build vmap for N parameters
    # Since we flattened the meshgrids into 1D arrays of size (grid_res^N,), we only need one vmap!
    vec_impl = jax.vmap(calc_implausibility)
        
    print(f"Calculating Implausibility grid for {len(flat_grids)} tuning parameters...")
    impl_scores = vec_impl(*flat_grids)
    nroy_mask = impl_scores < 3.0
    
    valid_points = [grid[nroy_mask] for grid in flat_grids]
    
    if len(valid_points[0]) == 0:
        print("Warning: Entire space ruled out, falling back.")
        bounds_R2 = bounds_R1
    else:
        bounds_R2 = []
        for i, v in enumerate(valid_points):
            min_v = min(float(jnp.min(v)), float(TRUE_THETA[i]))
            max_v = max(float(jnp.max(v)), float(TRUE_THETA[i]))
            bounds_R2.append((min_v, max_v))
        # Append the original x bounds
        bounds_R2.append(bounds_R1[-1])
    
    print("Round 2 Bounds shrunk:")
    for i, (b1, b2) in enumerate(zip(bounds_R1[:-1], bounds_R2[:-1])):
        print(f" theta_{i}: {b1} -> {b2}")

    print("\n--- Running Round 2 ---")
    key, subkey = jax.random.split(key)
    X_R2, y_R2 = generate_ppes(n_samples_R2, bounds_R2, subkey, SIM_FUNC)

    X_Global = jnp.vstack((X_R1, X_R2))
    y_Global = jnp.concatenate((y_R1, y_R2))
    
    X_Local = X_R2
    y_Local = y_R2
    
    # Calculate buffer dynamically for all tuning parameters
    in_buffer = jnp.ones(len(X_R1), dtype=bool)
    for i in range(len(bounds_R2) - 1):
        in_buffer = in_buffer & (X_R1[:, i] >= bounds_R2[i][0] - buffer_deltas[i]) & \
                                (X_R1[:, i] <= bounds_R2[i][1] + buffer_deltas[i])
    
    X_Buffer = jnp.vstack((X_R1[in_buffer], X_R2))
    y_Buffer = jnp.concatenate((y_R1[in_buffer], y_R2))

    print(f"Data Sizes -> Global: {len(y_Global)}, Local: {len(y_Local)}, Buffered: {len(y_Buffer)}")

    print("\n--- Training Gaussian Processes ---")
    gp_G_params, g_x_mu, g_x_sd, g_y_mu, g_y_sd = train_gp_jax(X_Global, y_Global)
    gp_L_params, l_x_mu, l_x_sd, l_y_mu, l_y_sd = train_gp_jax(X_Local, y_Local)
    gp_B_params, b_x_mu, b_x_sd, b_y_mu, b_y_sd = train_gp_jax(X_Buffer, y_Buffer)
    
    print("--- Training Neural Networks ---")
    key, *subkeys = jax.random.split(key, 4)
    # The emulators require NN input_dim to be set to len(bounds_R1)
    # Let's dynamically create our train states
    
    nn_G_state_tuple = train_nn_jax(X_Global, y_Global, subkeys[0], epochs=3000, lr=1e-2)
    nn_L_state_tuple = train_nn_jax(X_Local, y_Local, subkeys[1], epochs=3000, lr=1e-2)
    nn_B_state_tuple = train_nn_jax(X_Buffer, y_Buffer, subkeys[2], epochs=3000, lr=1e-2)

    print("\n--- Emulation RMSE Check ---")
    # Test slightly outside R2 bounds for ALL parameters
    test_bounds = []
    for i in range(len(bounds_R2)):
        test_bounds.append((bounds_R2[i][0]-0.5, bounds_R2[i][1]+0.5))
        
    key, subkey = jax.random.split(key)
    X_test, y_true_test = generate_ppes(500, test_bounds, subkey, SIM_FUNC)
    
    def get_rmse(y_pred, name):
        rmse = jnp.sqrt(jnp.mean((y_pred - y_true_test)**2))
        print(f"RMSE {name:15}: {rmse:.3f}")
        return float(rmse)

    yp_gG, _ = predict_gp_jax(X_test, gp_G_params, X_Global, y_Global, g_x_mu, g_x_sd, g_y_mu, g_y_sd)
    yp_gB, _ = predict_gp_jax(X_test, gp_B_params, X_Buffer, y_Buffer, b_x_mu, b_x_sd, b_y_mu, b_y_sd)
    yp_gL, _ = predict_gp_jax(X_test, gp_L_params, X_Local, y_Local, l_x_mu, l_x_sd, l_y_mu, l_y_sd)
    
    yp_nG, _ = predict_nn_jax(X_test, *nn_G_state_tuple)
    yp_nB, _ = predict_nn_jax(X_test, *nn_B_state_tuple)
    yp_nL, _ = predict_nn_jax(X_test, *nn_L_state_tuple)
    
    yp_gG_rmse = get_rmse(yp_gG, "GP Global")
    yp_gB_rmse = get_rmse(yp_gB, "GP Buffered")
    yp_gL_rmse = get_rmse(yp_gL, "GP Local")
    yp_nG_rmse = get_rmse(yp_nG, "NN Global")
    yp_nB_rmse = get_rmse(yp_nB, "NN Buffered")
    yp_nL_rmse = get_rmse(yp_nL, "NN Local")

    print("\n--- Running MCMC Inference ---")
    num_steps = 15000
    
    logprobs = {
        "True Simulator": get_true_logprob(x_obs, y_obs, obs_error, mcmc_prior_bounds, SIM_FUNC),
        "GP Global": get_gp_logprob(predict_gp_jax, gp_G_params, X_Global, y_Global, g_x_mu, g_x_sd, g_y_mu, g_y_sd, x_obs, y_obs, obs_error, mcmc_prior_bounds),
        "GP Buffered": get_gp_logprob(predict_gp_jax, gp_B_params, X_Buffer, y_Buffer, b_x_mu, b_x_sd, b_y_mu, b_y_sd, x_obs, y_obs, obs_error, mcmc_prior_bounds),
        "GP Local": get_gp_logprob(predict_gp_jax, gp_L_params, X_Local, y_Local, l_x_mu, l_x_sd, l_y_mu, l_y_sd, x_obs, y_obs, obs_error, mcmc_prior_bounds),
        "NN Global": get_nn_logprob(predict_nn_jax, nn_G_state_tuple, x_obs, y_obs, obs_error, mcmc_prior_bounds),
        "NN Buffered": get_nn_logprob(predict_nn_jax, nn_B_state_tuple, x_obs, y_obs, obs_error, mcmc_prior_bounds),
        "NN Local": get_nn_logprob(predict_nn_jax, nn_L_state_tuple, x_obs, y_obs, obs_error, mcmc_prior_bounds),
    }

    samples_dict = {}
    for name, lp_fn in logprobs.items():
        key, subkey = jax.random.split(key)
        print(f"Sampling {name}...")
        samples = run_mcmc_blackjax(lp_fn, initial_mcmc_position, num_steps, subkey)
        # Convert to numpy and throw away 3000 burn-in steps
        samples_dict[name] = np.array(samples[3000:])
        
    print("\n--- Generating Plots ---")
    os.makedirs('plots', exist_ok=True)
    import matplotlib.lines as mlines
    import matplotlib.colors as mcolors
    
    # 1. Plot Emulation RMSE Comparison (Scatter Plots)
    rmse_names = ["GP Global", "GP Buffered", "GP Local", "NN Global", "NN Buffered", "NN Local"]
    rmses = [yp_gG_rmse, yp_gB_rmse, yp_gL_rmse, yp_nG_rmse, yp_nB_rmse, yp_nL_rmse]
    yp_preds = [yp_gG, yp_gB, yp_gL, yp_nG, yp_nB, yp_nL]
    
    fig_rmse, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    min_val = min(jnp.min(y_true_test), min([jnp.min(yp) for yp in yp_preds]))
    max_val = max(jnp.max(y_true_test), max([jnp.max(yp) for yp in yp_preds]))
    
    for i, (name, yp, rmse) in enumerate(zip(rmse_names, yp_preds, rmses)):
        ax = axes[i]
        color = 'tab:blue' if 'GP' in name else 'tab:orange'
        alpha = 0.5 if 'Global' in name else (0.8 if 'Buffered' in name else 1.0)
        
        ax.scatter(y_true_test, yp, alpha=alpha, color=color, s=15)
        
        # 1-to-1 diagonal line
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="1-to-1")
        
        ax.set_title(f"{name}\n(RMSE: {rmse:.3f})", fontsize=12)
        ax.set_xlabel("True Simulator Output")
        ax.set_ylabel("Emulator Prediction")
        if i == 0:
            ax.legend(loc="upper left")
            
    fig_rmse.suptitle("Emulation Output vs PPE (Out-of-Bounds)", fontsize=16)
    fig_rmse.tight_layout()
    fig_rmse.savefig('plots/emulator_rmse_comparison.pdf')
    print("Saved RMSE comparison to plots/emulator_rmse_comparison.pdf")
    
    # 2. MCMC Corner Plots
    fig = plt.figure(figsize=(10, 10))
    style_config = {
        "True Simulator": {"color": "k"},
        "GP Global": {"color": "tab:blue", "ls": "dashed"},
        "GP Buffered": {"color": "tab:pink", "ls": "dashed"},
        "GP Local": {"color": "tab:orange", "ls": "dashed"},
        "NN Global": {"color": "tab:blue", "ls": "solid"},
        "NN Buffered": {"color": "tab:pink", "ls": "solid"},
        "NN Local": {"color": "tab:orange", "ls": "solid"}
    }

    for name, smp in samples_dict.items():
        is_base = (name == "True Simulator")
        alpha_val = 1.0 if is_base else 0.6
        conf = style_config[name]
        
        corner.corner(smp, fig=fig, color=conf["color"], 
                      hist_kwargs={"ls": conf.get("ls", "solid")},
                      contour_kwargs={"linestyles": conf.get("ls", "solid")},
                      labels=param_labels,
                      show_titles=is_base, title_fmt=".2f",
                      plot_datapoints=False, plot_density=False,
                      smooth=1.0, alpha=alpha_val,
                      truths=TRUE_THETA if is_base else None,
                      truth_color=mcolors.to_rgba("tab:red", alpha=0.5))
    
    handles = [mlines.Line2D([], [], color=conf["color"], linestyle=conf.get("ls", "solid"), label=name) 
               for name, conf in style_config.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=12)

    tgt_str = ", ".join([f"$\\theta_{i}$={t}" for i, t in enumerate(TRUE_THETA)])
    plt.suptitle(f"MCMC Posterior Corner Plot (Target: {tgt_str})", fontsize=18, y=1.02)
    plt.savefig('plots/mcmc_posteriors.pdf', bbox_inches="tight")
    print("Visual comparison saved to plots/mcmc_posteriors.pdf")

if __name__ == "__main__":
    run_experiment()
