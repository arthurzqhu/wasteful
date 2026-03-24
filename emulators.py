import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from tinygp import kernels, GaussianProcess
import jaxopt
from typing import Sequence
import jax.scipy.stats as jstats

# --- CRPS (Continuously Ranked Probability Score) ---

def gaussian_crps(mu, sigma, y):
    """
    Analytical CRPS for a Gaussian distribution:
    Ref: Gneiting et al. (2005)
    """
    # z = (y - mu) / sigma
    # CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    z = (y - mu) / jnp.maximum(sigma, 1e-6)
    phi = jstats.norm.pdf(z)
    Phi = jstats.norm.cdf(z)
    
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / jnp.sqrt(jnp.pi))
    return jnp.mean(crps)

# --- GP Emulator (tinygp) ---

def train_gp_jax(X, y, diag=1e-3, n_restarts=5, key=None):
    """
    Train a JAX-based Gaussian Process using jaxopt with standard scaling and ARD.
    Uses multiple random restarts and hyperparameter clamping to avoid degenerate solutions.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Standardize X and y for better optimization
    X_mean = jnp.mean(X, axis=0)
    X_std = jnp.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    y_mean = jnp.mean(jnp.log1p(y))
    y_std = jnp.std(jnp.log1p(y)) + 1e-8
    y_norm = (jnp.log1p(y) - y_mean) / y_std

    # Hyperparameter bounds to prevent degenerate solutions
    LOG_LENGTH_BOUNDS = (-5.0, 10.0)
    LOG_SCALE_BOUNDS = (-5.0, 15.0)
    LOG_NOISE_BOUNDS = (-8.0, 5.0)  # Noise floor ~exp(-8)≈3e-4 prevents interpolation

    def clamp(params):
        return {
            "log_scale": jnp.clip(params["log_scale"], *LOG_SCALE_BOUNDS),
            "log_length": jnp.clip(params["log_length"], *LOG_LENGTH_BOUNDS),
            "log_noise": jnp.clip(params["log_noise"], *LOG_NOISE_BOUNDS),
        }

    def loss(params):
        params = clamp(params)
        X_scaled = X_norm / jnp.exp(params["log_length"])
        kernel = jnp.exp(params["log_scale"]) * kernels.ExpSquared()
        gp = GaussianProcess(kernel, X_scaled, diag=diag + jnp.exp(params["log_noise"]))
        return -gp.log_probability(y_norm)

    n_dims = X.shape[1]
    solver = jaxopt.ScipyMinimize(fun=loss)

    # Default initialization
    init_params_list = [{
        "log_scale": jnp.log(1.0),
        "log_length": jnp.zeros(n_dims),
        "log_noise": jnp.log(diag),
    }]

    # Random restarts with varied initializations
    for i in range(n_restarts - 1):
        key, k1, k2, k3 = jax.random.split(key, 4)
        init_params_list.append({
            "log_scale": jax.random.uniform(k1, minval=-2.0, maxval=5.0),
            "log_length": jax.random.uniform(k2, shape=(n_dims,), minval=-2.0, maxval=3.0),
            "log_noise": jax.random.uniform(k3, minval=-6.0, maxval=0.0),
        })

    best_loss = jnp.inf
    best_params = None
    for init_params in init_params_list:
        soln = solver.run(init_params)
        l = loss(soln.params)
        if l < best_loss:
            best_loss = l
            best_params = soln.params

    best_params = clamp(best_params)
    return best_params, X_mean, X_std, y_mean, y_std

def predict_gp_jax(X_new, params, X_train, y_train, X_mean, X_std, y_mean, y_std, diag=1e-3):
    """Predict using the trained GP with standardized inputs and ARD scaling."""
    X_norm = (X_train - X_mean) / X_std
    y_norm = (jnp.log1p(y_train) - y_mean) / y_std
    X_new_norm = (X_new - X_mean) / X_std
    
    length = jnp.exp(params["log_length"])
    X_scaled = X_norm / length
    X_new_scaled = X_new_norm / length
    
    kernel = jnp.exp(params["log_scale"]) * kernels.ExpSquared()
    gp = GaussianProcess(kernel, X_scaled, diag=diag + jnp.exp(params["log_noise"]))
    
    _, cond_gp = gp.condition(y_norm, X_new_scaled)
    
    # Predict in log-space
    mu_log_norm = cond_gp.loc
    var_log_norm = cond_gp.variance
    
    # Clip normalized values to avoid runaway exp in extreme extrapolation
    mu_log_norm = jnp.clip(mu_log_norm, -15.0, 15.0)
    var_log_norm = jnp.clip(var_log_norm, 1e-8, 10.0)
    
    # Unscale to actual log-space
    mu_log = mu_log_norm * y_std + y_mean
    var_log = var_log_norm * (y_std ** 2)
    
    # Limit mu_log to prevent exp overflow (exp(88) is ~float64 limit)
    mu_log = jnp.clip(mu_log, -20.0, 20.0) 
    
    # Transfrom back to linear space using Delta Method
    mu_linear = jnp.expm1(mu_log)
    std_linear = jnp.exp(mu_log) * jnp.sqrt(var_log)
    
    return mu_linear, std_linear


# --- NN Emulator (Flax/Optax) ---

class EmulatorMLP(nn.Module):
    """A simple MLP that outputs a mean and log_variance."""
    hidden_dims: Sequence[int] = (64, 64, 64)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.swish(x)  # Use SiLU/Swish for smoother gradients than ReLU
        mean = nn.Dense(1)(x)
        log_var = nn.Dense(1)(x)
        # Clip log_var to prevent extreme over-confidence or numerical instability
        log_var = jnp.clip(log_var, -5.0, 10.0) 
        return mean, log_var

def create_train_state(rng, learning_rate=1e-3, input_dim=3, hidden_dims=(64, 64, 64), weight_decay=1e-4):
    model = EmulatorMLP(hidden_dims=hidden_dims)
    params = model.init(rng, jnp.ones((1, input_dim)))['params']
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_X, batch_y):
    def loss_fn(params):
        mean, log_var = state.apply_fn({'params': params}, batch_X)
        mean = mean.flatten()
        log_var = log_var.flatten()
        # Use CRPS instead of NLL for more robust weight estimation
        loss = gaussian_crps(mean, jnp.sqrt(jnp.exp(log_var)), batch_y)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def val_step(state, batch_X, batch_y):
    """Computes validation NLL for early stopping."""
    mean, log_var = state.apply_fn({'params': state.params}, batch_X)
    mean = mean.flatten()
    log_var = log_var.flatten()
    # Use NLL for validation monitoring (it's a global indicator of calibration)
    val_loss = jnp.mean(0.5 * jnp.exp(-log_var) * (batch_y - mean)**2 + 0.5 * log_var)
    return val_loss

def tune_and_train_nn_jax(X, y, key, num_searches=25):
    X_mean, X_std = jnp.mean(X, axis=0), jnp.std(X, axis=0) + 1e-8
    y_log = jnp.log1p(y)
    y_mean, y_std = jnp.mean(y_log), jnp.std(y_log) + 1e-8
    X_norm = (X - X_mean) / X_std
    y_norm = (y_log - y_mean) / y_std

    # Train/Val split
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    key, subkey = jax.random.split(key)
    perms = jax.random.permutation(subkey, n_samples)
    train_idx = perms[:n_train]
    val_idx = perms[n_train:]
    X_train, y_train = X_norm[train_idx], y_norm[train_idx]
    X_val, y_val = X_norm[val_idx], y_norm[val_idx]

    best_val_loss = jnp.inf
    best_state = None
    best_config = None

    width_range = jnp.arange(32, 288, 32) # [32, 64, ..., 256]
    layer_range = jnp.arange(2, 5)        # [2, 3, 4] — minimum 2 layers for expressiveness
    lr_range = 10**jnp.arange(-4.0, -1.5, 0.5) # [10^-4, 10^-3.5, ..., 10^-2]

    fixed_epochs = 2000
    check_interval = 25  # Check validation more frequently
    patience_limit = 5   # 5 * 25 = 125 epochs without improvement before stopping

    for i in range(num_searches):
        key, subkey = jax.random.split(key)
        k1, k2, k3 = jax.random.split(subkey, 3)

        # Sample configuration
        num_layers = int(jax.random.choice(k2, layer_range))
        lr = float(jax.random.choice(k3, lr_range))

        k1_multi = jax.random.split(k1, num_layers)
        width_samples = [int(jax.random.choice(k1_multi[j], width_range)) for j in range(num_layers)]
        h_dims = tuple(width_samples)

        key, init_key = jax.random.split(key)
        state = create_train_state(init_key, lr, input_dim=X.shape[1], hidden_dims=h_dims)

        best_trail_val_loss = jnp.inf
        best_trail_state = state
        patience_counter = 0

        for epoch in range(fixed_epochs):
            state, _ = train_step(state, X_train, y_train)

            if (epoch + 1) % check_interval == 0:
                v_loss = val_step(state, X_val, y_val)
                if v_loss < best_trail_val_loss:
                    best_trail_val_loss = v_loss
                    best_trail_state = state
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience_limit:
                    break

        if best_trail_val_loss < best_val_loss:
            best_val_loss = best_trail_val_loss
            best_state = best_trail_state
            best_config = {"hidden_dims": h_dims, "lr": lr, "epochs": epoch + 1, "val_nll": float(best_trail_val_loss)}

    print(f"Best Config Found: {best_config}")
    return best_state, X_mean, X_std, y_mean, y_std

def predict_nn_jax(X_new, state, X_mean, X_std, y_mean, y_std):
    X_norm = (X_new - X_mean) / X_std
    mean_norm, log_var_norm = state.apply_fn({'params': state.params}, X_norm)
    
    # Unscale mean and variance (in log-space)
    mu_log = mean_norm.flatten() * y_std + y_mean
    var_log = jnp.exp(log_var_norm.flatten()) * (y_std ** 2)
    
    # Harmonized with GP: Use the Delta Method (Median) to avoid uncertainty-induced bias.
    # The log-normal mean (exp(mu+0.5*var)) shifts the prediction significantly in 
    # regions of high uncertainty, which penalizes the likelihood incorrectly 
    # when recovering parameters of a deterministic function.
    mean_linear = jnp.expm1(mu_log)
    std_linear = jnp.exp(mu_log) * jnp.sqrt(var_log)
    std_linear = jnp.sqrt(jnp.maximum(std_linear**2, 1e-12))
    
    # Use a much smaller floor to preserve constraints on small signal
    std_linear = jnp.maximum(std_linear, 1e-4) 
    
    return mean_linear, std_linear
