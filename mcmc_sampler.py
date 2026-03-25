import jax
import jax.numpy as jnp
import blackjax
from simulator import simulator

def logprior_fn(params, bounds):
    """Uniform prior on tuning parameters theta."""
    lower = jnp.array([b[0] for b in bounds])
    upper = jnp.array([b[1] for b in bounds])
    # Ensure it returns a 0-dimensional jax scalar, not a 1D tensor
    in_bounds = jnp.all((params >= lower) & (params <= upper))
    return jnp.where(in_bounds, 0.0, -1e10).squeeze()

def get_true_logprob(x_obs, y_obs, obs_error, prior_bounds, sim_func):
    def logprob(params):
        lp = logprior_fn(params, prior_bounds)
        theta_mat = jnp.tile(params, (len(x_obs), 1))
        X = jnp.column_stack((theta_mat, x_obs))
        y_sim = simulator(X, sim_func)
        # Handle array-based obs_error for relative error support
        var = obs_error**2
        scaled_err = (y_obs - y_sim)**2 / var
        ll = -0.5 * jnp.sum(scaled_err + jnp.log(2 * jnp.pi * var))
        return lp + jnp.clip(ll, a_min=-1e10, a_max=0.0)
    return logprob

def get_gp_logprob(predict_fn, gp_params, X_train, y_train, X_mean, X_std, y_mean, y_std, x_obs, y_obs, obs_error, prior_bounds, uncertainty_multiplier=1.0):
    def logprob(params):
        lp = logprior_fn(params, prior_bounds)
        theta_mat = jnp.tile(params, (len(x_obs), 1))
        X = jnp.column_stack((theta_mat, x_obs))
        y_pred, gp_std = predict_fn(X, gp_params, X_train, y_train, X_mean, X_std, y_mean, y_std)
        
        emulator_var = (gp_std * uncertainty_multiplier)**2
        var = obs_error**2 + emulator_var
        
        # Full Gaussian likelihood includes log(var) term, critical if var isn't constant
        ll = -0.5 * jnp.sum((y_obs - y_pred)**2 / var + jnp.log(2 * jnp.pi * var))
        return lp + jnp.clip(ll, a_min=-1e10, a_max=0.0)
    return logprob

def get_nn_logprob(predict_fn, nn_state_tuple, x_obs, y_obs, obs_error, prior_bounds, uncertainty_multiplier=1.0):
    def logprob(params):
        lp = logprior_fn(params, prior_bounds)
        theta_mat = jnp.tile(params, (len(x_obs), 1))
        X = jnp.column_stack((theta_mat, x_obs))
        y_pred, nn_std = predict_fn(X, *nn_state_tuple)
        
        emulator_var = (nn_std * uncertainty_multiplier)**2
        var = obs_error**2 + emulator_var
        
        # Use full Gaussian LL
        ll = -0.5 * jnp.sum((y_obs - y_pred)**2 / var + jnp.log(2 * jnp.pi * var))
        return lp + jnp.clip(ll, a_min=-1e10, a_max=0.0)
    return logprob

def run_mcmc_blackjax(logprob_fn, initial_position, num_steps, rng_key):
    warmup_key, sample_key = jax.random.split(rng_key)
    
    adapt = blackjax.window_adaptation(blackjax.nuts, logprob_fn)
    
    # Run warmup
    num_warmup = num_steps//2
    res, info = adapt.run(warmup_key, initial_position, num_warmup)
    last_state = res.state
    kernel = blackjax.nuts(logprob_fn, **res.parameters)
    
    # Sample using the tuned kernel
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel.step(rng_key, state)
        return state, state.position

    keys = jax.random.split(sample_key, num_steps)
    _, positions = jax.lax.scan(one_step, last_state, keys)
    
    return positions
