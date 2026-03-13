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
        # Simplify the Gaussian LL and clip to prevent -inf propagation
        scaled_err = (y_obs - y_sim) / obs_error
        ll = -0.5 * jnp.sum(scaled_err**2)
        return lp + jnp.clip(ll, a_min=-1e10, a_max=0.0)
    return logprob

def get_gp_logprob(predict_fn, gp_params, X_train, y_train, X_mean, X_std, y_mean, y_std, x_obs, y_obs, obs_error, prior_bounds):
    def logprob(params):
        lp = logprior_fn(params, prior_bounds)
        theta_mat = jnp.tile(params, (len(x_obs), 1))
        X = jnp.column_stack((theta_mat, x_obs))
        y_pred, gp_std = predict_fn(X, gp_params, X_train, y_train, X_mean, X_std, y_mean, y_std)
        var = obs_error**2 + gp_std**2
        scaled_err = (y_obs - y_pred) / jnp.sqrt(var)
        ll = -0.5 * jnp.sum(scaled_err**2) # omit constant sum
        return lp + jnp.clip(ll, a_min=-1e10, a_max=0.0)
    return logprob

def get_nn_logprob(predict_fn, nn_state_tuple, x_obs, y_obs, obs_error, prior_bounds):
    def logprob(params):
        lp = logprior_fn(params, prior_bounds)
        theta_mat = jnp.tile(params, (len(x_obs), 1))
        X = jnp.column_stack((theta_mat, x_obs))
        y_pred, nn_std = predict_fn(X, *nn_state_tuple)
        var = obs_error**2 + nn_std**2
        scaled_err = (y_obs - y_pred) / jnp.sqrt(var)
        ll = -0.5 * jnp.sum(scaled_err**2)
        return lp + jnp.clip(ll, a_min=-1e10, a_max=0.0)
    return logprob

import blackjax.mcmc.random_walk as random_walk

def run_mcmc_blackjax(logprob_fn, initial_position, num_steps, rng_key, sigma=0.05):
    # scale sigma for N dims
    sigma_vec = jnp.full(initial_position.shape, sigma)
    rmh = random_walk.normal_random_walk(logprob_fn, sigma=sigma_vec)
    initial_state = rmh.init(initial_position)
    
    @jax.jit
    def one_step(state, rng_key):
        state, info = rmh.step(rng_key, state)
        return state, state.position

    keys = jax.random.split(rng_key, num_steps)
    _, positions = jax.lax.scan(one_step, initial_state, keys)
    
    return positions
