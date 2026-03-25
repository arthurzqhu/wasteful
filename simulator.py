import jax
import jax.numpy as jnp

def simulator(params, sim_func):
    """
    params: 2D array of shape (n_ppe, M) where columns are [theta_1, ..., theta_{M-1}, x].
    sim_func: A callable `f(theta, x)` that defines the mathematical structure of the target.
    """
    theta = params[:, :-1]
    x = params[:, -1]
    return sim_func(theta, x)

def true_process(x, true_theta, sim_func):
    """
    Convenience wrapper: evaluates the simulator at a single theta across many x values.
    """
    params = jnp.column_stack([jnp.tile(true_theta, (len(x), 1)), x.reshape(-1, 1)])
    return simulator(params, sim_func)
