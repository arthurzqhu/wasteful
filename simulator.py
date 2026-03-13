import jax
import jax.numpy as jnp

def simulator(params, sim_func):
    """
    params: 2D array of shape (n_samples, M) where columns are [theta_1, ..., theta_{M-1}, x].
    sim_func: A callable `f(theta, x)` that defines the mathematical structure of the target.
    """
    theta = params[:, :-1]
    x = params[:, -1]
    return sim_func(theta, x)

def true_process(x, true_theta, sim_func):
    """
    The ground-truth target physical process parameterized by `true_theta`.
    """
    # sim_func expects 2D theta matrix, so we expand true_theta and x
    theta_mat = jnp.tile(true_theta, (len(x), 1))
    return sim_func(theta_mat, x)

def generate_ppes(n_samples, bounds, key, sim_func):
    """
    Generate PPEs dynamically for N bounds. 
    """
    keys = jax.random.split(key, len(bounds))
    
    samples = []
    for i, b in enumerate(bounds):
        s = jax.random.uniform(keys[i], shape=(n_samples, 1), minval=b[0], maxval=b[1])
        samples.append(s)
        
    X = jnp.hstack(samples)
    y = simulator(X, sim_func)
    return X, y
