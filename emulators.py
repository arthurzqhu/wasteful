import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from tinygp import kernels, GaussianProcess
import jaxopt

# --- GP Emulator (tinygp) ---

def build_gp(params, X, y=None, diag=1e-3):
    """Builds a tinygp GaussianProcess object."""
    kernel = jnp.exp(params["log_scale"]) * kernels.ExpSquared(scale=jnp.exp(params["log_length"]))
    return GaussianProcess(kernel, X, diag=diag + jnp.exp(params["log_noise"]))

def train_gp_jax(X, y, diag=1e-3):
    """
    Train a JAX-based Gaussian Process using jaxopt with standard scaling.
    """
    # Standardize X and y for better optimization
    X_mean = jnp.mean(X, axis=0)
    X_std = jnp.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    y_mean = jnp.mean(y)
    y_std = jnp.std(y) + 1e-8
    y_norm = (y - y_mean) / y_std
    
    def loss(params):
        gp = build_gp(params, X_norm, diag=diag)
        return -gp.log_probability(y_norm)

    params = {
        "log_scale": jnp.log(1.0),
        "log_length": jnp.log(1.0),
        "log_noise": jnp.log(diag)
    }

    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params)
    
    return soln.params, X_mean, X_std, y_mean, y_std

def predict_gp_jax(X_new, params, X_train, y_train, X_mean, X_std, y_mean, y_std, diag=1e-3):
    """Predict using the trained GP with standardized inputs."""
    X_norm = (X_train - X_mean) / X_std
    y_norm = (y_train - y_mean) / y_std
    X_new_norm = (X_new - X_mean) / X_std
    
    gp = build_gp(params, X_norm, diag=diag)
    _, cond_gp = gp.condition(y_norm, X_new_norm)
    
    mu = cond_gp.loc * y_std + y_mean
    var = cond_gp.variance * (y_std ** 2)
    return mu, jnp.sqrt(var)


# --- NN Emulator (Flax/Optax) ---

class EmulatorMLP(nn.Module):
    """A simple MLP that outputs a mean and log_variance."""
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        mean = nn.Dense(1)(x)
        log_var = nn.Dense(1)(x)
        return mean, log_var

def create_train_state(rng, learning_rate=1e-3, input_dim=3):
    model = EmulatorMLP()
    params = model.init(rng, jnp.ones((1, input_dim)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_X, batch_y):
    def loss_fn(params):
        mean, log_var = state.apply_fn({'params': params}, batch_X)
        mean = mean.flatten()
        log_var = log_var.flatten()
        # Negative Log-Likelihood for Gaussian
        loss = 0.5 * jnp.exp(-log_var) * (batch_y - mean)**2 + 0.5 * log_var
        return jnp.mean(loss)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_nn_jax(X, y, key, epochs=2000, lr=5e-3):
    # Standardize inputs and outputs
    X_mean, X_std = jnp.mean(X, axis=0), jnp.std(X, axis=0) + 1e-8
    y_mean, y_std = jnp.mean(y), jnp.std(y) + 1e-8
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    state = create_train_state(key, learning_rate=lr, input_dim=X.shape[1])
    
    for _ in range(epochs):
        state, loss = train_step(state, X_norm, y_norm)
        
    return state, X_mean, X_std, y_mean, y_std

def predict_nn_jax(X_new, state, X_mean, X_std, y_mean, y_std):
    X_norm = (X_new - X_mean) / X_std
    mean_norm, log_var_norm = state.apply_fn({'params': state.params}, X_norm)
    
    # Unscale mean and variance
    mean = mean_norm.flatten() * y_std + y_mean
    
    var_norm = jnp.exp(log_var_norm.flatten())
    std = jnp.sqrt(var_norm) * y_std
    
    return mean, std
