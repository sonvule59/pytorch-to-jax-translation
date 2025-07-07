import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)
# Generate data
X = jnp.array(np.random.rand(100, 1) * 10)
y = 2 * X + 3 + jnp.array(np.random.randn(100, 1))

# Save to CSV
data = jnp.concatenate((X, y), axis=1)
df = pd.DataFrame(data, columns=['X', 'y'])
df.to_csv('data.csv', index=False)

# Load data
df = pd.read_csv('data.csv')
X = jnp.array(df['X'].values).reshape(-1, 1)
y = jnp.array(df['y'].values).reshape(-1, 1)

# Define model
def model(params, X):
    w, b = params
    return w * X + b

# Define loss function
def loss(params, X, y):
    preds = model(params, X)
    return jnp.mean((preds - y) ** 2)

# Define optimizer
def update(params, opt_state, X, y):
    grads = jax.grad(loss)(params, X, y)
    return grads, opt_state

# Initialize parameters
initial_params = jnp.array([0.0, 0.0])
opt_init, opt_update = jaxopt.SGD.create(initial_params, learning_rate=0.01)

# Training loop
for epoch in range(1000):
    # Get batch
    batch_size = 32
    indices = jnp.arange(X.shape[0])
    batch_indices = jax.random.permutation(jax.random.PRNGKey(42), indices, axis=0)[:batch_size]
    batch_X = X[batch_indices]
    batch_y = y[batch_indices]

    # Update parameters
    params, opt_state = opt_update(epoch, opt_state, batch_X, batch_y)

# Print learned parameters
params = opt_state.params
print(f"Learned weight: {params[0].item():.4f}, Learned bias: {params[1].item():.4f}")

# Test
X_test = jnp.array([[4.0], [7.0]])
preds = model(params, X_test)
print(f"Predictions for {X_test.tolist()}: {preds.tolist()}")