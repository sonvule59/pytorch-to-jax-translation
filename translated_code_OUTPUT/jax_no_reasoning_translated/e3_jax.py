import jax.numpy as jnp
from jax import multivariate_normal as jmvn
from jax.nn import LinearRegression
from jax.optim import MinimalGradientDescent

# Generate synthetic data
jax.random.seed(42)
X = jnp.random.uniform(0, 10, (100, 1))
y = 2 * X + 3 + jnp.random.normal(0, 0.2, (100, 1))

# Define the Linear Regression Model
model = LinearRegression(input_shape=(1,))

# Initialize parameters using jax.random.normal
initial_params = jnp.zeros((1,))
optimizer = MinimalGradientDescent(initial_params, jax.lru_cache(max_size=100))

# Training loop with JAX automatically handling gradients
for i in range(1000):
    # Forward and backward pass
    optimizer.apply(jax.fn.vectorize(model.apply)(X))[0]
    loss = optimizer.loss

    # Minimize the loss
    optimizer.minimize(loss)

    # Log every 100 epochs
    if i + 1 % 100 == 0:
        print(f"Epoch [{i + 1}/{1000}], Loss: {loss.item():.4f}")

# Get learned parameters
weight, bias = optimizer.get_params()[0]
print(f"Learned weight: {weight.item():.4f}, Learned bias: {bias.item():.4f}")

# Plotting (Note: JAX doesn't have built-in plotting; use a JAX-compatible plot library)
# For demonstration, using jax.plot.jaxplot (if available) or similar
# This is a placeholder for actual plotting code