import jax
import jax.numpy as jnp
import numpy as np
import optax

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + np.random.randn(100, 1)  # Linear relationship with noise

# Define the Linear Regression Model
def linear_regression_model(params):
    w, b = params
    def model(x):
        return w * x + b
    return model

# Initialize the model parameters
initial_params = jnp.array([0.0, 0.0])

# Define the loss function
def loss(params, X, y):
    model = linear_regression_model(params)
    preds = model(X)
    return jnp.mean(jnp.square(preds - y))

# Define the training function with optimizer
def train_step(params, opt_state, X, y):
    # Compute gradients
    grads = grad(loss)(params, X, y)
    # Update parameters
    updates, opt_state = optax.sgd(0.01)(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Initialize the optimizer
init_opt_state = optax.sgd(0.01).init(initial_params)

# Training loop
epochs = 1000
params = initial_params
opt_state = init_opt_state
for epoch in range(epochs):
    params, opt_state = train_step(params, opt_state, X, y)

# Display the learned parameters
w, b = params
print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.disable_jit():
    predictions = linear_regression_model(params)(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")