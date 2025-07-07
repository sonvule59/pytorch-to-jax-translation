# (only the code, no explanations)
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax.lax import scan
from jax import random
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = jnp.array(np.random.rand(100, 1) * 10)  # 100 data points between 0 and 10
y = 2 * X + 3 + jnp.array(np.random.randn(100, 1))  # Linear relationship with noise

# Define the Linear Regression Model within a CustomActivationModel class
class CustomActivationModel:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def custom_activation(self, x):
        return jnp.tanh(x) + x

    def forward(self, x):
        return self.custom_activation(self.w * x + self.b)

# Initialize the model, loss function, and optimizer
# Define the loss function
def loss(params, x, y):
    w, b = params
    y_pred = CustomActivationModel(w, b).forward(x)
    return jnp.mean((y_pred - y) ** 2)

# Define the optimizer
def update(params, opt_state, x, y):
    # Compute the gradient
    grad_loss = grad(loss)(params, x, y)
    # Update the parameters
    return params - grad_loss, opt_state

# Initialize the parameters
initial_w = jnp.array(0.0)
initial_b = jnp.array(0.0)
params = (initial_w, initial_b)

# Define the training loop
def train_step(params, opt_state, x, y):
    # Compute the loss and update the parameters
    params, opt_state = update(params, opt_state, x, y)
    return params, opt_state

# Initialize the optimizer
opt_init, opt_update = jaxopt.SGD.create(initial_params=params, learning_rate=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    params, opt_state = train_step(params, opt_state, X, y)
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss(params, X, y):.4f}")

# Display the learned parameters
w, b = params
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Plot the model fit to the train data
plt.figure(figsize=(4, 4))
plt.scatter(X, y, label='Training Data')
plt.plot(X, w * X + b, 'r', label='Model Fit')
plt.legend()
plt.show()

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.numpy as np:
    predictions = CustomActivationModel(w, b).forward(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
