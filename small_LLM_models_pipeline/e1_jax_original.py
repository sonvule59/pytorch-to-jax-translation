import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + np.random.randn(100, 1)  # Linear relationship with noise

# Define the Linear Regression Model
class LinearRegressionModel:
    def __init__(self):
        self.w = jnp.array(0.0)
        self.b = jnp.array(0.0)

    def predict(self, x):
        return self.w * x + self.b

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
def loss(model, x, y):
    return jnp.mean(jnp.square(model.predict(x) - y))

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model.predict(X)
    loss_val = loss(model, X, y)

    # Backward pass and optimization
    # This is where the problem is: how to compute gradients and update parameters
    # The code here is incomplete

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_val:.4f}")

# Display the learned parameters
print(f"Learned weight: {model.w:.4f}, Learned bias: {model.b:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.disable_jit():
    predictions = model.predict(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
