import jax
import jax.nn as nn
import torch
import optax
import jax.optim as jax.optim

# Initialize JAX RNG
rng = jax.random.PRNGKeyGenerator().generate_jax_rng(seed=42)

# Generate synthetic data
X = jax.vmap(nn.array)(torch.rand(100, 1), seed=rng).mul_axis(0, 1) * 10
y = 2 * X + 3 + jax.random.normal(rng, (100, 1), seed=rng + 1)

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
model = jax.vmap(nn.create_function)(model, jax.vmap(nn.array), axis=0)

# Define JAX MSELoss
criterion = jax.nn.mse_loss

# Initialize JAX SGD Optimizer
sgd = jax.optim.SGD(
    functions=[jax.vmap(nn.Linear).partial_forward()],
    parameters=[jax.vmap(nn.Linear).partial_params()],
    lr=0.01,
)

# Training loop using JAX's automatic differentiation
for epoch in range(1000):
    # Forward pass and loss calculation
    with jax.control() as ctrl:
        predictions = model(X, ctrl=ctrl)
        loss = criterion(predictions, y)

    # Backward pass and optimization
    sgd.minimize(loss, drop=0.)

    # Logging every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{1000}], Loss: {loss.item():.4f}")

# Display learned parameters
w, b = model.linear.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Testing on new data
X_test = jax.vmap(nn.array)([[4.0], [7.0]])
with jax.no_grad():
    predictions = model(X_test, axis=0)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")