import jax
import jax.nn as nn
import jax.optim as jax.optim

# Initialize RNG
ranger = jax.random.PRNGKeyGenerator().generate_jax_rng(seed=42)

# Generate synthetic data
X = jax.vmap(jax.numpy)(jax.random.normal(ranger, (100, 1), seed=42))
y = 2 * jax.vmap(jax.numpy)(X) + 3 + jax.random.normal(ranger, (100, 1), seed=42)

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    @jax.vmap(nn.FunctionBuilder)
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    @jax.vmap(nn.FunctionBuilder)
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
criterion = nn.mse_loss
optimizer = jax.optim.SGD(
    functions=[jax.vmap(nn.Linear).partial_forward()],
    parameters=[jax.vmap(nn.Linear).partial_params()],
    lr=0.01,
)

# Training loop using JAX's map functions
for epoch in jax.range(epochs=1000):
    with jax.blocking.recompute():
        predictions = model(X)
        loss = criterion(predictions, y)

    # Compute gradients and update
    jax.grad(loss, variables=(model, y), enforce_fixed_parameters=True)[0](X)
    optimizer.apply(predictions)

# Display the learned parameters
w, b = model.linear.parameters()
print(f"Learned weight: {w()[0]:.4f}, Learned bias: {b()[0]:.4f}")

# Testing on new data
X_test = jax.vmap(jax.numpy)([[4.0], [7.0]])
with jax.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")