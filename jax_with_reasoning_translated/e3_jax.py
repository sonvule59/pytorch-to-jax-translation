import jax
from jax import numpy as jnp
from jax.nn import Linear
from jax.optim import RootZeroGradient

# Generate synthetic data
jax.random.seed(42)
X = jnp.random.uniform(0, 10, (100, 1))
y = 2 * X + 3 + jnp.random.normal(0, 1, (100, 1))

# Define the Linear Regression Model
class CustomActivationModel(jax.nn.Module):
    def __init__(self):
        super(CustomActivationModel, self).__init__()
        self.linear = Linear(1, 1)

    def forward(self, x):
        # Compute linear transformation and apply custom activation
        x_transformed = self.linear(x)
        return jax.tanh(x_transformed) + x_transformed

# Initialize the model, loss function, and optimizer
model = CustomActivationModel()
criterion = jax.mse_loss
optimizer = RootZeroGradient(model, steps=1000)

# Training loop adapted for JAX's event-driven nature
# Note: This is a simplified representation; actual training loops
#       involve jax.vmap and event loops. For brevity, using a batch example.
for step in jax.range(1000):
    with jax.blocking scopes.ExpectedOutputDevice(1000):
        # Forward pass and loss calculation
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass and optimization
        optimizer.step()
        optimizer.apply(loss)

# Display the learned parameters
weight, bias = model.linear.get_parameters()
print(f"Learned weight: {weight()[0].item():.4f}, Learned bias: {bias()[0].item():.4f}")

# Plotting (Note: JAX doesn't have built-in plotting; use external libraries like matplotlib)
# This example skips plotting for brevity but should be similar to the PyTAX version