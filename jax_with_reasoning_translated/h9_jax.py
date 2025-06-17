import jax
from jax import numpy as jnp
from jax.vjp import jit
from jax import random

# Define a simple model
class SimpleModel(jax.vjp.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = jax.vjp.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Generate synthetic data
X = jnp.random.normal(0, 1, (1000, 10))
y = jnp.random.normal(0, 1, (1000, 1))

# Convert to JAX array format
dataset = jnp.array([X[i], y[i]] for i in range(1000)])
dataloader = jnp.array([dataloader[i] for i in range(len(dataloader))])

# Initialize model, loss function, and optimizer
model = SimpleModel().jit()
criterion = jax.vjp.MSELoss()
optimizer = jax.vjp.LinearOptimizer(model.parameters(), jax.vjp.ParameterUpdateStepSize(0.001))

# Enable mixed precision training
scaler = jax.vjp.GradScaler()

# Training loop
epochs = 5
for epoch in range(epochs):
    for i in range(0, len(dataloader), 32):
        X_batch, y_batch = dataloader[i:i+32]
        X_batch, y_batch = X_batch.to(jnp.float32), y_batch.to(jnp.float32)

        # Forward pass under autocast
        with jax.vjp.autocast(X_batch, X_batch.shape):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

        # Backward pass with scaled gradients
        optimizer.apply_scales()
        optimizer.apply_updates(loss)
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Test the model on new data
X_test = jnp.random.normal(0, 1, (5, 10)).jit()
with jax.vjp.autocast(X_test, X_test.shape):
    predictions = model(X_test)
    print("Predictions:", predictions)
JAX_CODE