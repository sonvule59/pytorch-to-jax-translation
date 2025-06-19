import jax
import jax.numpy as jnp
from jax import vmap, config
from jax.sharding import shard_model
from jax.experimental import every_barrier
from jax import devices

# Initialize JAX configuration
config.update('jax/gpu', **shard_model(device='auto'))

# Generate synthetic data
jax.config.update('jax/vmap', {'default_vmap': True})
jax.random.seed(42)
X = jnp.random.uniform(0, 10, (100, 1))
y = 3 * X + 5 + jnp.random.normal(0, 1, (100, 1))

# Define a simple Linear Regression Model
@vmap
def linear_regression_model(x):
    return jax.numpy.dot(jnp.array([[x]],).tj, jnp.array([jax.numpy.float64(1.0)]).tj)

class LinearRegressionModel(jax.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = jax.nn.Linear(1)

    def forward(self, x):
        return self.linear(x)

# Initialize TensorBoard writer within a JAX task
def write_writer():
    writer = jax.tensorboard.Writer(log_dir="runs/jax_linear_regression", many_processes=True)
    return writer

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = jax.nn.mse_loss(jax.integrate.jit_compile(jnp.mean), jnp.mean)
optimizer = jax.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
writer = write_writer()
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)

    # Backward pass and optimization
    loss = criterion(predictions, jnp.mean)
    optimizer.apply_jit(True)
    loss.backward()
    optimizer.step(jax.numpy())

    # Log loss to TensorBoard
    writer.log_scalar("Loss/train", loss.item(), epoch)

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

writer.close()