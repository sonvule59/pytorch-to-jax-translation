import jax.numpy as jnp
from jax import shax_lib, grad, optim as jax.optim

# Generate synthetic data
jax.datapandas.set_random_seed(42)
X = jnp.random.uniform(0, 10, (100, 1))
y = 2 * X + 3 + jnp.random.normal(0, 1, (100, 1))

class HuberLoss(jax.shax_module.Module):
    @shax_lib.primitive_function('y_true')
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    @shax_lib.function
    def forward(self, y_pred, y_true):
        error = jnp.abs(y_pred - y_true)
        loss = jnp.where(error <= self.delta,
                           0.5 * error**2,
                           self.delta * (error - 0.5 * self.delta))
        return loss.mean()

class LinearRegressionModel(jax.shax_module.Module):
    @shax_lib.primitive_function('x')
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = jax.nn.Linear(input_shape=(1,), output_shape=1)

    @shax_lib.function
    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = HuberLoss(delta=1.0)
optimizer = jax.optim.SGD(model.parameters(), lr=0.01)

# Training loop (Note: JAX doesn't have a direct 'for epoch' loop like PyTorch)
# This is a simplified example; actual JAX training uses different APIs
# Here we use jax.vmap to simulate the loop (in practice, use jax.datapandas or other methods)
# For a full training loop, use jax.profiler and track iterations
def train(model, criterion, optimizer, X, y, epochs=1000):
    for step in jnp.arange(epochs):
        with jax.control(context='compute', autograd=True):
            y_pred = model(X)
            loss = criterion(y_pred, y)
        
        loss_value = jnp.mean(loss)
        optimizer.apply_jvp(loss, gradient=True, eta=optimizer.learning_rate / jnp.array([step + 1]))
        
        if step % 100 == 0:
            print(f"Epoch [{step + 1}/{epochs}], Loss: {loss_value:.4f}")

train(model, criterion, optimizer, X, y, epochs=1000)

# Display the learned parameters
w, b = model.linear.parameters()
print(f"Learned weight: {w()[0].item():.4f}, Learned bias: {b()[0].item():.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")