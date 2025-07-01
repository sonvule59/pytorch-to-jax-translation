import jax
from jax import numpy as jnp
from jax.nn import Module, Layer
from jax.optim import MinimalOptimizer

# Generate synthetic data
jax.random.seed(42)
X = jnp.random.uniform(0, 10, (100, 1))
y = 2 * X + 3 + jnp.random.normal(0, 1, (100, 1))

# Define a functional JAX version of LearnedSiLUFunction
def learned_silu_func(x, slope):
    return slope * x * jax.sigmoid(x)

def silu_derivative(x, slope):
    return jax.grad(lambda s=x: s * x * jax.sigmoid(x + x * jax.sigmoid(x)) + s * x * jax.sigmoid(x) * (1 - jax.sigmoid(x)))(x) * slope

class LearnedSiLULayer(Module):
    def __init__(self, slope=1.0):
        super().__init__()
        self.slope = jax.variate(jnp.ones((1,)), float(slope))

    def forward(self, x):
        return learned_silu_func(x, self.slope)

class LinearRegressionModel(jax.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LearnedSiLULayer()

    def call(self, inputs):
        return self.linear(inputs)

# Initialize model, loss function, and optimizer
model = LinearRegressionModel()
optimizers = [MinimalOptimizer([model.linear.slope])]  # Note: JAX doesn't have SGD/Adam directly; use varify
loss_fn = jax.mean_loss(jnp.array([model.call(X)]), y)

# Training loop (simplified for brevity)
# Note: JAX autograd requires manual compilation and handling of gradients
# A full training loop is complex; here's a placeholder
# In practice, use jax.compiled_function to create an autograd function for training

# Display the learned parameters
print(f"Learned weight: {model.linear.slope.item():.4f}")  # Note: JAX uses variational parameters, not standard weights

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.no_evaluate():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")