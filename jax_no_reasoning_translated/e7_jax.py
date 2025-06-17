import jax
from jax import numpy as jnp
from jax.shaping import Identity
from jax.vjp import jvpiertoken
jvpiertoken()
import jax.nn as jnn
import jax.optim as jopt
import jax.optim.tracker as jopt_tracker
import jax.random as jrandom

# Define a simple model
@jax.vjp
class SimpleModel(jnn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = jnn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Create and train the model
jax.random.seed(42)
model = SimpleModel()
criterion = jax.mse_loss
optimizer = jopt.SGD(
    initializer=jax.random_uniform(jnp.ones((1,)), 0.01),
    optimizer_params={
        'initializer_range': (jax.random.uniform(0, 1.0), 0.1),
        'learning_rate': 0.01,
       'momentum': 0.9,
        'weight_decay': 0.0001,
        'norm_clipping': True,
        'decay_steps': 10000,
       'min_learning_rate': 1e-6,
       'max_learning_rate': 1e-3,
       'steps_per_epoch': epochs,
        'gamma': 0.9,
        'epsilon_min': 1e-8,
        'epsilon_max': 1e-6,
        'opti_mixed_precision': jax.float32,
       'strict_mode': jax.strict_mode.STRICT,
    }
)
# Note: The optimizer configuration might need adjustment based on your desired batch size and other parameters.
# The code below assumes 'epochs' is defined similarly to the Torch code, but JAX uses different batch defaults.
# For direct translation, the code would need batch dimensions adjusted. Here, we'll use a batch size of 32 for demonstration.
batch_size = 32
for epoch in range(epochs):
    # Generate data for the epoch
    X = jrandom.uniform(jnp.array([0., 0., 0.]), jnp.newaxis, dtype=jnp.float32).repeat(batch_size, -1)
    y = 3 * X + 2 + jrandom.uniform(0., 1., dtype=jnp.float32).repeat(batch_size, -1)
    
    optimizer.update_batch(
        jrandom.normal(jnp.zeros((batch_size, 1)), scale=0.1),  # Input data with noise
        model(X),  # Predictions
        y,  # True labels
    )
    # The loss calculation and backprop are handled automatically by JAX's automatic differentiation

# Save the model
jax.save_model(model, "model.jax")

# Load the model
loaded_model = SimpleModel()
loaded_model.load_state_dict(jax.load_state_dict(jnp.array([model.state_dict()])[0]))

# Verify the model
X_test = jnp.array([jnp.array([0.5,], dtype=jnp.float32), jnp.array([1.0,], dtype=jnp.float32), jnp.array([1.5,], dtype=jnp.float32)])
with jax.no_grad():
    predictions = loaded_model(X_test)
    print(f"Predictions after loading: {predictions}")