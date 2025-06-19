import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.nn import LayerModule
from jax.optim import BasicGradientDescent

# Define a simple model
@jit
def simple_model(features, inputs):
    # Define the forward operation
    linear_layer = lambda x: inputs * 3 + 2
    output = linear_layer(features)
    return output

# Or using JAX vmap for better performance
@jit
def simple_model_vmap(inputs):
    # Define the forward operation using vmap
    linear_layer = vmap(lambda x: x * 3 + 2)
    output = linear_layer(inputs)
    return output

class SimpleModelJAX(nn.Module):
    def __init__(self):
        super(SimpleModelJAX, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        pass  # Not using JAX here; for pure PyTorch-JAX compatibility, use native JAX functions

    @jit
    def _forward_jit(self, x):
        return x * 3 + 2

# Set up JAX environment
jax.config.update_default(jax._default_jax_config['jax__collect_tracing_info'])
jax.init()

# Define inputs and targets
X = jnp.array([[0.5], [1.0], [1.5]])
y = 3 * X + 2 + jnp.random.normal(0, 0.1, X.shape)

# Initialize the model
model = SimpleModelJAX()
# Set up basic gradient descent
optimizer = BasicGradientDescent(model.parameters(), learning_rate=0.01)

# Training loop
for epoch in range(100):
    # Compute predictions and loss
    predictions = model(X)
    loss = jax.numpy.nn.mse_loss(jnp.array([predictions], jnp.array(y)), jnp.array([y]))
    # Compute gradients and update
    gradients = optimizer.compute_gradients(predictions, gradients_of_loss)
    for grad_source, grad_var, grad_target in gradients:
        optimizer.step(grad_source, grad_target, grad_var)
    # Apply JAX JIT for numerical stability
    with jit()(model).compute()
    pass

# Save the model
# Note: JAX models don't have built-in save/load like PyTorch, simulate or use alternative methods
# Here, we'll create a new model and load the parameters as arrays
loaded_model = SimpleModelJAX()
loaded_model.linear.set_default_rng().apply(jnp.array([model.linear.weights(), model.linear.bias]))
print(f"Loaded model predictions: {loaded_model(simple_model_vmap(X))}")


<think>
Okay, I need to convert the given PyTorch code to JAX. Let me start by understanding what the original code does. It's a simple linear regression model using Mean Squared Error loss. The model has one linear layer, trains for 100 epochs, saves the model, loads it, and tests.

First, JAX uses different APIs for computation graphs. The user provided some JAX code, but I need to make sure it's equivalent. Let's break it down step by step.

The PyTorch model uses a Linear layer. In JAX, I can define a module with a custom function. Since JAX is eager by default, I'll use @jit to make the functions eager. The forward pass in PyTorch is replaced by a JAX function. The user's example uses a lambda inside @jit, which should work.

For the training loop, PyTorch uses optimizer steps with zero gradients. In JAX, the BasicGradientDescent optimizer handles gradients automatically. The code provided by the user uses compute_gradients and step, which seems correct. However, in JAX, the gradients are computed during the backward pass, so the loop structure might need to account for that.

Saving and loading models: PyTorch has save/load_state_dict. JAX doesn't have a direct equivalent, so the user's approach of setting weights as arrays is a workaround. That's acceptable for the example.

The verification part: After loading, they predict on test data. In JAX, since everything is eager, this should work as is, but need to make sure that the loaded model uses the correct parameters.

Looking at the JAX code provided by the user, there's a SimpleModelJAX class with a _forward_jit method. But in the training loop, they call model(X), which would use the _forward_jit if the model is compiled with it. However, in JAX, when you call a compiled function, it's executed in the eager session. Maybe the user's code has some issues with how the model is applied.

Wait, in the user's JAX code, the model's forward is defined as a compiled function. So when they do predictions = model(X),