# (replace the following line)
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.lax import scan
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = jnp.array(np.random.rand(100, 1) * 10)  # 100 data points between 0 and 10
y = 2 * X + 3 + jnp.random.randn(100, 1)  # Linear relationship with noise

# Define the LearnedSiLU function
def learned_silu(x, slope):
    return slope * x * jnp.sigmoid(x)

# Define the model
def model(params, x):
    slope = params['slope']
    return learned_silu(x, slope)

# Define the loss function
def loss(params, x, y):
    return jnp.mean((model(params, x) - y) ** 2)

# Define the optimizer and training loop
def train_model():
    # Initial parameters
    initial_params = {
        'slope': jnp.array(1.0)
    }

    # Define the update step
    def update_step(params, x, y):
        # Compute the gradient
        grad_loss = grad(loss)(params, x, y)
        # Update parameters
        return params - grad_loss * 0.01  # Learning rate 0.01

    # Training loop
    for epoch in range(1000):
        # Forward pass
        current_loss = loss(initial_params, X, y)
        print(f"Epoch [{epoch + 1}/1000], Loss: {current_loss:.4f}")

        # Update parameters
        initial_params = update_step(initial_params, X, y)

    # Display the learned parameters
    print(f"Learned slope: {initial_params['slope']:.4f}")

train_model()

# Wait, but in the original PyTorch code, the model is using a custom SiLU function with a learnable slope parameter. The JAX code above is not using the custom SiLU function with a learnable slope parameter. It's just using a simple learned slope parameter with the standard SiLU function. So the JAX code is not equivalent to the PyTorch code. The JAX code is missing the backward pass for the learned slope parameter. In the PyTorch code, the backward pass for the learned slope is computed as grad_slope = grad_output * x * sigmoid_x. In the JAX code, the gradient is computed using the built-in jax.grad, which may not account for the custom backward pass of the learned SiLU function. Therefore, the JAX code is not equivalent to the PyTorch code.

# So, the correct JAX code should implement the custom SiLU function with the backward pass as in the PyTorch code. However, in JAX, we can't directly implement a custom function with a backward pass like in PyTorch. Instead, we need to use the jax.custom_vjp or jax.custom_derivative functions to define the custom backward pass.

# Therefore, the correct JAX code should use jax.custom_vjp to define the custom SiLU function with the backward pass as in the PyTorch code.

# So, the correct JAX code would be:

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.lax import scan
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = jnp.array(np.random.rand(100, 1) * 10)  # 100 data points between 0 and 10
y = 2 * X + 3 + jnp.random.randn(100, 1)  # Linear relationship with noise

# Define the LearnedSiLU function with custom backward pass
def learned_silu_forward(x, slope):
    return slope * x * jnp.sigmoid(x)

def learned_silu_backward(g, x, slope):
    sigmoid_x = jnp.sigmoid(x)
    grad_input = g * slope * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
    grad_slope = g * x * sigmoid_x
    return grad_input, grad_slope

learned_silu = jax.custom_vjp(learned_silu_forward, learned_silu_backward)

# Define the model
def model(params, x):
    slope = params['slope']
    return learned_silu(x, slope)

# Define the loss function
def loss(params, x, y):
    return jnp.mean((model(params, x) - y) ** 2)

# Define the optimizer and training loop
def train_model():
    # Initial parameters
    initial_params = {
        'slope': jnp.array(1.0)
    }

    # Define the update step
    def update_step(params, x, y):
        # Compute the gradient
        grad_loss = grad(loss)(params, x, y)
        # Update parameters
        return params - grad_loss * 0.01  # Learning rate 0.01

    # Training loop
    for epoch in range(1000):
        # Forward pass
        current_loss = loss(initial_params, X, y)
        print(f"Epoch [{epoch + 1}/1000], Loss: {current_loss:.4f}")

        # Update parameters
        initial_params = update_step(initial_params, X, y)

    # Display the learned parameters
    print(f"Learned slope: {initial_params['slope']:.4f}")

train_model()

# But wait, in JAX, the custom_vjp requires that the forward function takes the arguments and returns the value, and the backward function takes the gradient and the arguments and returns the gradients. However, in the original PyTorch code, the backward function is called with the gradient and the saved tensors. In the JAX code, the backward function is called with the gradient and the arguments. So, the backward function in JAX needs to be implemented correctly.

# In the original PyTorch code, the backward function is called with the gradient and the saved tensors. In the JAX code, the backward function is called with the gradient and the arguments. So, the backward function in JAX needs to be implemented with the correct arguments.

# In the original PyTorch code, the backward function is:

def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    slope = ctx.slope
    sigmoid_x = torch.sigmoid(x)
    grad_input = grad_output * slope * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
    grad_slope = grad_output * x * sigmoid_x
    return grad_input, grad_slope

# In the JAX code, the backward function is:

def learned_silu_backward(g, x, slope):
    sigmoid_x = jnp.sigmoid(x)
    grad_input = g * slope * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
    grad_slope = g * x * sigmoid_x
    return grad_input, grad_slope

# But in the JAX code, the backward function is called with the gradient and the arguments. However, in the original PyTorch code, the backward function is called with the gradient and the saved tensors. In the JAX code, the backward function is called with the gradient and the arguments. So, the backward function in JAX needs to be implemented with the correct arguments.

# In the JAX code, the learned_silu_forward function is called with x and slope. The backward function is called with g (the gradient), x, and slope. So, the backward function is correct.

# But in the JAX code, the model function is:

def model(params, x):
    slope = params['slope']
    return learned_silu(x, slope)

# But in the JAX code, the learned_silu function is a custom_vjp function that takes x and slope as arguments. So, the model function is correct.

# Therefore, the JAX code should be as follows:

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = jnp.array(np.random.rand(100, 1) * 10)  # 100 data points between 0 and 10
y = 2 * X + 3 + jnp.random.randn(100, 1)  # Linear relationship with noise

# Define the LearnedSiLU function with custom backward pass
def learned_silu_forward(x, slope):
    return slope * x * jnp.sigmoid(x)

def learned_silu_backward(g, x, slope):
    sigmoid_x = jnp.sigmoid(x)
    grad_input = g * slope * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
    grad_slope = g * x * sigmoid_x
    return grad_input, grad_slope

learned_silu = jax.custom_vjp(learned_silu_forward, learned_silu_backward)

# Define the model
def model(params, x):
    slope = params['slope']
    return learned_silu(x, slope)

# Define the loss function
def loss(params, x, y):
    return jnp.mean((model(params, x) - y) ** 2)

def train_model():
    # Initial parameters
    initial_params = {
        'slope': jnp.array(1.0)
    }

    # Define the update step
    def update_step(params, x, y):
        # Compute the gradient
        grad_loss = grad(loss)(params, x, y)
        # Update parameters
        return params - grad_loss * 0.01  # Learning rate 0.01

    # Training loop
    for epoch in range(1000):
        # Forward pass
        current_loss = loss(initial_params, X, y)
        print(f"Epoch [{epoch + 1}/1000], Loss: {current_loss:.4f}")

        # Update parameters
        initial_params = update_step(initial_params, X, y)

    # Display the learned parameters
    print(f"Learned slope: {initial_params['slope']:.4f}")

train_model()

