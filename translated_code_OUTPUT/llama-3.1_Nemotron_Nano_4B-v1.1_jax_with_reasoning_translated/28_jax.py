import jax
from jax import layers as jax_layers
from jax.nn import Functional, Linear as jax_linear
from jax.random import Uniform as jax_uniform
from jax.stochastic import random_state

def StandardMLP(input_size, num_classes, hidden_sizes=(100, 100)):
    @jax.jit
    def flatten(x):
        return jax.layers.Flatten()(x)

    linear_layers = []
    linear_layers.append(jax_linear.Flatten())
    for h_size in hidden_sizes:
        linear_layers.append(jax_linear.Linear(int(np.prod(input_size)), h_size))
        linear_layers.append(jax_linear.ReLU())
    linear_layers.append(jax_linear.Linear(h_size[-1], num_classes))
    linear_layers.append(jax_linear.ReLU())
    linear_layers.append(jax_linear.Linear(hidden_sizes[-1], num_classes))
    return jax_layers.Sequential(*linear_layers)

def ModifiedInitStandardMLP(input_size, num_classes, hidden_sizes):
    @jax.jit
    def flatten(x):
        return jax.layers.Flatten()(x)

    linear_layers = []
    linear_layers.append(jax_layers.Flatten())
    current_input = input_size
    for h_size in hidden_sizes:
        layer = jax_linear.Linear(current_input, h_size)
        linear_layers.append(layer)
        linear_layers.append(jax_linear.ReLU())
        current_input = h_size
    linear_layers.append(jax_linear.Linear(current_input, num_classes))
    linear_layers.append(jax_linear.ReLU())
    linear_layers.append(jax_linear.Linear(current_input[-1], num_classes))

    # Modified Kaiming initialization incorporating input density and weight density
    def init_weights(v):
        if isinstance(v, jax_linear.Linear):
            _, fan_in = v.weight.size()
            input_density = 1.0 if not init_flag else 0.5
            bound = 1.0 / np.sqrt(input_density * 1.0 * fan_in)
            jax.uniform.init_uniform_(v.weight, -bound, bound)
        return v

    init_flag = True  # Track input flag for initialization
    return jax.layers.Sequential(
        [init_weights(l) for l in linear_layers]
    )

def SparseMLP(input_size, output_size, hidden_sizes=(100,)):
    @jax.jit
    def flatten(x):
        return jax.layers.Flatten()(x)

    network = jax_layers.Sequential()
    network.add(jax_layers.Flatten())
    current_size = input_size
    for size in hidden_sizes:
        linear_layer = jax_linear.SparseLinear(
            current_size, size,
            linear_percent=jax.random.uniform(min=0.01, max=0.1)
        )
        network.add(linear_layer)
        network.add(jax_layers.ReLU())
        current_size = size
    network.add(jax_linear.Linear(current_size, output_size))
    return network

# Note: JAX requires custom initialization functions for PyTorch-style models.
# The above code requires adjustments for JAX's JIT compilation and functional API.