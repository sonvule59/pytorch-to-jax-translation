2.0+
import jax
from jax import numpy as jnp
from jax import random
import jax.vjp as vjp
from jax import compactor
from jax import geom_input
from jax.experimental.build_lib import custom_jax_fns
import jax.shax
import jax.numpy as jnp
import jax.device_put
import jax.core.transformed_function_toolbox import FunctionToolbox
from jax import jit
from jax import device
import jax.vjp as vjp
from jax import vmap
from jax import matmul
from jax import addjvp
from jax import compactor
from jax import geom_input
from jax import vjp as jax_vjp
from jax import custom_jax_fns
import jax

# PyTorch to JAX conversion

# Import necessary JAX modules
import jax
from jax import numpy as jnp
from jax import random
from jax import vjp as jax_vjp
from jax import device
from jax import compactor
from jax import geom_input
from jax import shax
import jax.core as jax_core
from jax import Value
from jax import functional as jax_fn
from jax import matmul
from jax import addjvp
from jax import compactor
from jax import vmap
from jax import custom_jax_fns
from jax import jit
from jax import device
from jax import vjp as jax_vjp
from jax import shax
import jax.core.transformed_function_toolbox as tftf

# Define PyTorch model classes in JAX-compatible way

# _netG JAX equivalent
def _netG_jax(name):
    return jax.vjp(compactor(_netG_jax_fn)(name, ngpu=ngpu))

# _netD JAX equivalent
def _netD_jax(name):
    return jax.vjp(compactor(_netD_jax_fn)(name, ngpu=ngpu))

# Define JAX functions for forward pass

# Data Loading (JAX equivalent)
# Note: JAX doesn't have direct equivalents to DataLoader, so use jax.data.Dataset

# Assuming MNIST data is loaded in a JAX-friendly format

# Initialize models in GPU
G_jax = _netG_jax('G')
D_jax = _netD_jax('D')
device_jax = device()

# Set device
G_jax.set_device(device_jax)
D_jax.set_device(device_jax)

# Optimizer (JAX equivalent)
# JAX doesn't have optimizers like PyTorch; use jax.optim.legacy
from jax.optim import legacy as jax_opt
G_solver_jax = jax_opt.legacy.Adam(G_jax.parameters(), lr=lr)
D_solver_jax = jax_opt.legacy.Adam(D_jax.parameters(), lr=lr)

# Training loop adaptation
# Note: JAX autograd doesn't handle variables like PyTorch's Variable

# Sample noise generation (JAX equivalent)
# noise = random.normal(jnp.random.device(device_jax), size=(mb_size, nz, 1, 1))
# noise.resize_(mb_size, nz, 1, 1)

# Forward pass and training steps adapted with JAX autograd

# Note: JAX autograd uses just-in-time compilation, so functions must be defined with jax.jit
# Define forward functions with JIT compilation

# Example of forward pass in JAX
def G_forward_jax(input_jax, ngpu=ngpu):
    return jax.vjp(compactor(_netG_jax_fn)(G_jax, input_jax, ngpu=ngpu))

def D_forward_jax(x_jax):
    return jax.vjp(compactor(_netD_jax_fn)(D_jax, x_jax))

# Training loop would involve:
# 1. Loading data (using JAX Dataset)
# 2. For each batch:
#    a. Generate noise
#    b. Forward through G to get G_sample
#    c. Compute D(X) and D(G_sample)
#    d. Compute losses and backprop

# Since JAX autograd doesn't support PyTorch's Variable, handle batches manually

# Example code structure (adapted)
# data = jax.data.Dataset(...) # MNIST dataset in JAX format
# G_jax.set_device(device)
# D_jax.set_device(device)

# for data_batch in data.take(mb_size):
#     X_jax = data_batch[0].to(device)
#     Z_jax = jnp.random.normal(device, (mb_size, nz, 1, 1))
#     G_sample_jax