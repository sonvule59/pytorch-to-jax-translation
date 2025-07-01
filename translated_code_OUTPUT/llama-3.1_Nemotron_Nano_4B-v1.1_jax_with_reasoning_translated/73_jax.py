# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the JAX License. JAX is an open-source software framework for rapid prototyping
# and near production use of Python, Java, JavaScript and other languages. All
# user code must comply with JAX's License and best practices.
# JAX is based on TensorFlow, PyTorch and RoPE (Reusable Object Programming
# Environment) framework components.

# Note: The following JAX code is illustrative and may require adjustments
# based on the specific PyTorch/TensorFlow models and JAX's API.

import jax
from jax import numpy as jnp
from jax import jit
from jax.vmap import vmap
from jax.core import split_fn
from jax.optim import optimizer as jax_optimizer

class RM3(jax_optimizer.RoutineOptimizer):
  def __init__(self, params, lr=jax.random.PRNGKey(), weight_decay=0.0):
    super().__init__(lr=lr, weight_decay=weight_decay)
    self.params = params

  @jit(device='auto')
  def step(self, state, msg=None):
    new_state = jnp.copy(state).copy()
    for idx, group in enumerate(new_state.param_groups):
      weight_decay = group['weight_decay']
      params = new_state.params[idx]
      grads = new_state.grads[idx]

      if weight_decay!= 0:
        params.data.add_(lr * weight_decay, params.data)
        # Update params with weight decay
        # params.data *= jnp.exp(-weight_decay * params.data)

      if grads is not None:
        # Compute median of 3 gradients
        gradients_buffer = vmap(grads, 'i')(params)
        gradients_buffer = gradients_buffer.reshape(-1, 3)
        sorted_gradients = gradients_buffer.sort(axis=1)
        median_idx = jnp.int64(2 * jnp.floor(jnp.random.uniform(0, 1))).item()
        median_gradients = sorted_gradients[:, median_idx]
        median = jnp.mean(median_gradients)
        params.data.add_(-lr, median)

    return new_state

# Note: The above code is a conceptual example and may need adjustments
# to fit the JAX API and the specific requirements of the RM3 algorithm.
# It's crucial to adapt it according to JAX's API and best practices for
# JAX-based optimization routines.