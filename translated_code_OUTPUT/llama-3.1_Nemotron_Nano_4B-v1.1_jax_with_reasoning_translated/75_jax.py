import jax
from jax import numpy as jnp
from jax.sharding import shard_one_dimension
from jax.experimental import install_jax_pytorch

#... same structure as original, but with JAX imports and functions