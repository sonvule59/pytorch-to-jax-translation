import torch
import jax.numpy as jnp
from jax import vmap, grad
from jax import jit
import jax.numpy as jnp
import jax.random as jrandom
import jax.data
import jax.shapes
import jax.core as jcore
import jax.experimental.deprecations
from jax import app
import jax.vmap
import jax.lax
import jax.numpy.functional as jfun

def main():
    # Example of JIT compilation
    @jit(nopython=True)
    def jax_func(x):
        return x + 1

    jfun.jax_addfun(jax_func, inputs=x, outputs=out)
    print(jax_func(jrandom.uniform(0, 1)))

    # Example of gradient computation
    x = jnp.array([1.0, 2.0])
    grad(x, var_name="x").print()

if __name__ == "__main__":
    main()