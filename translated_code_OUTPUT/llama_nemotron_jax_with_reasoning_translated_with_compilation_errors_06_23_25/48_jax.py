2.24.0
JAX-Mode: GPU
Device Type: CUDA

#! /usr/bin/env python

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.vjp import jaxvjp
from jax.experimental.build_lib import custom_jax_delegates
from jax import compache
import onnx
import onnx.transformers as onnx_tf
import numpy as np

def main():
    # Load ONNX model
    model = onnx.load("digits.onnx")
    print(model.graph.graph.out_edges)
    # Compile JAX version of the model
    model_jax = jax.vmap(jit)(model.graph).compile()
    # Export JAX model to ONNX again for compatibility
    jax_onnx = onnx.transformers.factory_onnx_tf.model_from_tensor(
        jnp.array(model.input_names[0]),
        exported_params=model.graph.params
    )
    onnx.checker.check_model(jax_onnx)
    # Train the model (example: simple loop)
    inputs = jnp.random.normal((1, 28, 28, 1), seed=42)
    outputs = model_jax(inputs)
    print(outputs.shape)

if __name__ == '__main__':
    jax.config.update('jaxlib.benchmark_mode', True)
    main()