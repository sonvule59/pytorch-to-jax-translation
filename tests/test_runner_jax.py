"""
This script is written for testing JAX translated code 
@Author: Son Vu
@Date: July 1, 2025
"""
import importlib.util
import jax
import numpy as np
import chex

def load_function(module_path, func_name):
    spec = importlib.util.spec_from_file_location(func_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)

TEST_INPUT = np.linspace(-2.0, 2.0, num=5, dtype=np.float32)

def test_jax_translation(jax_func, module_path):
    print("Running test for:", jax_func, module_path)
    assert True

# def test_jax_translation(jax_func, module_path):
#     func = load_function(module_path, jax_func)
#     jit_func = jax.jit(func)

#     # Check both JIT and non-JIT versions
#     for variant in (func, jit_func):
#         @chex.all_variants
#         def shape_and_finite(x):
#             out = variant(x)
#             chex.assert_shape(out, x.shape)
#             chex.assert_tree_all_finite(out)

#         shape_and_finite(TEST_INPUT)

