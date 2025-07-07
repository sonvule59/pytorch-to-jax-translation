# tests/test_jax_correctness.py
import os
import importlib.util
import numpy as np
import pytest
import torch

JAX_DIR = "/home/hungphd/Son_Google_team/small_LLM_models_pipeline/reasonfluxcoder_4b_jax_translated_code_v1"
PYTORCH_DIR = "/home/hungphd/Son_Google_team/INPUT"
TEST_INPUT = np.arange(10, dtype=np.float32)  # Example test input

def dynamic_import(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_all_py_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".py")]

@pytest.mark.parametrize("fname", get_all_py_files(JAX_DIR))
def test_jax_translation_correctness(fname):
    base = fname.replace("_jax.py", ".py")
    jax_path = os.path.join(JAX_DIR, fname)
    pytorch_path = os.path.join(PYTORCH_DIR, base)
    assert os.path.exists(jax_path), f"{jax_path} not found"
    assert os.path.exists(pytorch_path), f"{pytorch_path} not found"

    # Import and run PyTorch code
    pytorch_mod = dynamic_import(pytorch_path, f"pytorch_{base.replace('.py','')}")
    if not hasattr(pytorch_mod, "main"):
        pytest.skip(f"No main() in {base}")
    torch.manual_seed(0)
    expected = pytorch_mod.main(TEST_INPUT)

    # Import and run JAX code
    jax_mod = dynamic_import(jax_path, f"jax_{base.replace('.py','')}")
    if not hasattr(jax_mod, "main"):
        pytest.skip(f"No main() in {fname}")
    import jax
    import jax.numpy as jnp
    result = jax_mod.main(jnp.asarray(TEST_INPUT))

    # Check correctness
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)

