import os
import pytest

@pytest.fixture(scope="session")
def translated_folder():
    # Path to the folder containing all your translated JAX files
    root_dir = os.path.dirname(__file__)  # tests/
    return os.path.join(root_dir, "/translated_code_OUTPUT/llama-3.1_Nemotron_Nano_4B-v1.1_jax_with_reasoning_translated/")

def pytest_generate_tests(metafunc):
    print("pytest_generate_tests called:", metafunc.fixturenames)
    if "jax_func" in metafunc.fixturenames and "module_path" in metafunc.fixturenames:
        folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../translated_code_OUTPUT/llama-3.1_Nemotron_Nano_4B-v1.1_jax_with_reasoning_translated"))
        print("Looking in:", folder)
        py_files = [f for f in os.listdir(folder) if f.endswith(".py")]
        print("Files found:", py_files)
        params = [
            (os.path.splitext(fname)[0], os.path.join(folder, fname))
            for fname in py_files
        ]
        metafunc.parametrize(("jax_func", "module_path"), params)
