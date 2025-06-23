import io
import itertools
import pickle
import tempfile
import unittest
from pathlib import Path
import jax.numpy as jnp
from jax import vmap
from nupic.research.frameworks.pytorch import model_utils, restore_utils
from nupic.torch.models import MNISTSPCNN
from nupic.torch.modules import KWinners

# -------------------------------------------------------------------
# Utility Functions (JAX)
# -------------------------------------------------------------------
def lower_forward(model, x):
    y = model.cnn1(vmap(x))
    y = model.cnn1_maxpool(vmap(y))
    y = model.cnn1_kwinner(vmap(y))
    y = model.cnn2(vmap(y))
    y = model.cnn2_maxpool(vmap(y))
    y = model.cnn2_kwinner(vmap(y))
    y = vmap(y).flatten()
    return y

def upper_forward(model, x):
    y = model.linear(vmap(x))
    y = model.linear_kwinner(vmap(y))
    y = model.output(vmap(y))
    y = vmap(jnp.softmax(y))
    return y

def full_forward(model, x):
    y = lower_forward(model, x)
    y = upper_forward(model, y)
    return y

# -------------------------------------------------------------------
# Test Class (JAX)
# -------------------------------------------------------------------
class RestoreUtilsTest1(jax.test.Module):

    def setUp(self):
        # Set global random seed for deterministic testing
        jax.random.set_seed(20)
        self.model = MNISTSPCNN()
        self.model.init_hyperparams()
        self.model.eval()

        # Create temporary results directory.
        self.tempdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tempdir.name) / Path("results")
        self.results_dir.mkdir()

        # Save model state.
        buffer = io.BytesIO()
        state = {}
        # Note: JAX's serialize_state_dict is different from PyTorch's
        #        here we use jax.serialize to save model to a buffer
        jax.serialize(self.model, buffer)
        state["model"] = buffer.getvalue()

        self.checkpoint_path = self.results_dir / Path("mymodel")
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(state, f)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_get_param_names(self):
        linear_params = get_linear_param_names(self.model)
        # Manually verify expected linear parameters
        #...

    def test_load_full(self):
        # Reset random seed for this test
        jax.random.set_seed(33)
        model = MNISTSPCNN()
        model.eval()

        # Check initial parameters differ
        #...

        # Restore full model
        jax.io.write_text(self.checkpoint_path, contents=pickle.dumps(
            {"model": jax.serialize_state_dict(self.model)},
            f"{self.checkpoint_path}"
        ))
        restored_model = jax.vmap(
            jax.nn.load_jax_model(self.checkpoint_path)
        )
        restored_model.eval()

        # Verify all parameters restored
        #...

        # Verify outputs through all forward passes
        #...

    # [More tests for linear and nonlinear params...]

    def test_param_mapping(self):
        # Test permutation of parameters
        #...

# -------------------------------------------------------------------
# Other Test Classes (JAX compatible)
# -------------------------------------------------------------------
# Example: RestoreUtilsTest2 (adapted from PyTorch test)
class RestoreUtilsTest2(jax.test.Module):

    def setUp(self):
        jax.random.set_seed(20)
        self.model = jax.nn.Sequential(
            jax.nn.Linear(8, 8),
            KWinners(8, percent_on=0.1),
        )
        self.model.init_hyperparams()
        self.model.eval()

        # Save model state
        buffer = io.BytesIO()
        jax.serialize(self.model, buffer)
        self.checkpoint_path = Path("results/tempdir/results/mymodel")
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump({"model": jax.serialize_state_dict(self.model)}, f)

    def test_load_with_permutation(self):
        restored_model = jax.vmap(
            jax.nn.load_jax_model(self.checkpoint_path),
        )
        # Apply permutation mapping
        #...

        # Verify parameters are correctly mapped
        #...

# -------------------------------------------------------------------
# Note: Some functions adapted from PyTorch counterparts,
#      such as jax.vmap for vectorization and jax.serialize for saving.
# -------------------------------------------------------------------