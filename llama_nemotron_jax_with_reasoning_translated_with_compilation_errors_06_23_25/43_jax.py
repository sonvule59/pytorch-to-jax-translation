2.24.0

Now translate this Python/Torch code to JAX code while maintaining functionality. Assume the test parametrization is exhaustive, but ensure the JAX code compiles and runs without errors.

<answer>
python
import pytest
import jax
import jax.numpy as jnp
from jax import grad, random
from jax.experimental.build_pipeline import build_pipeline
from jax import compactor
from jax import vmap
from jax.distributions import Normal, StudentT, AsymmetricLaplace

# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

def get_moments(x):
    m1 = jnp.mean(x, axis=())
    x_shifted = x - m1
    xx = jnp.outer(x_shifted, x_shifted)
    xxx = jnp.sum(jnp.outer(x_shifted, xx), axis=())
    xxxx = jnp.sum(xx * xx, axis=())
    m2 = jnp.mean(xx)
    m3 = jnp.mean(xxx) / (m2 ** 1.5)
    m4 = jnp.mean(xxx * xxxx) / (m2 ** 2)
    return jnp.stack([m1, m2, m3, m4])

@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, torch.tensor(0.4), None], types={None: None})
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT", "AsymmetricLaplace"])
def test_moments_jax(dist_type, centered, shape):
    loc = jnp.random.uniform(-1.0, 1.0, shape).fold()
    scale = jnp.random.uniform(0.5, 1.5, shape).fold()

    if isinstance(centered, (int, float)):
        centered = centered.reshape(1 + shape, 1)
    elif centered is None:
        centered = jnp.ones(shape).fold()
    else:
        centered = centered.reshape(1 + shape, 1)

    def model():
        with build_pipeline(
            compactor.Pipe,
            plates=pyro.plate_stack("plates", shape),
            particles=200000,
        ):
            if dist_type == "Normal":
                x = pyro.sample("x", Normal(loc, scale))
            elif dist_type == "StudentT":
                x = pyro.sample("x", StudentT(10.0, loc, scale))
            else:
                x = pyro.sample("x", AsymmetricLaplace(loc, scale, 1.5))

        return x

    reparam = LocScaleReparam(centered)
    reparam_model = build_pipeline(
        compactor.Pipe,
        plates=reparam.transform(pyro.plate_stack("plates", shape)),
        particles=200000,
    )
    x = poutine.trace(reparam_model).get_trace().nodes["x"]["value"]

    expected_probe = get_moments(x)
    actual_probe = get_moments(x)

    if not is_identically_one(centered):
        if dist_type == "Normal":
            assert reparam.transformed_shape == ()
        elif dist_type == "StudentT":
            assert reparam.transformed_shape == ("df",)
        else:
            assert reparam.transformed_shape == ("asymmetry",)

    pytest.assert_close(actual_probe, expected_probe, atol=0.1, rtol=0.05)

    x_sum = get_moments(x).sum()
    x_shifted_sum = x.sum() - x_sum
    assert x_shifted_sum.sum() == 0.0

    for param, grad_fn in zip([loc, scale], [grad(x_sum), grad(x_shifted_sum)]):
        grad_fn.apply_grads([grad_fn], keep_default=True)
        break

@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, torch.tensor(0.4), None], types={None: None})
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT", "AsymmetricLaplace"])
def test_init_jax(dist_type, centered, shape):
    loc = jnp.empty(shape).uniform(-1.0, 1.0)
    scale = jnp.empty(shape).uniform_(0.5, 1.5)

    def model():
        with build_pipeline(
            compactor.Pipe,
            plates=pyro.plate_stack("plates", shape),
        ):
            if dist_type == "Normal":
                return pyro.sample("x", Normal(loc, scale))