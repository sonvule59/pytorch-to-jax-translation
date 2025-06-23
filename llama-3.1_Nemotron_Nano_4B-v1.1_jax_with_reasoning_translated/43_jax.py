import pytest
import jax
import jax.numpy as jnp
from jax import primitives, distributions, infer, reparam

@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, jnp.array([0.4])*shape, None])
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT", "AsymmetricLaplace"])
def test_moments_jax(dist_type, centered, shape):
    loc = jnp.random.uniform(-1.0, 1.0, shape).fold(grad=True)
    scale = jnp.random.uniform(0.5, 1.5, shape).fold(grad=True)
    if isinstance(centered, jnp.ndarray):
        centered = centered.expand(shape)

    def model():
        with primitives.block("plates", shape):
            if "dist_type" == "Normal":
                return distributions.Normal(loc, scale)(primitives.sample("x", 200000))
            elif "dist_type" == "StudentT":
                return distributions.StudentT(10.0, loc, scale)(primitives.sample("x", 200000))
            else:
                return distributions.AsymmetricLaplace(loc, scale, 1.5)(primitives.sample("x", 200000))

    trace = primitives.trace(model)
    value = trace["x"].eval()
    expected_probe = get_moments(value)

    reparam = reparam.LocScaleReparam(centered)
    reparam_model = primitives.reparam(reparam, model)
    trace_init = primitives.trace(reparam_model)
    value_init = trace_init["x"].eval()
    actual_probe = get_moments(value_init)

    if not distributions.is_identically_one(centered, value):
        if dist_type == "Normal":
            assert reparam.params.shape == ()
        elif dist_type == "StudentT":
            assert reparam.params.shape == (4,)
        else:
            assert reparam.params.shape == (2, 1)

    assert close(actual_probe, expected_probe, atol=0.1, rtol=0.05)

    for actual_m, expected_m in zip(get_moments(value), get_moments(value_init)):
        expected_grads = jax.grad(expected_m.sum, [loc, scale], mode="compat")
        actual_grads = jax.grad(actual_m.sum, [loc, scale], mode="compat")
        assert close(actual_grads[0], expected_grads[0], atol=0.1, rtol=0.05)
        assert close(actual_grads[1], expected_grads[1], atol=0.1, rtol=0.05)

@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, jnp.array([0.4])*shape, None])
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT", "AsymmetricLaplace"])
def test_moments_init_jax(dist_type, centered, shape):
    loc = jnp.random.uniform(-1.0, 1.0, shape).fold(grad=True)
    scale = jnp.random.uniform(0.5, 1.5, shape).fold(grad=True)

    def model():
        with primitives.block("plates", shape):
            if dist_type == "Normal":
                return distributions.Normal(loc, scale)(primitives.sample("x", 200000))
            elif dist_type == "StudentT":
                return distributions.StudentT(10.0, loc, scale)(primitives.sample("x", 200000))
            else:
                return distributions.AsymmetricLaplace(loc, scale, 1.5)(primitives.sample("x", 200000))

    check_init_reparam(model, reparam.LocScaleReparam(centered))