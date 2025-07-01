import pytest
import jax.numpy as jnp
from jax import grad
from jax.experimental.run_local_silver import setup_local_silver
from jax.infer import loc_scale_reparam
from jax import poutine
from jax import trace
from jax import vmap
from tests.common import assert_close

def get_moments(x):
    m1 = jnp.mean(x, axis=0)
    x_shifted = x - m1
    xx = jnp.square(x_shifted)
    xxx = jnp.dot(x_shifted, xx)
    xxxx = jnp.square(xx)
    m2 = jnp.mean(xx, axis=0)
    m3 = jnp.mean(xxx, axis=0) / (m2 ** 1.5)
    m4 = jnp.mean(xxxx, axis=0) / (m2 ** 2)
    return jnp.stack([m1, m2, m3, m4])

@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, jnp.tensor(0.4), None])
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT", "AsymmetricLaplace"])
def test_moments(dist_type, centered, shape):
    loc = jnp.zeros(shape).random_uniform(-1.0, 1.0)
    scale = jnp.ones(shape).random_uniform(0.5, 1.5)

    def model():
        with vmap("plates", shape):
            with vmap("particles", shape):
                if dist_type == "Normal":
                    jax.random.normal(loc, scale, shape)
                elif dist_type == "StudentT":
                    jax.random.t-distribution(10.0, loc, scale, shape)
                else:
                    jax.random.asymmetric_laplace(loc, scale, 1.5, shape)

    trace_model = trace(model)
    value = trace_model["x"]["value"].eval()
    expected_probe = get_moments(value)

    reparam = loc_scale_reparam(centered)
    reparam_model = poutine.reparam(trace_model, {"x": reparam})
    actual_probe = reparam_model["x"]["value"].eval()

    if not is_identically_one(centered):
        params = []
        if dist_type == "Normal":
            assert params == ()
        elif dist_type == "StudentT":
            params = ("df",,)
        else:
            params = ("asymmetry",,)

        for param in params:
            if param == "df":
                assert reparam.params.shape == (1,)
                assert reparam.params.value[0] == 10.0
            elif param == "asymmetry":
                assert reparam.params.shape == (1,)
                assert reparam.params.value[0] == 1.5

    assert_close(actual_probe, expected_probe, atol=0.1, rtol=0.05)

    for actual_m, expected_m in zip(actual_probe, expected_probe):
        expected_grads = grad(expected_m.sum(), [loc, scale])
        actual_grads = grad(actual_m.sum(), [loc, scale])
        assert_close(actual_grads[0], expected_grads[0], atol=0.1, rtol=0.05)
        assert_close(actual_grads[1], expected_grads[1], atol=0.1, rtol=0.05)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, jnp.tensor(0.4), None])
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT", "AsymmetricLaplace"])
def test_init(dist_type, centered, shape):
    loc = jnp.zeros(shape).random_uniform(-1.0, 1.0)
    scale = jnp.ones(shape).random_uniform(0.5, 1.5)

    def model():
        with vmap("plates", shape):
            if dist_type == "Normal":
                jax.random.normal(loc, scale, shape)
            elif dist_type == "StudentT":
                jax.random.t_distribution(10.0, loc, scale, shape)
            else:
                jax.random.asymmetric_laplace(loc, scale, 1.5, shape)

    check_init_reparam(model, loc_scale_reparam(centered))