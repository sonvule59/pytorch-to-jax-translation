4.0


# PyTorch to JAX conversion
# Assuming the original code is using PyTorch for inference only (not training)
# Since JAX autograd runs on CPU by default, for GPU code, we'd need to use TAU
# But given the test functions are for prediction, we can directly cast

Converted Code (JAX 3.14.0):
import jax
import jax.numpy as jnp
from jax import vmap, grad
from jax.experimental.deprecation import disable_jax_deprecations  # Optional

@disable_jax_deprecations()
def test_infer_net1d(test_infer_net1d_jax, test_infer_net2d_jax, test_infer_net3d_jax):
    # Assuming test_infer_net1d_jax is a JAX-compatible forward pass
    test_infer_net1d_jax(jnp.zeros(1, 10, 3))
    test_infer_net2d_jax(jnp.zeros(1, 10, 10, 3))
    test_infer_net3d_jax(jnp.zeros(1, 10, 10, 10, 3))

def test_predict_batch(relu_net_jax):
    x = np.array([[-1., 1., 2., -2.])]
    expected = np.array([[0., 1., 2., 0.]])
    preds = relu_net_jax(x)
    assert np.all(preds == expected)

def test_validate_batch(relu_net_jax, metrics):
    x = np.array([[.2,.3,.5, -1], [-1,.6,.4, -1]])
    y = np.array([2, 2])
    (loss, acc), preds = relu_net_jax.validate_on_batch(x, y, **{m: vmap(m)(x, y) for m in metrics})
    expected_loss = -(math.log(x[0, y[0]]) + math.log(x[1, y[1]])) / 2
    assert jnp.isclose(loss, expected_loss, rel_tol=1e-7)
    assert jnp.isclose(acc, 50.)

def test_optimizer(relu_net_jax):
    optimizer = jax.optimizers.SGD(relu_net_jax.parameters(), lr=0.01)
    # Test adding an unnamed optimizer
    relu_net_jax.add_optimizer(optimizer)
    assert len(relu_net_jax.optimizer_manager.optimizers) == \
        len(relu_net_jax.optimizer_manager.names) == 1
    assert optimizer in relu_net_jax.optimizer_manager.optimizers
    assert "optimizer_0" in relu_net_jax.optimizer_manager.names

    # Test adding a named optimizer
    name = "sgd_optim"
    relu_net_jax.clear_optimizers()
    relu_net_jax.add_optimizer(optimizer, name=name)
    assert len(relu_net_jax.optimizer_manager.optimizers) == \
        len(relu_net_jax.optimizer_manager.names) == 1
    assert optimizer in relu_net_jax.optimizer_manager.optimizers
    assert name in relu_net_jax.optimizer_manager.names

    # Test multiple optimizers
    optimizer2 = jax.optimizers.SGD(relu_net_jax.parameters(), lr=0.02)
    relu_net_jax.clear_optimizers()
    relu_net_jax.add_optimizer(optimizer, name=name)
    relu_net_jax.add_optimizer(optimizer2)
    assert len(relu_net_jax.optimizer_manager.optimizers) == \
        len(relu_net_jax.optimizer_manager.names) == 2
    assert optimizer in relu_net_jax.optimizer_manager.optimizers
    assert optimizer2 in relu_net_jax.optimizer_manager.optimizers
    assert name in relu_net_jax.optimizer_manager.names
    assert "optimizer_1" in relu_net_jax.optimizer_manager.names

    # Test removing an optimizer
    optimizer_info = relu_net_jax.remove_optimizer()
    assert optimizer_info["name"] == "optimizer_1"
    assert optimizer_info["optimizer"] is optimizer2
    assert len(relu_net_jax.optimizer_manager.optimizers) == \
        len(relu_net_jax.optimizer_manager.names) == 1
    relu_net_jax.add_optimizer(optimizer)
    optimizer_info = relu_net_jax.remove_optimizer(name=name)
    assert optimizer_info["name"] == name
    assert optimizer_info["optimizer"] is optimizer
    assert len(relu_net_jax.optimizer_manager.optimizers) == \
        len(relu_net_jax.optimizer_manager.names) == 1
    relu_net_jax.clear_optimizers()

def test_loss(relu_net_jax, binary_loss_fn_jax, multi_binary_loss_fn_jax):
    x = jnp.array([[-1., 1., 0., 1.],
                  [-2., 0., 1.,