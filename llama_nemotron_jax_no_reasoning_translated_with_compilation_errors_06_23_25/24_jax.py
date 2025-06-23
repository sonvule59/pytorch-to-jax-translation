from jax import numpy as jnp
from jax.vjp import jit

def test_weights_are_correct_sizes():
    similarity = MultiHeadedSimilarity(num_heads=3, tensor_1_dim=9, tensor_1_projected_dim=6,
                                        tensor_2_dim=6, tensor_2_projected_dim=12)
    assert similarity.tensor_1_projection.shape == (9, 6)
    assert similarity.tensor_2_projection.shape == (6, 12)
    with pytest:
        def check_error(message):
            raise ConfigurationError(message)
        similarity = MultiHeadedSimilarity(num_heads=3, tensor_1_dim=10)
        check_error("Expected tensor_1_dim to be 6")

    def create_params():
        params = {
            'num_heads': 3,
            'tensor_1_dim': 9,
            'tensor_2_dim': 6
        }
        return params

    def test_from_params():
        params = create_params()
        similarity = MultiHeadedSimilarity.from_params(params)
        assert similarity.tensor_1_projection.shape == (9, 6)
        similarity.check_error("Invalid params")  # Should raise an error

def test_forward():
    similarity = MultiHeadedSimilarity(num_heads=3, tensor_1_dim=6)
    similarity._tensor_1_projection = jnp.ones((6,))
    similarity._tensor_2_projection = jnp.ones((6,))

    a_vectors = jnp.array([[[[1.0, 1.0, -1.0, -1.0, 0.0, 1.0], [-2.0, 5.0, 9.0, -1.0, 3.0, 4.0]]]])
    b_vectors = jnp.array([[[[1.0, 1.0, 1.0, 0.0, 2.0, 5.0], [0.0, 1.0, -1.0, -7.0, 1.0, 2.0]]]])
    result = similarity(a_vectors, b_vectors).data.numpy()
    assert result.shape == (1, 1, 2, 3)
    assert jnp.allclose(result, jnp.array([[[[2.0, -1.0, 5.0], [5.0, -2.0, 11.0]]]]))

jit(test_forward)