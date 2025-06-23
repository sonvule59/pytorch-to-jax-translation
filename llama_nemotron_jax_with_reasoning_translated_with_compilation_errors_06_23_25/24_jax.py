import jax
from jax import vjp
from jax import numpy as jnp
from jax.vjp import Vector

class TestMultiHeadedSimilarityFunction(vjp.test_module):
    @vjp.test
    def test_forward(self):
        similarity = vjp.MultiHeadedSimilarity(
            num_heads=3,
            tensor_1_dim=6,
        )
        similarity._tensor_1_projection = vjp.Parameter(jnp.eye(6))
        similarity._tensor_2_projection = vjp.Parameter(jnp.eye(6))
        a_vectors = Vector([[[[1.0, 1.0, -1.0, -1.0, 0.0, 1.0], [-2.0, 5.0, 9.0, -1.0, 3.0, 4.0]]]])
        b_vectors = Vector([[[[1.0, 1.0, 1.0, 0.0, 2.0, 5.0], [0.0, 1.0, -1.0, -7.0, 1.0, 2.0]]]]])
        result = similarity(a_vectors, b_vectors).data.numpy()
        assert result.shape == (1, 1, 2, 3)
        jax.array_equal(result, jnp.array([[[[2.0, -1.0, 5.0], [5.0, -2.0, 11.0]]]]"))

    def test_tensor_sizes(self):
        similarity = vjp.MultiHeadedSimilarity(num_heads=3, tensor_1_dim=9, tensor_1_projected_dim=6,
                                               tensor_2_dim=6, tensor_2_projected_dim=12)
        assert similarity._tensor_1_projection.size() == (9, 6)
        assert similarity._tensor_2_projection.size() == (6, 12)
        with jax.vjp.config.update({'mode':'static'}):
            with self.assertRaises(jax.vjp.ConfigurationError):
                vjp.MultiHeadedSimilarity(num_heads=3, tensor_1_dim=10)
        params = vjp.Params({'num_heads': 3, 'tensor_1_dim': 9, 'tensor_2_dim': 10})
        with self.assertRaises(jax.vjp.ConfigurationError):
            vjp.MultiHeadedSimilarity.from_params(params)