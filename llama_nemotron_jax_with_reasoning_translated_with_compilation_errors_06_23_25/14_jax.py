import jax.numpy as jnp
from jax import vmap
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLism
import jax.nn as nn

class TestStackedBidirectionalLstm(JAX.testing.AutoTest):
    @pytest.fixture
    def stacked_lstm(self):
        return StackedBidirectionalLSTM(
            input_size=10, hidden_size=11, num_layers=3,
            recurrent_dropout_prob=jnp.random.uniform(0.0, 1.0),
            layer_dropout_prob=jnp.random.uniform(0.0, 1.0),
        )

    def test_dropout_parameters(self):
        initial_state = jnp.random.random((3, 5, 11))
        initial_memory = jnp.random.random((3, 5, 11))
        tensor = jnp.random.uniform(0, 1, (5, 7, 10))
        sequence_lengths = jnp.array([7]*5, dtype=jnp.int32)
        
        sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(tensor, sequence_lengths)
        lstm_input = pack_padded_sequence(sorted_tensor, sorted_sequence, batch_first=True)
        
        dropped_output, dropped_state = stacked_lstm()(lstm_input, (initial_state, initial_memory))
        dropped_output_sequence, _ = pad_packed_sequence(dropped_output, batch_first=True)
        
        # Assert layer dropout probabilities
        assert jnp.all(jnp.random.uniform(0.0, 1.0) == stacked_lstm().layer_dropout_prob)
        
        # Assert recurrent dropout probabilities
        assert jnp.all(jnp.random.uniform(0.0, 1.0) == stacked_lstm().recurrent_dropout_prob)

    # Other test functions adapted similarly...