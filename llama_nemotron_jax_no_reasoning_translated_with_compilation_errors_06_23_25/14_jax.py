20.10.0

Replace torch with torch.jax, input_tensor with jax.numpy.array, and pack_padded_sequence with jax.nn.pack_padded_sequence and pad_packed_sequence accordingly. Also, ensure that the initializers and applicators are compatible with JAX.

Here is the converted code:

jax
import jax.numpy as jnp
import pytest
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.common.params import Params
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.nn.util import sort_batch_by_length

def test_stacked_bidirectional_lstm_completes_forward_pass():
    input_tensor = jnp.random.rand(4, 5, 3)
    input_tensor[1, 4:, :] = 0.0
    input_tensor[2, 2:, :] = 0.0
    input_tensor[3, 1:, :] = 0.0
    batch_first = True
    input_padded = jax.nn.pack_padded_sequence(input_tensor, [5, 4, 2, 1], batch_first=batch_first)
    lstm = StackedBidirectionalLstm(input_size=5, hidden_size=7, num_layers=3)
    output, _ = lstm(input_padded, jax.random.PRNGKey(0))
    output_padded, _ = jax.nn.pad_packed_sequence(output, batch_first=batch_first)
    output_sequence = output_padded.data[:, 1, :]
    numpy.testing.assert_array_equal(
        output_sequence[1, 4:, :].numpy(), jnp.zeros_like(output_sequence[1, 4:, :].dtype),
        decimal=12
    )
    numpy.testing.assert_array_equal(
        output_sequence[2, 2:, :].numpy(), jnp.zeros_like(output_sequence[2, 2:, :].dtype),
        decimal=12
    )
    numpy.testing.assert_array_equal(
        output_sequence[3, 1:, :].numpy(), jnp.zeros_like(output_sequence[3, 1:, :].dtype),
        decimal=12
    )

def test_stacked_bidirectional_lstm_can_build_from_params():
    params = Params({
        "type": "stacked_bidirectional_lstm",
        "input_size": 5,
        "hidden_size": 9,
        "num_layers": 3,
    })
    encoder = Seq2SeqEncoder.from_params(params)
    numpy.testing.assert_array_equal(encoder.get_input_dim(), jnp.array([5]))
    numpy.testing.assert_array_equal(encoder.get_output_dim(), jnp.array([18]))
    numpy.testing.assert_array_equal(encoder.is_bidirectional, jnp.bool_(True))

def test_stacked_bidirectional_lstm_can_build_from_params_seq2vec():
    params = Params({
        "type": "stacked_bidirectional_lstm",
        "input_size": 5,
        "hidden_size": 9,
        "num_layers": 3,
    })
    encoder = Seq2VecEncoder.from_params(params)
    numpy.testing.assert_array_equal(encoder.get_input_dim(), jnp.array([5]))
    numpy.testing.assert_array_equal(encoder.get_output_dim(), jnp.array([18]))

def test_stacked_bidirectional_lstm_can_build_from_params_seq2vec_encode_output_dim_correctly(self):
    params = Params({
        "type": "stacked_bidirectional_lstm",
        "input_size": 3,
        "hidden_size": 9,
        "num_layers": 3,
    })
    encoder = Seq2VecEncoder.from_params(params)
    numpy.testing.assert_array_equal(encoder.get_output_dim(), jnp.array([18]))

def test_stacked_bidirectional_lstm_can_complete_forward_pass_seq2vec():
    params = Params({
        "type": "stacked_bidirectional_lstm",
        "input_size": 3,
        "hidden_size": 9,
        "num_layers": 3,
    })
    encoder = Seq2SeqEncoder.from_params(params)
    input_tensor = jnp.random.rand(4, 5, 3)
    mask = jnp.ones((4, 5), dtype=jnp.bool_)
    output = encoder(input_tensor, mask=jnp.ones((4, 5)))
    numpy.testing.assert_array_equal(output.data[1, 1, :], jnp.zeros_like(output.data[1, 1, :]), decimal=12)

def test_stacked_bidirectional_lstm_dropout_version_is_different():
    stacked_lstm = StackedBidirectionalLstm(input_size=10, hidden_size=11, num_layers=3)
    dropout_name = "layer_dropout_probability"
    dropped_stacked_lstm = StackedBidirectionalLstm(
        input_size=10,
        hidden_size=11,
        num_layers=3