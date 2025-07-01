3.23.0

Please output only the equivalent JAX code replacing the PyTorch code. Output nothing else.
<ththink>
python
import jax
import jax.numpy as jnp
from jax import PADDED_SEQUENCE, sort_batch_by_length
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.common.params import Params
from allennlp.nn import InitializerApplicator, Initializer

class TestStackedBidirectionalLstm:
    def test_stacked_bidirectional_lstm_completes_forward_pass(self):
        input_tensor = jnp.random.random(4, 5, 3)
        mask = jnp.ones(4, 5, dtype=jnp.bool_)
        packed = sort_batch_by_length(input_tensor, mask.arange(4, 5).tolist(), value_mask=mask)
        stacked_lstm = StackedBidirectionalLstm(3, 7, 3)
        output, _ = stacked_lstm(packed[0], (jnp.zeros([3, 5, 3]), jnp.zeros([3, 5, 3])))
        output_seq, _ = PADDED_SEQUENCE.output(output, packed[1])
        assert output_seq[1, 4:, :].array().equal(0.0)
        assert output_seq[2, 2:, :].array().equal(0.0)
        assert output_seq[3, 1:, :].array().equal(0.0)

    def test_stacked_bidirectional_lstm_can_build_from_params(self):
        params = Params({
            "type": "stacked_bidirectional_lstm",
            "input_size": 5,
            "hidden_size": 9,
            "num_layers": 3,
        })
        encoder = Seq2SeqEncoder.from_params(params)
        assert encoder.input_dim == 5
        assert encoder.output_dim == 18
        assert encoder.is_bidirectional

    def test_stacked_bidirectional_lstm_can_build_from_params_seq2vec(self):
        params = Params({
            "type": "stacked_bidirectional_lstm",
            "input_size": 5,
            "hidden_size": 9,
            "num_layers": 3,
        })
        encoder = Seq2VecEncoder.from_params(params)
        assert encoder.input_dim == 5
        assert encoder.output_dim == 18

    def test_stacked_bidirectional_lstm_can_complete_forward_pass_seq2vec(self):
        params = Params({
            "type": "stacked_bidirectional_lstm",
            "input_size": 3,
            "hidden_size": 9,
            "num_layers": 3,
        })
        encoder = Seq2VecEncoder.from_params(params)
        input_tensor = jnp.random.random(4, 5, 3)
        output = encoder(input_tensor, mask=jnp.ones(4, 5, dtype=jnp.bool_))
        assert output.shape == (4, 18)

    @pytest.mark.parametrize(
        "dropout_name", ("layer_dropout_probability", "recurrent_dropout_probability")
    )
    def test_stacked_bidirectional_lstm_dropout_version_is_different(self, dropout_name: str):
        stacked_lstm = StackedBidirectionalLstm(10, 11, 3)
        if dropout_name == "layer_dropout_probability":
            dropped_stacked_lstm = StackedBidirectionalLstm(
                input_size=10, hidden_size=11, num_layers=3, layer_dropout_probability=0.9
            )
        elif dropout_name == "recurrent_dropout_probability":
            dropped_stacked_lstm = StackedBidirectionalLstm(
                input_size=10, hidden_size=11, num_layers=3, recurrent_dropout_probability=0.9
            )
        else:
            raise ValueError("Invalid dropout name")
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 0.1}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(stacked_lstm)
        initializer(dropped_stacked_lstm)

        initial_state = jnp.zeros([3, 5, 11])
        initial_memory = jnp.zeros([3, 5, 11])
        tensor = jnp.random.random(5, 7, 10)
        sequence_lengths = jnp.LongTensor([7] * 5)
        mask = jnp.ones(4, 5, dtype=jnp.bool_)

        sorted_input = sort_batch_by_length(tensor, sequence_lengths, value_mask=mask)
        packed = pack_padded_sequence(sorted_input[0], sorted_input[1], batch_first=True)

        stacked_output, stacked_state = stacked_lstm(packed, (initial_state, initial_memory))
        dropped_output, dropped_state = dropped_stacked_lstm(packed