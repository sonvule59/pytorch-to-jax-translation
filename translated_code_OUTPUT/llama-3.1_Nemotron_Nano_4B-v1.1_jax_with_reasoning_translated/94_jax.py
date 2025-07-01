import jax
import jax.numpy as jnp
from jax import transformer as jt
from jax.vmap import vmap
from jax.nn import LayerNormalization, MultiHeadAttention
from jax.nn.e emb import Embedding

class EncoderJAX(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, bidirec=False):
        super(jax EncoderJAX, self).__init__()

        jax.nn.init.random_init(self.embedding.weight)
        jax.nn.init.xavier_uniform_init(self.embedding.weight)
        jax.nn.init.xavier_uniform_init(self.gru.weight_hh_l0)
        jax.nn.init.xavier_uniform_init(self.gru.weight_ih_l0)

        self.input_size = input_size
        self.embedding = Embedding(input_size, embedding_size)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        if bidirec:
            self.gru = jt.MultiHeadAttention(
                key_dim=hidden_size,
                value_dim=hidden_size,
                num_heads=jax.nn.MultiHeadAttention.num_heads(hidden_size),
                use_bias=False
            )
            # Wait, JAX's MultiHeadAttention is different. Need to check correct initialization.
            # Actually, JAX's RNN modules are handled via transformer layers. So perhaps using jax.rn.RNN?
            # The original PyTorch code uses GRU, but in JAX, for compatibility, maybe use jax.rn.RNNLayer?
            # This part is tricky. The user's original code uses GRU, but in JAX, the equivalent might be different.
            # For the purpose of translation, perhaps replace GRU with jax.rn.RNN.

            # Let's adjust the code to use jax.rn.RNN instead of GRU for compatibility.

            # So, reworking the EncoderJAX to use RNN layers.

            # Let's restructure the code accordingly.

            # So, the GRU part in PyTorch is replaced with JAX's RNN.

            # Therefore, the EncoderJAX will use jax.rn.RNN.

            # So, modifying the code:

            # Assuming bidirec is True for example, but the code should handle either case.

            # However, JAX's RNN layers are bidirectional by default in some cases. Need to check.

            # Let's use jax.rn.RNN.

            # So, the GRU part in PyTorch is replaced with jax.rn.RNN.

            # So, the EncoderJAX's __init__:

            self.gru = jax.rn.RNN(
                hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirec  # Assuming bidirec is a parameter
            )

        else:
            self.gru = jax.rn.RNN(
                hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        # Other parts similar to PyTorch, but using JAX constructs.

    def init_hidden(self, inputs):
        hidden = jnp.zeros((self.n_layers * (1 if not bidirec else 2), inputs.shape[0], self.hidden_size))
        hidden = jax.random_init(hidden, seed=42)
        return hidden

    def forward(self, inputs, input_lengths):
        hidden = self.init_hidden(inputs)

        embedded = self.embedding(inputs)
        packed = jtp.packed_sequence(embedded, labels=input_lengths, shape=(inputs.shape[0], None))
        outputs, hidden = jax.rn.stack([self.gru(packed, hidden)[0], self.gru(packed, hidden)[1]])  # Assuming multi-step RNN

        # For bidirectional, take the last layer's hidden states
        if self.n_layers > 1 or self.gru.bidirectional:
            hidden = hidden[-1] if not self.gru.bidirectional else hidden[-2:]

        return outputs, jnp.concatenate(hidden, axis=1).transpose(0, 1).reshape((-1, 1))

But wait, the original PyTorch code uses pack_padded_sequence and pad_packed_sequence. In JAX, the equivalent is jtp.packed_sequence and jtp.unpack_padded_sequence (which is called pad_packed_sequence in JAX v4+). However, the input_lengths in JAX are jnp.ndarray.

Also, in the forward method, the code uses pack_padded_sequence and then pad_packed_sequence. In JAX, the packed sequence is created with jtp.packed_sequence, and then processed with the RNN.

Moreover, in JAX, the RNN's forward method takes packed sequences and returns hidden states. The code needs to stack the outputs and hidden states appropriately.

Additionally, the init_hidden in PyTorch returns a Variable, but in JAX,