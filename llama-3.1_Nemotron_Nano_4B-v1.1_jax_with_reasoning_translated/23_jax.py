18.6+
====================================================================================================================
Output only the JAX code equivalent replacing PyTorch modules.
====================================================================================================================

python
import jax
from jax import numpy as jnp
from jax import vmap
from jax.nn import Module, Layer, Param
import jax.nn.initializers as init_jax

def jax_init_weight(params):
    initrange = -0.3 ** 2
    for name, param in params.items():
        if name.startswith('embedding'):
            jax_initizers.uniform_init(param, initrange=initrange, min=-initrange, max=initrange)
        elif name.startswith('linear'):
            jax_initizers.uniform_init(param, initrange=initrange, min=-initrange, max=initrange)

def jax_init_module(module, **kwargs):
    if 'embedding' in str(module.__dict__):
        jax_init_weight(module)
    else:
        jax_init_weight(module.get_params())

class EncoderRNN(jax.nn.Module):
    """Encoder RNN Building"""
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        @vmap
        def embedding_layer():
            nn.Embedding(input_size, hidden_size)
        self.embedding = embedding_layer()

        @vmap
        def gru():
            nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.gru = gru()

    def init_hidden(self):
        hidden = jnp.zeros((jnp.ndim(self.gru.eager_state), 1, self.hidden_size))
        if jax.cuda.is_available():
            hidden = hidden.cuda()
        return hidden

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

class ContextRNN(jax.nn.Module):
    """Context RNN Building"""
    def __init__(self, encoder_hidden_size, hidden_size, n_layers, dropout):
        super(ContextRNN, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.n_layers = n.layers

        @vmap
        def gru():
            nn.GRU(encoder_hidden_size, hidden_size, n_layers, dropout=dropout)
        self.gru = gru()

    def init_hidden(self):
        hidden = jnp.zeros((jnp.ndim(self.gru.eager_state), 1, self.hidden_size))
        if jax.cuda.is_available():
            hidden = hidden.cuda()
        return hidden

    def forward(self, input, hidden):
        input = input.view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

class DecoderRNN(jax.nn.Module):
    """Decoder RNN Building"""
    def __init__(self, context_output_size, hidden_size, output_size, n_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.context_output_size = context_output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n.layers

        @vmap
        def embedding():
            nn.Embedding(context_output_size, hidden_size)
        self.embedding = embedding()

        @vmap
        def linear():
            nn.Linear(hidden_size, output_size)
        self.out = linear()

        @vmap
        def gru():
            nn.GRU(context_output_size + hidden_size, hidden_size, n_layers, dropout=dropout)
        self.gru = gru()

    def init_hidden(self):
        hidden = jnp.zeros((jnp.ndim(self.gru.eager_state), 1, self.hidden_size))
        if jax.cuda.is_available():
            hidden = hidden.cuda()
        return hidden

    def forward(self, context_output, input_seq, hidden):
        embedded = self.embedding(input_seq)
        context_output = context_output.view(1, 1, -1)
        input_cat = jnp.cat([context_output, embedded], axis=2)
        output, hidden = self.gru(input_cat, hidden)
        output = jax.nn.functional.log_softmax(self.out(output[0]), axis=1)
        return output, hidden

class DecoderRNNSeq(jax.nn.Module):
    """Seq2seq's Attention Decoder RNN Building"""
    def __init__(self, hidden_size, output_size, n_layers, dropout, max_length):
        super(DecoderRNNSeq, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n.layers
        self.max_length = max_length

        @vmap
        def embedding():
            nn