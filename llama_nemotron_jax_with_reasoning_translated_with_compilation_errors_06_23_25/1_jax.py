8.0+

import jax.numpy as jnp
from jax import concat, reshape, sum, slice
from jax.scipy.linalg import expm

class ConvRec(jax.macro.variation_model_base.BaseMacroVar):
    """
    JAX version of the ConvRec model from PyTorch.
    """

    def __init__(self, name, **kwargs):
        super(ConvRec, self).__init__(name, **kwargs)
        # Embeddings
        self.embeddings = jax.vmap(jax.nn.Linear(self.kwargs['vocab'], 
                                                  self.kwargs['emb_dim']))

        # CNN
        self.conv_layers = []
        self.kernel_sizes = self.kwargs['kernel_sizes']
        self.out_channels = self.kwargs['out_channels']

        for layer, (H, W) in enumerate(self.kernel_sizes):
            if layer == 0:
                C_i, _ = 1, self.emb_dim
            else:
                C_i, _ = self.out_channels, 1

            # Compute padding based on kernel size
            padding = jnp.floor(W / 2) * 2 if self.kwargs['conv_type'] == 'wide' else 0

            # Create convolution layer
            conv = jax.nn.Conv2d(
                input_shape=(C_i, H, W, 1),  # Note input_shape is inferred from embeddings
                output_shape=(C_i, self.out_channels, H, 1),
                kernel_size=(H, W), padding=(0, padding),
                kernel_type='valid'
            )
            self.register_module(f'Conv_{layer}', conv)

            # Apply activation and pooling
            conv_out = jax.nn.relu(conv)(conv)  # Using 'act' from kwargs
            pooled_out = jax.nn.max_pool2d(
                conv_out, ksize=(1, self.kwargs['pool_size']),
                strides=(1, 2).concat(*conv_out.shape[2:].last_split(-1)[0]))
            self.conv_layers.append((conv_out, pooled_out))  # Store output after each layer

        # RNN
        self.rnn = jax.nn.RNN(
            # inferred from kwargs
            hidden_dim=self.kwargs['hid_dim'],
            time_steps=floor(self.kwargs['seq_len'] / self.kwargs['pool_size']),  # Assuming seq_len is provided elsewhere?
            cell=self.kwargs['cell'],
            bidirectional=self.kwargs['bidi']
        )

        # Proj: Assuming rnn_output shape is (batch x time_steps x hid_dim * 2)
        # Average n-1 steps and sum last step
        def proj(rnn_output):
            # rnn_output shape: (batch, time_steps, hid_dim*2)
            # Average n-1 steps and sum last step
            time_steps_inf = jnp.inf  # This is an issue; need actual time_steps
            # For the sake of code, assuming time_steps is known, but since it's part of kernel_sizes?
            # Alternatively, compute time_steps from kernel_sizes and pool_size
            actual_time_steps = jnp.floor_divide(
                jnp.array(self.kwargs['seq_len']), self.kwargs['pool_size'])
            # Reshape to (batch, hid_dim*2 * actual_time_steps)
            rnn_output = reshape(rnn_output, (rnn_output.shape[0], 
                                  actual_time_steps, -1))
            avg_n_minus_1 = jnp.sum(rnn_output[:-1], axis=1).mean(axis=0)
            sum_last = rnn_output[-1]
            proj_out = jnp.concatenate((avg_n_minus_1, sum_last), axis=1)
            return proj_out

        self.register_function(proj, 'proj')

        self.rnn_out = self.rnn(self.conv_layers[-1][1])  # Input to RNN is after pooling
        self.rnn_out = proj(self.rnn_out)

    # Note: The original PyTorch code has an issue with input dimensions to RNN.
    # In JAX, need to ensure the input to RNN has proper time_steps inferred from the actual sequence length after pooling.

# Usage example (assuming input is a JAX array with shape (batch, vocab, seq_len, 1)):
# conv_rec = ConvRec(
#     n_classes=5,
#     vocab=1000,
#     emb_dim=256,
#     out_channels=64,
#     kernel_sizes=(5, 3),
#     pool_size=2,
#     hid_dim=256,
#     cell='LSTM',
#     bidi=True,
#     dropout=0.5,
#     act='relu',
#     seq_len=50,
#     **kwargs
# )

# Potential issues:
# 1. seq_len is not directly a parameter but inferred from input data shape.
# 2. Input shape handling for RNN (specifically time_steps).
# 3