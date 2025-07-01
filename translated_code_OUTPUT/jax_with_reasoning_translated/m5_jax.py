import jax
import jax.numpy as jnp
from jax import device, vmap
from jax.nn import Module, Layer, Sequential, DenseRNN

# Set device to CPU (default is CPU, can change to GPU with device('gpu'))
jax.device.set_string_type('cpu')

# Generate synthetic sequential data
def generate_data():
    sequence_length = 10
    num_samples = 100
    X = jnp.linspace(0, 4 * jnp.pi, num_samples).unsqueeze(1)
    y = jnp.sin(X)
    return X, y

X, y = generate_data()

# Create input-output sequences
def create_sequences(data, seq_length):
    in_out = []
    for i in range(len(data) - seq_length):
        in_seq = data[i:i+seq_length]
        out_seq = data[i+seq_length]
        in_out.append((in_seq, out_seq))
    X_seq, y_seq = jnp.stack([x[0] for x in in_out], axis=0),
                           jnp.stack([x[1] for x in in_out], axis=0)
    return X_seq, y_seq

X_seq, y_seq = create_sequences(y, sequence_length)

# Define the JAX RNN Model
class RNNModel(jax.nn.Module):
    @jax.jit.compile(vmap(jax.nn.Sequential(
        jax.nn.DenseRNN(
            input_shape=(jnp.ndim(X_seq) - 1, 1, sequence_length),
            hidden_size=50,
            num_layers=1,
            use_dropout=False,
            connections='deterministic'
        )
    )), jax.nn.Dense(1, jnp.size(y_seq), use_bias=False))
    def __call__(self, x):
        return self._forward(x)

    @jax.jit.compile
    def _forward(self, x):
        x, _ = self._rnn(x)
        return self._fc(x[-1, :])

    def _rnn(self, x):
        return jax.nn.RNN(self._dense_rnn_layer)(x, None)

    @jax.jit.compile
    def _dense_rnn_layer(self, x):
        return jax.nn.Dense(50, uses_bias=False)(x)

    @jax.jit.compile
    def _fc(self, x):
        return jax.nn.Dense(1)(x)

# Initialize model, loss, and optimizer
model = RNNModel()
criterion = jax.nn.mse_loss(jax.device('/cpu'), jax.device('/cpu'))
optimizer = jax.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 500
for epoch in range(epochs):
    for i, (seqs, labels) in enumerate(zip(X_seq, y_seq)):
        seqs = jnp.expand_dims(seqs, axis=0)
        labels = jnp.expand_dims(labels, axis=0)
        
        with jax.blocking scopes.checkpoint():
            loss = criterion(model(seqs), labels)
        
            jax.grad(loss)(model)
            optimizer.step()
            optimizer.apply(jax.random_state(i), jax.random_device('/cpu'))
    
    jax.print_values(loss.item())

# Testing on new data
X_test = jnp.linspace(4 * jnp.pi, 5 * jnp.pi, 10).unsqueeze(1)
X_test = jnp.expand_dims(X_test, axis=0)

with jax.blocking scopes.checkpoint:
    predictions = model(X_test)
    jax.print_values(predictions.tolist())