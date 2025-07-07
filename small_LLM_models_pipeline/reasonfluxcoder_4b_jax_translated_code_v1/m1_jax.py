# (only the code, no explanations)
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, grad, jit, vmap, lax
import matplotlib.pyplot as plt

# Generate synthetic sequential data
np.random.seed(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = jnp.linspace(0, 4 * 3.14159, steps=num_samples).reshape(1, -1)
y = jnp.sin(X)

# Prepare data for LSTM
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# Define the LSTM model
def lstm_cell(inputs, H, C, Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo, Wxc, Whc, bc):
    # inputs: (batch_size, input_dim)
    # H: (batch_size, hidden_dim)
    # C: (batch_size, hidden_dim)
    # Wxi: (input_dim, hidden_dim)
    # Whi: (hidden_dim, hidden_dim)
    # bi: (hidden_dim)
    # Wxf: (input_dim, hidden_dim)
    # Whf: (hidden_dim, hidden_dim)
    # bf: (hidden_dim)
    # Wxo: (input_dim, hidden_dim)
    # Who: (hidden_dim, hidden_dim)
    # bo: (hidden_dim)
    # Wxc: (input_dim, hidden_dim)
    # Whc: (hidden_dim, hidden_dim)
    # bc: (hidden_dim)
    # Output: (batch_size, hidden_dim), (batch_size, hidden_dim)
    # Compute the gates
    I_t = jnp.sigmoid(jnp.dot(inputs, Wxi) + jnp.dot(H, Whi) + bi)
    F_t = jnp.sigmoid(jnp.dot(inputs, Wxf) + jnp.dot(H, Whf) + bf)
    O_t = jnp.sigmoid(jnp.dot(inputs, Wxo) + jnp.dot(H, Who) + bo)
    C_tilde = jnp.tanh(jnp.dot(inputs, Wxc) + jnp.dot(H, Whc) + bc)
    # Update the cell state and hidden state
    C = F_t * C + I_t * C_tilde
    H = O_t * jnp.tanh(C)
    return H, C

def custom_lstm_model(inputs, params):
    # params is a dictionary containing the parameters
    # inputs: (batch_size, seq_len, input_dim)
    # Output: (batch_size, seq_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)
    batch_size, seq_len, input_dim = inputs.shape
    hidden_dim = params['hidden_dim']
    Wxi = params['Wxi']
    Whi = params['Whi']
    bi = params['bi']
    Wxf = params['Wxf']
    Whf = params['Whf']
    bf = params['bf']
    Wxo = params['Wxo']
    Who = params['Who']
    bo = params['bo']
    Wxc = params['Wxc']
    Whc = params['Whc']
    bc = params['bc']
    # Initialize the hidden and cell states
    H = jnp.zeros((batch_size, hidden_dim))
    C = jnp.zeros((batch_size, hidden_dim))
    # Iterate over the sequence
    Hs = []
    for t in range(seq_len):
        inputs_t = inputs[:, t, :]
        H, C = lstm_cell(inputs_t, H, C, Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo, Wxc, Whc, bc)
        Hs.append(H)
    # Stack the hidden states
    Hs = jnp.stack(Hs, axis=1)
    # Output the final hidden state and the sequence of hidden states
    return Hs, H, C

def custom_lstm_model_with_initial_state(inputs, params, H_C):
    # params is a dictionary containing the parameters
    # inputs: (batch_size, seq_len, input_dim)
    # H_C: (batch_size, hidden_dim), (batch_size, hidden_dim)
    # Output: (batch_size, seq_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)
    batch_size, seq_len, input_dim = inputs.shape
    hidden_dim = params['hidden_dim']
    Wxi = params['Wxi']
    Whi = params['Whi']
    bi = params['bi']
    Wxf = params['Wxf']
    Whf = params['Whf']
    bf = params['bf']
    Wxo = params['Wxo']
    Who = params['Who']
    bo = params['bo']
    Wxc = params['Wxc']
    Whc = params['Whc']
    bc = params['bc']
    # Use the initial H and C
    H, C = H_C
    # Iterate over the sequence
    Hs = []
    for t in range(seq_len):
        inputs_t = inputs[:, t, :]
        H, C = lstm_cell(inputs_t, H, C, Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo, Wxc, Whc, bc)
        Hs.append(H)
    # Stack the hidden states
    Hs = jnp.stack(Hs, axis=1)
    # Output the final hidden state and the sequence of hidden states
    return Hs, H, C

def custom_lstm_model_with_initial_state_and_output(inputs, params, H_C):
    # params is a dictionary containing the parameters
    # inputs: (batch_size, seq_len, input_dim)
    # H_C: (batch_size, hidden_dim), (batch_size, hidden_dim)
    # Output: (batch_size, seq_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)
    batch_size, seq_len, input_dim = inputs.shape
    hidden_dim = params['hidden_dim']
    Wxi = params['Wxi']
    Whi = params['Whi']
    bi = params['bi']
    Wxf = params['Wxf']
    Whf = params['Whf']
    bf = params['bf']
    Wxo = params['Wxo']
    Who = params['Who']
    bo = params['bo']
    Wxc = params['Wxc']
    Whc = params['Whc']
    bc = params['bc']
    # Use the initial H and C
    H, C = H_C
    # Iterate over the sequence
    Hs = []
    for t in range(seq_len):
        inputs_t = inputs[:, t, :]
        H, C = lstm_cell(inputs_t, H, C, Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo, Wxc, Whc, bc)
        Hs.append(H)
    # Stack the hidden states
    Hs = jnp.stack(Hs, axis=1)
    # Output the final hidden state and the sequence of hidden states
    return Hs, H, C

def inbuilt_lstm_model(inputs, params):
    # params is a dictionary containing the parameters
    # inputs: (batch_size, seq_len, input_dim)
    # Output: (batch_size, seq_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)
    batch_size, seq_len, input_dim = inputs.shape
    hidden_dim = params['hidden_dim']
    # Use the inbuilt LSTM
    _, (H, C) = lax.scan(lstm_cell, (jnp.zeros((batch_size, hidden_dim)), jnp.zeros((batch_size, hidden_dim))), inputs)
    # Output the final hidden state and the sequence of hidden states
    return H, C

def inbuilt_lstm_model_with_initial_state(inputs, params, H_C):
    # params is a dictionary containing the parameters
    # inputs: (batch_size, seq_len, input_dim)
    # H_C: (batch_size, hidden_dim), (batch_size, hidden_dim)
    # Output: (batch_size, seq_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)
    batch_size, seq_len, input_dim = inputs.shape
    hidden_dim = params['hidden_dim']
    # Use the inbuilt LSTM
    _, (H, C) = lax.scan(lstm_cell, (H_C[0], H_C[1]), inputs)
    # Output the final hidden state and the sequence of hidden states
    return H, C

def inbuilt_lstm_model_with_initial_state_and_output(inputs, params, H_C):
    # params is a dictionary containing the parameters
    # inputs: (batch_size, seq_len, input_dim)
    # H_C: (batch_size, hidden_dim), (batch_size, hidden_dim)
    # Output: (batch_size, seq_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)
    batch_size, seq_len, input_dim = inputs.shape
    hidden_dim = params['hidden_dim']
    # Use the in