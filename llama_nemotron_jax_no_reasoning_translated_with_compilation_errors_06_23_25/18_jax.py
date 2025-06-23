import jax
from jax import numpy as jnp
from jax.vjp import jax_vjp_library as jvp

# Input arguments for the JAX version
@jvp.jax_custom_function
def vision_m(jax_context, x):
    """
    Vision_M model forward pass
    """
    return jax.nn.Conv2d(jnp.int32(3), jnp.int32(32), jnp.tensor([8], jnp.int64), jnp.tensor([4], jnp.int64),
                          jnp.tensor([jnp.int64(x.shape[1]], jnp.int64), jnp.tensor([jnp.int64(x.shape[2]), jnp.int64(x.shape[2]+1)], jnp.int64)])
    # Note: Actual kernel_size, stride should be properly handled. 
    #       This example is simplified. In practice, kernel_size and stride should be 
    #       passed as jnp.array([8], jnp.int64) and jnp.array([4], jnp.int64) respectively.
    
@jvp.jax_custom_function
def language_m(jax_context, vocab_size, embed_dim, hidden_size, x):
    """
    Language_M model forward pass
    """
    embeddings = jax.nn.Embedding(jnp.int64(vocab_size), jnp.vectorize(lambda i: jnp.array([i], jnp.int64)), jnp.full(jnp.int64(vocab_size), 0, jnp.int64))
    lstm = jax.nn.LSTM(jnp.array([embed_dim], jnp.int64), jnp.array([hidden_size], jnp.int64), num_layers=1, batch_first=True)
    out, hn = lstm(x)
    return out

@jvp.jax_custom_function
def mixing_m(jax_context, visual_encoded, instruction_encoded):
    """
    Mixing Module forward pass
    """
    batch_size = visual_encoded.shape[0]
    visual_flatten = visual_encoded.view(batch_size, -1)
    instruction_flatten = instruction_encoded.view(batch_size, -1)
    mixed = jnp.cat([visual_flatten, instruction_flatten], axis=1)
    return mixed

class Action_M(jax.nn.Module):
    def __init__(self, batch_size=1, hidden_size=256):
        super().__init__()
        self.batch_size = jnp.array(batch_size, jnp.int64)
        self.hidden_size = jnp.array(hidden_size, jnp.int64)
        
        self.lstm_1 = jax.nn.LSTMCell(
            jnp.array([3264], jnp.int64), 
            jnp.array([self.hidden_size], jnp.int64))
        self.lstm_2 = jax.nn.LSTMCell(
            jnp.array([self.hidden_size], jnp.int64), 
            jnp.array([self.hidden_size], jnp.int64))
        
        self.hidden_1 = (jnp.zeros(jnp.array([self.hidden_size], jnp.int64)), 
                        jnp.zeros(jnp.array([self.hidden_size], jnp.int64)))
        self.hidden_2 = (jnp.zeros(jnp.array([self.hidden_size], jnp.int64)), 
                        jnp.zeros(jnp.array([self.hidden_size], jnp.int64)))
        
    def forward(self, x):
        # x shape: [batch_size, 1, 3264]
        x = jax.broadcast_to(x, (x.shape[0], 1, 3264))
        h1, c1 = self.lstm_1(x, self.hidden_1)
        h2, c2 = self.lstm_2(h1, self.hidden_2)
        self.hidden_1 = (h1, c1)
        self.hidden_2 = (h2, c2)
        return h2

# Other classes remain unchanged (e.g., SavedAction, temporal_AutoEncoder, Language_Prediction)
# Ensure all classes are defined in JAX compatible way