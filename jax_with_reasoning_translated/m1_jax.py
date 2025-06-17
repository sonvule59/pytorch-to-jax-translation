import jax
from jax import numpy as jnp
from jax.vjp import jit
jit(num_parallel=4)

class CustomLSTMModel(jax.jit.Module):
        #... same as PyTorch but using jax constructs
        def forward(self, inputs, H_C=None):
            #... same logic but adapted for JAX
            # Use jax.random.uniform for initialization if needed
            # Ensure all operations are compatible with JAX transformations

    class LSTMModel(jax.jit.Module):
        def __init__(self):
            super().__init__()
            self.lstm = jax.lstm.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
            self.fc = jax.nn.Linear(50, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    # Initialize JAX equivalents of optimizers, loss function, etc.
    # Use jax.optim.Adam instead of torch.optim.Adam

    # Adapt training loops to use JAX autograd
    # Ensure data is in JAX arrays and transformations are applied

# Example usage with JAX transformations
jax.vjp.transformations.enable(jax.vjp.transformations.jit)
# Note: Actual JAX code would need proper data loading and transformation steps,
# possibly using jax.data.pandas or other JAX data loaders.