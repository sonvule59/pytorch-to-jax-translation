import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from typing import Dict, Tuple, Any
from jax import random, jit, value_and_grad

# CNN Module
class CNN(nn.Module):
    """Convolutional Neural Network for processing embedded sequences."""
    param: Dict

    def setup(self):
        self.embedding_size = self.param['embedding_size']
        self.conv1 = nn.Conv(features=64, kernel_size=(3,), padding='SAME',
                            kernel_init=nn.initializers.normal(stddev=0.1),
                            bias_init=nn.initializers.zeros)
        self.conv2 = nn.Conv(features=self.embedding_size, kernel_size=(3,), padding='SAME',
                            kernel_init=nn.initializers.normal(stddev=0.1),
                            bias_init=nn.initializers.zeros)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Process input through CNN and pool to fixed size."""
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=1)  # Global average pooling
        return x

# MemN2N Model
class MemN2N(nn.Module):
    """End-to-end Memory Network (MemN2N) implemented in JAX using Flax."""
    param: Dict

    def setup(self):
        """Initialize model parameters."""
        self.hops = self.param['hops']
        self.vocab_size = self.param['vocab_size']
        self.embedding_size = self.param['embedding_size']

        # Define layers
        self.embedding = nn.Embed(num_embeddings=self.vocab_size,
                                features=self.embedding_size,
                                embedding_init=nn.initializers.normal(stddev=0.1))
        self.cnn = CNN(param=self.param)
        self.linear = nn.Dense(features=self.embedding_size,
                              kernel_init=nn.initializers.normal(stddev=0.1),
                              bias_init=nn.initializers.zeros)
        self.softmax = nn.softmax

    def __call__(self, utter: jnp.ndarray, memory: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the MemN2N model."""
        utter_emb = self.embedding(utter)  # (batch_size, utterance_length, embedding_size)
        utter_emb_sum = self.cnn(utter_emb)  # (batch_size, embedding_size)
        contexts = [utter_emb_sum]

        for _ in range(self.hops):
            memory_unbound = jnp.transpose(memory, (1, 0, 2))  # (memory_size, batch_size, story_length)
            memory_emb_sum = jnp.array([self.cnn(self.embedding(story))
                                      for story in memory_unbound])  # (memory_size, batch_size, embedding_size)
            memory_emb_sum = jnp.transpose(memory_emb_sum, (1, 0, 2))  # (batch_size, memory_size, embedding_size)

            context_temp = contexts[-1][:, None, :]  # (batch_size, 1, embedding_size)
            attention = jnp.sum(memory_emb_sum * context_temp, axis=2)  # (batch_size, memory_size)
            attention = self.softmax(attention, axis=-1)  # (batch_size, memory_size)

            attention = jnp.expand_dims(attention, axis=-1)  # (batch_size, memory_size, 1)
            attn_stories = jnp.sum(attention * memory_emb_sum, axis=1)  # (batch_size, embedding_size)

            new_context = self.linear(contexts[-1]) + attn_stories  # (batch_size, embedding_size)
            contexts.append(new_context)

        return contexts[-1]

# Synthetic Dataset Generator
def generate_synthetic_data(key: random.PRNGKey, batch_size: int, vocab_size: int,
                           utterance_length: int, memory_size: int, story_length: int
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate synthetic data for MemN2N."""
    key_utter, key_memory = random.split(key)
    utter = random.randint(key_utter, (batch_size, utterance_length), 0, vocab_size)
    memory = random.randint(key_memory, (batch_size, memory_size, story_length), 0, vocab_size)
    embed = nn.Embed(num_embeddings=vocab_size, features=128,
                    embedding_init=nn.initializers.normal(stddev=0.1))
    variables = embed.init(key, utter)
    target = jnp.mean(embed.apply(variables, utter), axis=1)
    return utter, memory, target

# Loss Function
def loss_fn(params: Dict, apply_fn, utter: jnp.ndarray, memory: jnp.ndarray,
            target: jnp.ndarray) -> float:
    """Compute mean squared error loss."""
    pred = apply_fn({'params': params}, utter, memory)
    return jnp.mean((pred - target) ** 2)

# Training Step
def train_step(params: Dict, opt_state: Any, apply_fn, utter: jnp.ndarray,
               memory: jnp.ndarray, target: jnp.ndarray) -> Tuple[Dict, Any, float]:
    """Perform one training step."""
    loss, grads = value_and_grad(loss_fn)(params, apply_fn, utter, memory, target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Apply JIT compilation with static_argnums
train_step = jit(train_step, static_argnums=(2,))

# Main Program
def main():
    # Hyperparameters
    param = {
        'hops': 3,
        'vocab_size': 1000,
        'embedding_size': 128
    }
    batch_size = 32
    utterance_length = 10
    memory_size = 5
    story_length = 20
    num_epochs = 10
    learning_rate = 0.001

    # Initialize model and optimizer
    model = MemN2N(param=param)
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    variables = model.init(init_rng, utter=jnp.ones((batch_size, utterance_length), dtype=jnp.int32),
                         memory=jnp.ones((batch_size, memory_size, story_length), dtype=jnp.int32))
    params = variables['params']
    global optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(num_epochs):
        rng, data_rng = random.split(rng)
        utter, memory, target = generate_synthetic_data(data_rng, batch_size, param['vocab_size'],
                                                      utterance_length, memory_size, story_length)
        params, opt_state, loss = train_step(params, opt_state, model.apply,
                                            utter, memory, target)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # Example inference
    rng, infer_rng = random.split(rng)
    utter, memory, _ = generate_synthetic_data(infer_rng, 1, param['vocab_size'],
                                             utterance_length, memory_size, story_length)
    output = model.apply({'params': params}, utter, memory)
    print(f"\nExample inference output shape: {output.shape}")
    print(f"Example inference output: {output[0, :5]}...")  # Print first 5 elements

if __name__ == "__main__":
    main()