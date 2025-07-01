import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np


# === Modules ===

class CNNJAX(nn.Module):
    param: dict

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.param['vocab_size'],
            features=self.param['embedding_size']
        )
        self.conv = nn.Conv(
            features=self.param['embedding_size'],
            kernel_size=(3, 3)
        )

    def __call__(self, x):
        x_emb = self.embedding(x)  # [batch, seq, embed]
        x_emb = x_emb[..., None]   # [batch, seq, embed, 1]
        conv_out = self.conv(x_emb)
        conv_out = nn.relu(conv_out)
        pooled = jnp.mean(conv_out, axis=(1, 2))  # Global average pooling
        return pooled


class MemN2N(nn.Module):
    param: dict

    def setup(self):
        self.hops = self.param['hops']
        self.embedding = nn.Embed(
            num_embeddings=self.param['vocab_size'],
            features=self.param['embedding_size']
        )
        self.cnn = CNNJAX(self.param)
    @nn.compact
    def __call__(self, utter, memory):
        utter_emb = self.embedding(utter)  # [batch, seq, embed]
        context = jnp.sum(utter_emb, axis=1)  # [batch, embed]

        for _ in range(self.hops):
            memory_emb = self.cnn(memory)  # [batch, embed]
            context_temp = utter_emb[:, -1, :]  # [batch, embed]

            attn_logits = jnp.sum(memory_emb * context_temp, axis=-1)
            attn = jax.nn.softmax(attn_logits, axis=-1)
            attn = attn[:, None]
            attn_applied = attn * memory_emb  # [batch, embed]
            combined = jnp.concatenate([context, attn_applied], axis=-1)  # [batch, 2*embed]

            # ✅ Dynamic Dense layer avoids shape mismatch
            context = nn.Dense(features=self.param['out_size'])(combined)

        return context


# === Training setup ===

def create_train_state(rng, model, learning_rate, utter, memory):
    params = model.init(rng, utter, memory)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_loss(params, model, utter, memory, targets):
    preds = model.apply(params, utter, memory)
    return jnp.mean((preds - targets) ** 2)


@jax.jit
def train_step(state, utter, memory, targets):
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params, model, utter, memory, targets)

    state = state.apply_gradients(grads=grads)
    return state, loss


# === Parameters and Data ===

param = {
    'vocab_size': 1000,
    'embedding_size': 64,  # Must match CNN features and attn dims
    'hops': 3,
    'out_size': 128,       # Final output size as required
}

batch_size = 32
seq_len = 10
num_epochs = 10
learning_rate = 0.001

# Dummy training data
rng = jax.random.PRNGKey(0)
utter_train = jnp.array(np.random.randint(0, param['vocab_size'], (batch_size, seq_len)))
memory_train = jnp.array(np.random.randint(0, param['vocab_size'], (batch_size, seq_len)))
targets_train = jnp.array(np.random.randn(batch_size, param['out_size'])).astype(jnp.float32)

# === Train ===

model = MemN2N(param)
state = create_train_state(rng, model, learning_rate, utter_train, memory_train)

for epoch in range(1, num_epochs + 1):
    state, loss = train_step(state, utter_train, memory_train, targets_train)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

# === Inference ===

test_utter = jnp.array(np.random.randint(0, param['vocab_size'], (1, seq_len)))
test_memory = jnp.array(np.random.randint(0, param['vocab_size'], (1, seq_len)))
output = model.apply(state.params, test_utter, test_memory)

print("\nExample inference output shape:", output.shape)
print("Example inference output:", output[0][:5])  # Show first 5 values
