import jax
import jax.nn as jnn
import jax.numpy as jnp
from typing import Dict, Tuple, Any
from Any import CNNJAX
class MemN2N(jax.nn.Module):
    __init__(self, param):
        super(MemN2N, self).__init__(dim=-1)
        self.param = param
        self.hops = self.param['hops']
        self.vocab_size = self.param['vocab_size']
        self.embedding_size = self.param['embedding_size']

        self.embedding = jnn.Embedding(self.vocab_size, self.embedding_size)
        self.cnn = CNNJAX(self.param, embedding=self.embedding)  # Assume CNNJAX is defined
        self.linear = jnp.array([jnp.array(self.param['cnn_weights'])], axis=0)
        self.softmax = jnp.array(jax.nn.Softmax()(jnp.ones((1, self.embedding_size, self.embedding_size, self.embedding_size))))
        
        self.weights_init()

    def forward(self, utter, memory):
        utter_emb = self.embedding(uter)
        contexts = [utter_emb.sum(axis=-1)]  # Assuming sum over last dimension

        for _ in range(self.hops):
            # Process memory (assumed shape compatibility)
            memory_emb = self.cnn(memory)
            context_temp = utter_emb[-1, :]  # Assuming attention mechanism
            attention = jnp.dot(memory_emb, context_temp).softmax(dim=-1)
            attention = jax.nn.Softmax(attention)
            attention = jnp.expand_dims(attention, -1)
            attn_stories = jnp.sum(attention * memory_emb, axis=-1)
            new_context = jnp.dot(self.linear, [contexts[-1], attn_stories])
            contexts.append(new_context)

        return contexts[-1]

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, jnn.Linear) or isinstance(m, jnn.Embedding):
                m.init_random_uniform(scale=0.1)
            if isinstance(m, jnn.Embedding):
                m.init_zero_mean_uniform(dim=0)

# Note: The CNNJAX and embedding handling in forward pass are simplified
# and may require adjustments based on the actual CNN implementation.