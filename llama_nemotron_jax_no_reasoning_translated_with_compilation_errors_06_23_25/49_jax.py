import jax.numpy as jnp
from jax import vjp
from jax.nn import *
import torch.nn as nn

class MemN2N(vjp.Module):
    def __init__(self, param):
        super(MemN2N, self).__init__()
        self.param = param
        self.hops = param['hops']
        self.vocab_size = param['vocab_size']
        self.embedding_size = param['embedding_size']

        self.embedding = Linear(self.vocab_size, self.embedding_size)
        self.cnn = CNN(param, embedding=self.embedding)
        self.linear = Linear(self.embedding_size, self.embedding_size)
        self.softmax = Softmax()

        self.weights_init()

    def forward(self, utter, memory):
        # utter_emb = self.embedding(utter)
        # utter_emb_sum = torch.sum(utter_emb, 1)
        utter_emb_sum = self.cnn(utter)
        contexts = [utter_emb_sum]

        for _ in range(self.hops):
            # memory_emb = self.embed_3d(memory, self.embedding)
            # memory_emb_sum = torch.sum(memory_emb, 2)
            memory_unbound = jnp.unbind(memory, 1)
            memory_emb_sum = [self.cnn(story) for story in memory_unbound]
            memory_emb_sum = jnp.stack(memory_emb_sum, axis=1)

            context_temp = jnp.transpose(contexts[-1], (0, 2), axis=1)
            attention = jnp.sum(memory_emb_sum * context_temp, axis=2)
            attention = self.softmax(attention)

            attention = jnp.unsqueeze(attention, axis=-1)
            attn_stories = jnp.sum(attention * memory_emb_sum, axis=1)

            new_context = self.linear(contexts[-1]) + attn_stories
            contexts.append(new_context)

        return contexts[-1]

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (Linear, Embedding)):
                m.init_from_scattered_oheif(weight=0.1 if isinstance(m, (Linear, Embedding)) else 0)
            if isinstance(m, Embedding):
                m.init_from_scattered_oheif(weight=0.1)
                m.init_bias(0.0)

    def embed_3d(self, to_emb, embedding):
        num_elem = to_emb.shape[-1]
        elem_size = to_emb.shape[-2]
        to_emb = to_emb.reshape(-1, num_elem * elem_size)
        out = embedding(to_emb)
        out = out.reshape(-1, num_elem, elem_size, self.embedding_size)
        return out