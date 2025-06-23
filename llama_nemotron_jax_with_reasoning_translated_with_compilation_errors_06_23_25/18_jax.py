import jax
from jax import numpy as jnp
from jax import random
from jax import vmap
from jax.nn import LayerNormalization
from jax.optim import LeastSquares
from jax import compactor
from jax import vmap
from jax.data import Dataset
from jax.random import RealDistribution

class Vision_M_jax(nn.jax.nn.Module):
    def __init__(self):
        super(Vision_M_jax, self).__init__()
        self.conv1 = nn.jax.conv2d(in_channels=3, out_channels=32, kernel_size=(8,8), stride=(4,4))
        self.conv2 = nn.jax.conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=(2,2))
        self.conv3 = nn.jax.conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class Language_M_jax(nn.jax.nn.Module):
    def __init__(self, vocab_size=10, embed_dim=128, hidden_size=128):
        super(Language_M_jax, self).__init__()
        self.embeddings = nn.jax.embeddings(vocab_size, embed_dim)
        self.lstm = nn.jax.lstm(LayerNormalization()(nn.jax.lstm(embed_dim, hidden_size, batch_first=True)))
        
    def forward(self, x):
        embedded_input = self.embeddings(x)
        out, hn = self.lstm(embedded_input)
        return out[:x.size()[0], :]  # Slicing to match PyTorch output shape
        
    def LP(self, x):
        return nn.jax.linear(x, self.embeddings.weight)

class Mixing_M_jax(nn.jax.nn.Module):
    def __init__(self):
        super(Mixing_M_jax, self).__init__()
        
    def forward(self, visual_encoded, instruction_encoded):
        batch_size = visual_encoded.shape[0]
        visual_flatten = visual_encoded.view(batch_size, -1)
        instruction_flatten = instruction_encoded.view(batch_size, -1)
        mixed = torch.cat([visual_flatten, instruction_flatten], dim=1)
        return mixed

class Action_M_jax(nn.jax.nn.Module):
    def __init__(self, batch_size=1, hidden_size=256):
        super(Action_M_jax, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        # Initialize hidden states with random values
        self.h1 = Variable(jnp.random.normal(0., scale=1., size=(batch_size, self.hidden_size)), seed=None)
        self.c1 = Variable(jnp.random.normal(0., scale=1., size=(batch_size, self.hidden_size)), seed=None)
        self.h2 = Variable(jnp.random.normal(0., scale=1., size=(batch_size, self.hidden_size)), seed=None)
        self.c2 = Variable(jnp.random.normal(0., scale=1., size=(batch_size, self.hidden_size)), seed=None)
        
        self.lstm_1 = nn.jax.lstm_cell(input_size=3264, hidden_size=self.hidden_size)
        self.lstm_2 = nn.jax.lstm_cell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 3264)
        
        # Feed forward through LSTM cells
        h1, c1 = self.lstm_1(x, self.h1)
        h2, c2 = self.lstm_2(h1, self.h2)
        
        # Update hidden states
        self.h1 = h1
        self.c1 = c1
        self.h2 = h2
        self.c2 = c2
        
        # Return the hidden state of the upper layer
        return h2

# Example usage of the JAX version of Vision_M, Language_M, etc.
# Note: This is just a skeleton; actual training loops and datasets are omitted.
#       Also, JAX's JIT compilation is assumed for all modules.
vision_m = Vision_M_jax()
language_m = Language_M_jax(vocab_size=10, embed_dim=128, hidden_size=128)
mixing_m = Mixing_M_jax()
action_m = Action_M_jax(batch_size=1, hidden_size=256)
policy = Policy(action_space=range(10))  # Assuming action space is 0-9
temporal_ae = temporal_AutoEncoder(policy, vision_m)
language_pred = Language_Prediction(language_m, vision_m, mixing