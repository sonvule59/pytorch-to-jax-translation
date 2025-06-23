0.10


---
jax

import jax.numpy as jnp
from jax import concat, reshape, range, vmap
from jax.nn import LayerMixin, Conv2d, MaxPool2d, Linear, LogSoftmax
from jax.rnn import RNNFamily

class ConvRec(LayerMixin):
    """
    JAX version of the ConvRec model from PyTorch.
    """
    def __init__(self, **kwargs):
        super(ConvRec, self).__init__(**kwargs)
        # embeddings
        self.embeddings = Linear(self.vocab, self.emb_dim, padding_idx=self.padding_idx)
        # cnn
        self.conv_layers = []
        # Assuming out_channels is defined in kwargs; adjust as necessary
        # For the purpose of this example, let's assume out_channels is part of kwargs
        # However, in the original PyTorch code, out_channels is part of the class parameters.
        # Since the original code's __init__ method is not fully provided, 
        # we need to infer from the PyTorch code.
        # Given the PyTorch code, the out_channels is passed as out_channels=128, 
        # and kernel_sizes is (5,3)
        # For JAX, we'll replicate this.
        # However, in the original PyTorch code, the out_channels is used as the out_channels of each Conv2d.
        # So each Conv2d has out_channels as the filter size.
        # So in JAX, each Conv2d will have out_channels specified.
        # However, in the PyTorch code provided, the out_channels is passed as a parameter to the 
        # __init__ method of ConvRec, but in the code given, the out_channels is used in the 
        # loop over kernel_sizes. So each Conv2d uses out_channels as the out_channels.
        # Therefore, in JAX, we need to create Conv2d layers with out_channels.
        # However, in the original PyTorch code, the out_channels is passed as a parameter to the 
        # ConvRec class, not to each Conv2d layer. Wait, looking back at the PyTorch code:

        # Original PyTorch code:
        # for layer, W in enumerate(self.kernel_sizes):
        #     C_i, H = None, None
        #     if layer == 0:
        #         C_i, H = 1, self.emb_dim
        #     else:
        #         C_i, H = self.out_channels, 1
        #     conv = nn.Conv2d(C_i, self.out_channels, (H, W), padding=(0, self.padding))

        # So, the out_channels for each Conv2d is self.out_channels, which is a parameter of the 
        # ConvRec class, not per layer. So each Conv2d uses the same out_channels as specified 
        # in the parent class. Therefore, in JAX, each Conv2d will have the same out_channels.

        # Therefore, in JAX, the code for Conv layers would be:

        # Assuming self.out_channels is defined in kwargs, but in the original code, 
        # it's part of the parent class. However, in the given PyTorch code, the ConvRec's 
        # __init__ method does not receive out_channels as a parameter. Wait, looking at the original 
        # code provided by the user:

        # The ConvRec's __init__ method in PyTorch has:
        # def __init__(self, n_classes, vocab, emb_dim=100, padding_idx=None,
        #             out_channels=128, kernel_sizes=(5, 3),...)

        # So out_channels is a parameter of the ConvRec class. Therefore, in JAX, we need to 
        # pass out_channels to the Conv2d layers. However, in JAX, when creating a Conv2d layer, 
        # you can specify the in_channels, out_channels, etc. So each Conv2d layer in the 
        # JAX code should have out_channels as the same as self.out_channels.

        # Therefore, in JAX, the loop over kernel_sizes would create Conv2d layers with out_channels.

        # However, in the original PyTorch code, the code is:

        # for layer, W in enumerate(self.kernel_sizes):
        #     C_i, H =... 
        #     if layer ==0:
        #         C_i = 1, H = emb_dim
        #     else:
        #         C_i = self.out_channels, H =1
        #     conv = nn.Conv2d(C_i, self.out_channels,...)

        # So, for layer 0, input channels are 1 (from embeddings), output channels are self.out_channels.
        #