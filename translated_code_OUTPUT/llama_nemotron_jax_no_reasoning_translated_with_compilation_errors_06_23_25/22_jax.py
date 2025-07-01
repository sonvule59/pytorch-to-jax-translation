14.0


'''
Your task is to rewrite the provided PyTorch code using JAX-compatible functions and structures.
- Replace all PyTorch modules with JAX equivalents (e.g., use `jax.nn.Module`, `jax.nn.Conv2d`, etc.).
- Ensure that the JAX code is compatible with JAX V3.6+ (i.e., uses JAX Vectors and JAX Arrays).
- Assume inputs are JAX Arrays/Forms.
'''

import jax
import jax.nn as jnn
import jax.nn.layers as jnn_layers
from jax import vectorize, maybe_op

# JAX code starts here
jax.set_print_options(jax.print_options_)
import jax.numpy as jnp

class Inception(jax.nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super().__init__()
        # 1x1 conv branch
        self.b1 = jnn.Sequential(
            jnn.Conv2d(in_planes, n1x1, kernel_size=1),
            jnp.BatchNorm2d(n1x1),
            jnn.ReLU('relu'),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = jnn.Sequential(
            jnn.Conv2d(in_planes, n3x3red, kernel_size=1),
            jnp.BatchNorm2d(n3x3red),
            jnn.ReLU('relu'),
            jnn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            jnp.BatchNorm2d(n3x3),
            jnn.ReLU('relu'),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = jnn.Sequential(
            jnn.Conv2d(in_planes, n5x5red, kernel_size=1),
            jnp.BatchNorm2d(n5x5red),
            jnn.ReLU('relu'),
            jnn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            jnp.BatchNorm2d(n5x5),
            jnn.ReLU('relu'),
            jnn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            jnp.BatchNorm2d(n5x5),
            jnn.ReLU('relu'),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = jnn.Sequential(
            jnn.MaxPool2d(3, stride=1, padding=1),
            jnn.Conv2d(in_planes, pool_planes, kernel_size=1),
            jnp.BatchNorm2d(pool_planes),
            jnn.ReLU('relu'),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return jnp.cat([y1, y2, y3, y4], axis=1)


class GoogLeNet(jax.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layers = jnn.Sequential(
            jnn.Conv2d(3, 192, kernel_size=3, padding=1),
            jnp.BatchNorm2d(192),
            jnn.ReLU('relu'),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = jnn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = jnn.AvgPool2d(