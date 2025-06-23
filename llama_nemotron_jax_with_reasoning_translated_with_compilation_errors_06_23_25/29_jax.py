import jax
from jax import numpy as jnp
from jax import compactor
from jax import vmap
from jax.nn import layers, convolution, convolution2d, batch_normal, linear
import jax.random as jrandom
import jax.numpy as jnp


class Bottleneck(jax.nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super().__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        # Convert PyTorch layers to JAX equivalents
        self.conv1 = convolution(jnp.ones((1, 64, 1, 1)), jnp.random.normal(0., 1.), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = batch_normal2d(jnp.ones((1, 64, 1, 1)), jrandom.normal(0., 1., seed=0))
        self.conv2 = convolution(jnp.ones((1, 1, 3, 3)), jrandom.normal(0., 1.), kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = batch_normal2d(jnp.ones((1, 1, 3, 3)), jrandom.normal(0., 1., seed=0))
        self.conv3 = convolution(jnp.ones((1, out_planes+dense_depth, 1, 1)), jrandom.normal(0., 1.), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = batch_normal2d(jnp.ones((1, out_planes+dense_depth, 1, 1)), jrandom.normal(0., 1., seed=0))

        self.shortcut = nn.Sequential()  # JAX equivalent of PyTorch shortcut
        if first_layer:
            conv = convolution(jnp.ones((1, out_planes+dense_depth, 1, 1)), jrandom.normal(0., 1.), kernel_size=1, stride=stride, bias=False)
            bn = batch_normal2d(jnp.ones((1, out_planes+dense_depth, 1, 1)), jrandom.normal(0., 1., seed=0))
            self.shortcut = nn.Sequential(conv, bn)

    def forward(self, x):
        out = jnp.relu(self.bn1(self.conv1(x)))
        out = jnp.relu(self.bn2(self.conv2(out)))
        out = jnp.relu(self.bn3(self.conv3(out)))
        x = self.shortcut(x)
        d = self.out_planes
        out = jnp.cat([x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :] + out[:, d:, :, :]], axis=1)
        out = jnp.relu(out)
        return out


class DPN(jax.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        # Convert PyTorch layers to JAX equivalents
        self.conv1 = convolution(jnp.ones((1, 64, 3, 3)), jrandom.normal(0., 1., seed=0), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = batch_normal2d(jnp.ones((1, 64, 3, 3)), jrandom.normal(0., 1., seed=0))
        self.last_planes = 64
        # Define JAX equivalents for the layers
        def make_layer(in_planes, out_planes, num_blocks, dense_depth, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for i, stride in enumerate(strides):
                layers.append(Bottleneck(last_planes=in_planes, 
                                         in_planes=in_planes, 
                                         out_planes=out_planes, 
                                         dense_depth=dense_depth[i], 
                                         stride=stride, 
                                         first_layer=(i==0)))
                self.last_planes = out_planes + (i+2)*dense_depth
            return jax.nn.Sequential(*layers)
        
        self.layer1 = make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = make_layer(in_planes[3], out_planes[3],