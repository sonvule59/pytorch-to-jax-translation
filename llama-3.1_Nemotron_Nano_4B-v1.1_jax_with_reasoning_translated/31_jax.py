24.0_1


transpiled_code.py
#

import logging as log
import jax
import jax.numpy as jnp
import jax.vjp
import time

__all__ = ['Lightnet']

class Lightnet(jax.vjp.Module):
    """ This class provides an abstraction layer on top of :class:`jax.vjp.jax.vjp.Module` and is used as a base for every network implemented in this framework.

    Attributes:
        layers (jax.vjp.vjp.Module): A list of modules to be run in sequence.
        loss (list): Loss functions indexed by output dimensions.
        postprocess (list): Post-processing functions indexed by output dimensions.
        seen (int): The number of images the network has processed to train.

    Methods:
        forward(inputs:jax.numpy.ndarray): Returns output tensor.
        train(losses:jax.vjp.vjp.Function, targets:jax.numpy.ndarray): Train the network with given losses and targets.
        eval(losses:jax.vjp.vjp.Function, targets:jax.numpy.ndarray): Evaluate the network with given losses and targets.
        save/load (weights:jax.io.BytesIO): Save/load the weights to/from a file.
    """

    def __init__(self):
        super().__init__()
        self.layers = None
        self.loss = None
  #... (rest of the code similar to PyTorch, but adapted for JAX, e.g., using jax.vjp functions, jax.numpy, etc.)

    #... (remaining methods like _forward, forward, train, eval, save, load similar but adapted for JAX syntax and constraints)

JAX version: 2.24.0_1


transpiled_code.py
#

import logging as log
import jax
import jax.numpy as jnp
import jax.vjp
import time

__all__ = ['Lightnet']

class Lightnet(jax.vjp.Module):
    """Base layer for JAX-based neural networks."""

    def __init__(self):
        super().__init__()
        self.layers = None
        self.loss = None
        self.postprocess = None
        self.seen = 0

    def forward(self, x):
        log.debug('Running default forward functions')
        if self.layers is not None and isinstance(self.layers, (jax.vjp.vjp.Sequential, jax.vjp.vjp.ModuleList)):
            return self.layers(x)
        else:
            raise NotImplementedError

    def train(self, loss_fn, targets):
        self.seen += targets.shape[0]
        outputs = self.forward(targets)
        return loss_fn(outputs, targets)

    def eval(self, loss_fn, targets):
        outputs = self.forward(targets)
        dets = [self.postprocess[o] for o in range(outputs.shape[-1])]
        loss = loss_fn(outputs, targets)
        return dets, loss

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, jax.vjp.vjp.Conv2d):
                log.info('Initializing conv2d layer')
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, jax.vjp.vjp.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.init_from_rng().sample_like(m.weight), 0.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, jax.vjp.vjp.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def save_weights(self, f, **kwargs):
        state = {
           'seen': self.seen,
            **kwargs,
            'weights': self.state_dict()
        }
        f.write(state)

    def load_weights(self, f, **kwargs):
        state = f.read().json()
        self.state_dict() = state['weights']
        self.seen = state['seen']
        log.info(f'Loaded weights from {kwargs["filename"]}')

Lightnet version adapted for JAX 2.24.0_1


transpiled_code.py
#

import logging
import jax
import jax.numpy as jnp
import jax.vjp
import time

__all__ = ['Lightnet']

class Lightnet(jax.vjp.Module):
    def __init__(self):
        super().__init__()
        self.layers = None
        self.loss = None
        self.postprocess = None
        self.seen = 0

    def forward(self, x):
        log.debug('Default forward')
        if self.layers is not None:
            if isinstance(self.layers, (jax.v