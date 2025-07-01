14.5


# Given that JAX doesn't support some WildcatPool2d implementation,
# we'll adapt the pooling to JAX's native functions. Modify accordingly.

from jax import tensor, shapes
import jax.vjp
import jax.numpy as jnp

def jax_wildcat_pool2d(kx, ky, kmin, kmax=None, alpha=1.0, stride=(1, 1)):
    """JAX implementation of WildCat Pooling 2D"""
    h, w = shapes(kx, ky)
    if kmin is None:
        kmin = (stride[0], stride[1])

    # Reshape input to separate spatial dimensions
    x = jnp.reshape(kx, (h * w, ky[0], ky[1]))
    y = jnp.reshape(kx, (kx.size(0), ky[1], h))

    # Perform strided convolution-like operations for downsampling
    scale = alpha
    if kmax is None:
        kmax = (stride[0] // int(kx.size(0)**0.5), stride[1] // int(kx.size(0)**0.5))
    else:
        kmax = (kmax[0], kmax[1])

    # Effective downsampling factors
    fh = int(kx.size(0) ** 0.5)
    fw = int(kx.size(1) ** 0.5)

    # PyTorch-style strided convolution
    output = jnp.zeros((h // fh, w // fw, ky[0] * ky[1]))
    jax.vjp.func(
        jax.vjp.jit_compile(jax.vjp.jit_compile(jax.numpy(jx)).conv2d,
        inputs=(x, jnp.ones((h, w), jnp.int32()), jnp.ones((ky[1], h), jnp.int32())),
        output=output,
        strides=(None, (h // fh, ky[0] * ky[1]), (ky[1], None))
    )

    # Average pooling over spatial dimensions
    return jnp.mean(output, axis=(0, 1))


class ResNetWSL_jax(jax.app.Application):

    def __init__(self, num_classes=13, pretrained=False, kmax=1, kmin=None, alpha=1.0, num_maps=1):
        """
        Initialize the JAX version of ResNet-WSL for JAX VJP.
        """
        super().__init__()

        # Load pre-trained model features and classifier
        if pretrained:
            # Assuming a pre-trained ResNet model is loaded (this is simplified)
            self.features = jax.nn.Sequential(
                jax.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                jax.nn.BatchNorm2d(64),
                jax.nn.ReLU(),
                jax.nn.MaxPool2d(2),
                jax.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                jax.nn.BatchNorm2d(128),
                jax.nn.ReLU(),
                jax.nn.MaxPool2d(2),
                jax.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                jax.nn.BatchNorm2d(256),
                jax.nn.ReLU(),
                jax.nn.MaxPool2d(2),
                jax.nn.Conv2d(256, 1024, kernel_size=1, padding=0))  # Adjust based on actual ResNet

        else:
            # Initialize from scratch
            self.features = jax.nn.Sequential(
                jax.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                jax.nn.BatchNorm2d(64),
                jax.nn.ReLU(),
                jax.nn.MaxPool2d(2),
                jax.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                jax.nn.BatchNorm2d(128),
                jax.nn.ReLU(),
                jax.nn.MaxPool2d(2),
                jax.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                jax.nn.BatchNorm2d(256),
                jax.nn.ReLU(),
                jax.nn.MaxPool2d(2),
                jax.nn.Conv2d(256, 1024, kernel_size=1, padding=0)))

        # Classification head with JAX
        num_features = self.features[-1].out_features
        self.classifier = jax.nn.Sequential(
            jax.nn.Conv2d(num_features, num_classes * num_maps, kernel_size=1, stride=1, padding=0, bias=True),
            jax.nn.BatchNorm2d(num_classes * num_maps),
            jax.nn.Sigmoid