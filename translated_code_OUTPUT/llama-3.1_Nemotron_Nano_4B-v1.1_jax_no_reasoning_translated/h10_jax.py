import jax
import jax.numpy as jnp
from jax import castle, gradient
from jax.sharding import shard
from jax.experimental.build_model_harness import model_harness
from jax.nn import layers as jnn
import jax.data
import jax.random

# Define the ResNet model with JAX-compatible layers
def resnet18(x):
    x = jax.random.normal(jax.float32, jnp.ones((1, *x.shape[1:])) / jnp.sqrt(10.0), cast=x.dtype)
    x = jax.nn.conv2d(x, 3, strides=2, kernel_size=(3,3), output_mode='same', bias=jax.nn.zeros_like(x))
    x = jax.nn.relu(x)
    x = jax.nn.max_pool2d(x, (2,2), strides=2, kernel_size=(2,2), padding='same')
    x = jax.nn.conv2d(x, 1, strides=2, kernel_size=(5,5), output_mode='same', bias=jax.nn.zeros_like(x))
    x = jax.nn.relu(x)
    x = jax.nn.max_pool2d(x, (1,1), strides=1, kernel_size=(2,2), padding='same')
    x = jax.nn.conv2d(x, 3, strides=1, kernel_size=(3,3), output_mode='same', bias=jax.nn.zeros_like(x))
    x = jax.nn.relu(x)
    x = jax.nn.max_pool2d(x, (2,2), strides=1, kernel_size=(2,2), padding='same')
    x = jax.nn.dropout(x, jax.nn.sigmoid(), rate=0.5)
    x = jax.nn.adaptive_avg_pool2d_with_stride(x, jax.nn.avg_pool_type_stride(2), kernel_size=(2,2))
    x = jax.nn.dropout(x, jax.nn.sigmoid(), rate=0.5)
    x = jax.nn.max_pool2d(x, (2,2), strides=1, kernel_size=(2,2), padding='same')
    x = jax.nn.conv2d(x, 1, strides=1, kernel_size=(1,1), output_mode='same', bias=jax.nn.zeros_like(x))
    x = jax.nn.log_softmax(x, axis=-1)
    return x

# Define the model harness
model_h = model_harness('geometric', jax sharding shard(0, 0), model, jax.l3.register_jax_library())
model_h.initialize()

# Prepare input data
data = jax.random.uniform(jax.int32(), jax.range(0, 1), structures=(jnp.ndim(data.shape), jnp.size(data.shape)[0:]))
data = jax.data.Dataset.from_tensor_slices((image, jnp.ones((1, *data.shape[1:]))))

# Forward pass
output = model_h(data)
predicted_class = jnp.argmax(output, axis=-1)[0]

# Compute gradients
grads = model_h.compute_gradients((output, jnp.ones((1, *output.shape[1:]))))[0][1]

# Average weights over spatial dimensions
weights = jnp.mean(grads, axis=[1, 2])

# Compute Grad-CAM weights
heatmap = jnp.sum(weights * output, axis=[1, 2])
heatmap = jnp.exp(heatmap) / jnp.sum(jnp.exp(heatmap), axis=[1, 2], keepdims=True)

# Reshape and transpose for visualization
heatmap = jnp.transpose(heatmap, (2, 0, 1))
heatmap = jnp.expand_dims(heatmap, axis=0)

# Note: JAX doesn't have direct PIL image handling; use jax.vjp
import jax.vjp as vjp
image_jax = vjp.as_jax_object(image)
heatmap_jax = vjp.as_jax_object(heatmap)

# Display the image with the Grad-CAM heatmap (requires VJP for PIL interaction)
# vjp.show(image_jax, heatmap_jax, style='div style')