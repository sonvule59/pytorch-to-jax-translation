// In[1]
import... // JAX imports
import jax.numpy as jnp
import jax import device
import jax.nn as jnn

// Replace loadimg with jax equivalent
def loadimg(path):...

// In[2]
gpu_num = 2
gradient = False
type2 = 'rgb' if not gradient else 'gd'

res_root = jax.data.ArraySpec(res_root)
scenes = jax.data.Dataset(scenes) // Convert glob pattern to dataset

for scene in scenes:
    //... // Wrap loops, model initialization, and processing in jax.for_i_iter

    // Model loading and training in JAX
    densenet = jnn.Sequential(
        // Define JAX model architecture similar to PyTorch's
    ).compile(jax.device_map.function_target)

    // Replace training loop with JAX equivalent

    // Save model and outputs using jax.save or jax.model_export