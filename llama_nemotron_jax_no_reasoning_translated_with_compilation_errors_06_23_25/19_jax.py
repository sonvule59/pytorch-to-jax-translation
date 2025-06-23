.7_2
LOCAL_RANK: 0
GLOBAL_RANK:?

# ---------------------------------------------------
# JAX code to convert PyTorch to JAX
# ---------------------------------------------------
import jax.numpy as jnp
import jax.vendor import jax_vendor
import jax.experimental.ragged
import jax.random as jax_rand
import jax.sharding
import jax.util
import jax.data.array_array
import jax.data.ao
import jax.data.ao_input
import jax.data.ao_output
import jax.data.ao_regularizer
from jax import misc

def jax_local_rank(): 
    """Returns the local rank of the process."""
    return jax.sharding.get_local_rank()

def jax_global_rank(): 
    """Returns the global rank of the process."""
    return jax.sharding.get_global_rank()

def jax_world_size(): 
    """Returns the world size of the process."""
    return jax.sharding.get_world_size()

# Replace PyTorch imports with JAX equivalents
# Original:
# import torch.distributed as dist
# Convert to:
def pytorch_to_jax_dist():
    """Convert PyTorch distribution system to JAX."""
    jax_vendor.install_channel(jax_vendor.LOAPEX)  # Ensure JAX is installed with LOAPEX for distributed
    return misc.jax.dist

# Original:
# import torch.utils.data.DataLoader
# Convert to:
def pytorch_to_jax_data_loader(data, sampler, batch_size, num_workers, pin_memory, drop_last):
    """Convert PyTorch DataLoader to JAX equivalent."""
    from jax.data import Dataset, ArrayArray
    # Assuming data is a JAX Dataset and sampler is a JAX sampler
    return jax.data.AIObservableArrayArray(
        jnp.array([data[i].to(jnp.float32) for i in jax.data.ArrayArray.data.indices]),
        jnp.array([data[i].to(jnp.float32) for i in jax.data.ArrayArray.data.indices]),
        jax.random.PRNGKey(0),  # Use a fixed key for reproducibility
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle=False,  # Set to True if needed, but JAX's AIObservableArray doesn't shuffle by default
        # Note: JAX's AIObservableArray doesn't have a shuffle parameter.
        # For shuffled data loading, consider using JAX's ShuffleSampler with a custom dataset.
    )

# Original:
# from.cached_image_folder import CachedImageFolder
# Convert to JAX equivalent (assuming similar API)
def jax_cached_image_folder():
    """JAX equivalent of CachedImageFolder (PyTorch)."""
    # Since JAX doesn't have direct equivalents, this is a placeholder.
    # Actual implementation would require JAX's imageio or similar, but for simplicity,
    # we'll assume a similar Dataset implementation with image data.
    from jax.data import Dataset, Functional
    class JaxCachedImageFolder(Dataset):
        def __init__(self, images_jax, ann_file_jax):
            # Assuming images_jax is a JAX array of images and ann_file_jax is a JAX array of labels
            self.images = images_jax
            self.labels = ann_file_jax
            self.index_map = jax.data.index_map.IndexMap.from_arrays(
                jnp.ones((len(images_jax),)), jax.data.ArrayArray([images_jax]))
        
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            return image, label
        
        def __len__(self):
            return jnp.size(self.images, axis=0)
    
    # Create a JAX Dataset from the JaxCachedImageFolder instance
    from jax.data import Dataset as JaxDataset
    return JaxDataset(JaxCachedImageFolder(images_jax, ann_file_jax))

def pytorch_to_jax_loader(data_loader_pytorch):
    """Convert PyTorch DataLoader to JAX equivalent."""
    # Assuming data_loader_pytorch is a JAX-compatible PyTorch DataLoader
    # This is a simplified example; actual conversion may vary based on data types
    return pytorch_to_jax_data_loader(data_loader_pytorch)

def pytorch_to_jax_model(config):
    """Convert PyTorch model to JAX model."""
    # Assuming config.MODEL is a PyTorch config; need to convert to JAX config
    # This is non-trivial and requires model serialization/deserialization
    # Placeholder for actual conversion
    jax_model =...  # Actual JAX model code here
    return jax_model

# Conversion of the sampler
def pytorch_to_jax_sampler