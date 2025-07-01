20.10_1

Note: The code uses PyTorch's `nn.Module` and converts it to JAX's `jax.nn.Module`. Some PyTorch-specific functions may need to be replaced or adapted for JAX.

---
jax
import jax
import jax.numpy as jnp
from jax import config, vmap
from jax.vmap import vadd, vmul
from jax.experimental.run_eager import set_custom_eager_context
from jax import device, barrier

def sq(x):
    m2 = 1.0
    m1 = -20.0
    m0 = 50.0
    return m2 * x * x + m1 * x + m0

def qu(x):
    m3 = 10.0
    m2 = 5.0
    m1 = -20.0
    m0 = -5.0
    return m3 * x * x * x + m2 * x * x + m1 * x + m0

@vmap
def sq_jax(x):
    """JAX version of sq using vmap for vectorization."""
    return x * x * m2 + x * m1 + m0

@vmap
def qu_jax(x):
    """JAX version of qu using vmap for vectorization."""
    return x * x * x * m3 + x * x * m2 + x * m1 + m0

class Net(jax.nn.Module):
    def __init__(self, mode="sq"):
        super(Net, self).__init__()
        if mode == "square":
            self.mode = False
            self.param = jnp.array([1.0, -1.0], dtype=jnp.float32)
        else:
            self.mode = True
            self.param = jnp.array([1.0, -1.0, 1.0], dtype=jnp.float32)

    def forward(self, x):
        barrier()
        if not self.mode:
            return jax.vmap(vmul(vadd(x, jnp.array([0.0], dtype=jnp.float32)), self.param[0]))
        else:
            return vadd(
                vmul(vadd(vmul(vpow(x, jnp.array([1.0], dtype=jnp.float32)), self.param[0]), self.param[1]),
                     self.param[2]),
                x * self.param[1] + self.param[2]
            )

def train_jax(config):
    from jax import config as jax_config
    from jax import hwd as hwd_jax
    hwd_jax.init()
    device = device("cuda" if jax_config.is_jit_mode() and jax_config.data_num_threads > 0 else "cpu")
    mode = config["mode"]
    net = Net(mode)
    optimizer = jax.optim.SGD(
        net.parameters(),
        lr=config["lr"],
        use_jit=True
    )
    hwd_jax.set_per_device_params(net, root_rank=0)
    hwd_jax.broadcast_optimizer_state(optimizer, root_rank=0)

    num_steps = 5
    print(hwd_jax.size())
    jnp.random.seed(1 + hwd_jax.rank())
    x_max = config["x_max"]
    start = jax.time().new()
    for step in range(1, num_steps + 1):
        features = jnp.random.rand(1) * 2 * x_max - x_max
        if mode == "square":
            labels = sq_jax(features)
        else:
            labels = qu_jax(features)
        optimizer.apply_async(loss_function, args=(features, labels))
        barrier()
        if step % 100 == 0:
            loss = optimizer.get_result()
            print(f"Step {step}: Loss {loss}")
    total_time = start()
    avg_time = (jax.time().new() - total_time).item()
    print(f"Took {avg_time:.3f} s. Avg: {avg_time / num_steps:.3f} s.")

def loss_function(features, labels):
    net = Net(mode=config["mode"])
    net.set_state({**config["state"], "mode": config["mode"]})
    outputs = net(features)
    return jnp.mean(jax.vpow(outputs - labels, 2))

def tune_jax(hosts_per_trial, slots_per_host, num_samples, use_gpu, mode="square", x_max=1.):
    from jax import core as jax_core
    from jax import hwd as hwd_jax
    from jax import tune

    horovod_creator = tune.horovod.DistributedTrainableCreator(
        train_jax,
        use_gpu=use_gpu,
        num_hosts=hosts_per_trial,
        num_slots=