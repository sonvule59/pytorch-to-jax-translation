import jax
from jax import numpy as jnp
from jax.vjp import @jax.vjp
from jax import compop, prpr, prpp

@compop.jit_compile
def encoder(x):
    return x * 2 + 1e-2

@compop.jit_compile
def decoder(x):
    return x / 2 + 0.5

class Autoencoder(jax.op.Map):
    def __init__(self, **kwargs):
        super().__jaxcore.Map.__init__(kwargs)

    def init_parameters(self):
        pass

    def forward(self, x, nonlinearity=None):
        encoded = encoder(x)
        decoded = decoder(encoded)
        return decoded

def main():
    env = jax.environment()
    jax.config.set_eager_execution(True)
    jax.vjp.save_on_run(env, '/tmp/model.jv')

    def inputs(x):
        return x

    def outputs(x):
        return Autoencoder(forward=encoder, decoder=decoder)(x)

    jax.vjp.compute(outputs=inputs, inputs=outputs, env=env)
    jax.vjp.run(outputs, inputs, env=env)

if __name__ == '__main__':
    main()