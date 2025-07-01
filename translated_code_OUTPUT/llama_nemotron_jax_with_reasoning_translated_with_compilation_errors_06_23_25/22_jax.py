import jax
import jax.numpy as jnp
from jax import concat, maxpool, sequential_context, convolution, batch_norm, activation
from jax import compactor_fused
from jax.experimental.build_devices import JITCompiler
from jax import Primitives as jpr

class Inception(jax.macro):
    @staticmethod
    def construct(in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        b1 = sequential_context([
            convolution(jnp.ones((1, n1x1)), kernel_size=1, stride=1, padding=0, output_shape=(in_planes, n1x1)),
            batch_norm(output_shape=(in_planes, n1x1), range=n1x1),
            activation('relu'),
        ])
        b2 = sequential_context([
            convolution(jnp.ones((1, n3x3red)), kernel_size=1, stride=1, padding=0, output_shape=(in_planes, n3x3red)),
            batch_norm(output_shape=(in_planes, n3x3red), range=n3x3red),
            activation('relu'),
            convolution(jnp.ones((1, n3x3)), kernel_size=3, stride=1, padding=1, output_shape=(in_planes, n3x3)),
            batch_norm(output_shape=(in_planes, n3x3), range=n3x3),
            activation('relu'),
        ])
        b3 = sequential_context([
            convolution(jnp.ones((1, n5x5red)), kernel_size=1, stride=1, padding=0, output_shape=(in_planes, n5x5red)),
            batch_norm(output_shape=(in_planes, n5x5red), range=n5x5red),
            activation('relu'),
            convolution(jnp.ones((1, n5x5)), kernel_size=3, stride=1, padding=1, output_shape=(in_planes, n5x5)),
            batch_norm(output_shape=(in_planes, n5x5), range=n5x5),
            convolution(jnp.ones((1, n5x5)), kernel_size=3, stride=1, padding=1, output_shape=(in_planes, n5x5)),
            batch_norm(output_shape=(inplanes, n5x5), range=n5x5),
            activation('relu'),
        ])
        b4 = sequential_context([
            maxpool(kernel_size=3, strides=(3,1), padding='same'),
            convolution(jnp.ones((1, in_planes)), kernel_size=1, stride=1, padding=0, output_shape=(in_planes, pool_planes)),
            batch_norm(output_shape=(in_planes, pool_planes), range=pool_planes),
            activation('relu'),
        ])
        return concat(['b1', 'b2', 'b3', 'b4'], axis=1)

class GoogLeNet(jax.macro):
    @staticmethod
    def compile():
        jpr.register_jit_function('pre_layers', 
            inputs=(jnp.int64, jnp.float32, jnp.int64),
            outputs=(jnp.int64, jnp.float32),
            returns=(jpr.compact_function('pre_layers',...),),
        )
        #... define a3, b3, maxpool, a4, b4, c4, d4, e4, a5, b5, avgpool, linear similarly
        # For brevity, key parts are defined. Actual compilation requires defining each layer with JAX primitives.
        # This is a simplified example. Full compilation needs detailed definitions for each op.

        # Example compilation for a3 layer
        def a3(in_planes, _):
            b1 = jpr.compact_function('conv2d_1x1', 
                inputs=(in_planes, jnp.float32),
                outputs=(in_planes, 64),
                kernel_size=1,
                stride=1,
                padding=0,
                range=64,
                name='b1',
                activation=jpr.compact_activation('relu'),
            )
            #... similarly define other blocks
            return jpr.compact_function('concat', 
                inputs=[b1, b2, b3, b4],
                outputs=(in_planes,),
                name='concat_after_inception',
            )

    def forward(self, x):
        out = jpr.compact_function('pre_layers', 
            inputs=(self.input_shape, x),
            outputs=(self.output_shape,),
            returns=(self.pre_layers_fn, self.a3_fn, self.b3_fn,...),
        )
        #... processing through layers
        # Example: after a3 and b3, apply maxpool
        out = maxpool(out, kernel_size=3, strides=(3,2), padding='same')
        #