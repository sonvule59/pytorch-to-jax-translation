import jax
import jax.numpy as jnp
from jax import vmap, grad
from jax.experimental import custom_jax_optimizer
import tensorflow as tf

def main(config, net_config, args):
    jax.set_print_backend('blitz')
    jax.config.update({'jax_backend': 'pytorch'})
    
    def load_data():
        train_data, val_data, test_data = tf.io.numpy_loadtxt(
            ['train_split.npy', 'val_split.npy', 'test_split.npy'])
        return jnp.array(train_data), jnp.array(val_data), jnp.array(test_data)
    
    train_data, val_data, test_data = load_data()
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data, train_data),
                                              num_parallel_workers=args.workers,
      predefined_shuffles=False)
    
    dataset = dataset.shuffle(tf.random.uniform(*dataset.feature_ranges)).batch(args.batch_size * 4)
    train_dataset = dataset.filter(lambda x: jnp.array(x[0]).shape == (1, args.batch_size * 4))
    val_dataset = dataset.take(1000).filter(lambda x: jnp.array(x[0]).shape == (1, args.batch_size))
    
    model = build_model(config, net_config)
    
    optimizer = custom_jax_optimizer.adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum)
    
    @grad
    def loss_fn(output, target):
        return tf.reduce_mean(tf.keras.losses.mse_loss(output, target))
    
    training_params = {'model': model, 'dataset': train_dataset,
                      'optimizer': optimizer, 'loss_fn': loss_fn,
                      'epochs': args.epochs,'save_dir': args.save_dir}
    
    result = training_params.run(steps=10000, epochs=args.epochs)
    print(f"Training completed: {result['steps']} steps, {result['epochs'] * result['steps'] / result['steps_per_epoch']} epochs")
    
    if args.val:
        val_params = {'model': model, 'dataset': val_dataset,
                     'optimizer': optimizer, 'loss_fn': loss_fn,
                     'epochs': 1,'save_dir': args.save_dir}
        val_result = val_params.run()
        print(f"Validation completed: {val_result['steps']} steps")
    
    if args.test:
        test_params = {'model': model, 'dataset': test_dataset,
                      'optimizer': optimizer, 'loss_fn': loss_fn,
                      'epochs': 1,'save_dir': args.save_dir}
        test_result = test_params.run()
        print(f"Testing completed: {test_result['steps']} steps")

def build_model(config, net_config):
    inputs = tf.keras.layers.Input(shape=(config['input_shape']))
    layers = []
    for layer in net_config['layers']:
        if isinstance(layer, str):
            layers.append(tf.keras.layers.GlobalAveragePooling1D(input_size=layer['input_size']))
        else:
            layers.append(
                tf.keras.layers.Conv2D(
                    filters=layer['filters'],
                    kernel_size=layer['kernel_size'],
                    activation=layer['activation']
                )
            )
    outputs = layers[-1](inputs)
    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    main(config, net_config, args)