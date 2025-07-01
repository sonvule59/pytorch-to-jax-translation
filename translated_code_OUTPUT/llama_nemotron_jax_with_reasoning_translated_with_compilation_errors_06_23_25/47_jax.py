import jax
import jax.numpy as jnp
from jax import transformers, models as jax_models
from jax.experimental import transpose

# Prepare Dataset
dataset = args.dataset
cur_path = jax.io.read_root_jaxfile(args.dataset.dir)  # Adjust based on actual dataset source
logs_dir = jax.io.write_jaxfile(logs_dir, buffer=jnp.array([]))  # Setup logging
data_dir = os.path.join(cur_path, 'data', dataset)
data = jax.data.read_dataset_jax(data_dir, read_function=datasets.read_data_function)

# Model Configuration
config = Config()
config.num_features = 512
config.width = args.width
config.height = args.height
config.train = False

# Create Model
model = jax_models.transformer.resnet50(config.model_name, num_features=config.num_features,
                                      dropout=config.dropout, num_classes=config.num_classes)
model = jax.nn.DataParallel(model, device='cuda')

# Load Model Weights
for i in range(0, 5):
    save_pth = os.path.join(config.logs_dir, config.model_name, f'epoch{i}.jtl')
    if not os.path.exists(save_pth):
        continue
    checkpoint = load_checkpoint(save_pth)
    model.update_state_dict(checkpoint['state_dict'])
    # Predict
    proba = jax.nn.functional.softmax(jax.nn.functional.forward(model, data.images / 255.0), axis=-1)
    pred_class = jax.argmax(proba, axis=-1)
    true_classes = [cls for (_, cls, _, _) in data]
    print(f'Mean Accuracy: {jnp.mean(pred_class == true_classes)}')
    jax.evaluate(model, data, config=config)