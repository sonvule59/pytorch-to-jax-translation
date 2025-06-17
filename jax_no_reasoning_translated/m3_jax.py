import jax
import jax.numpy as jnp
from jax import layers, transforms as jax_transform, functional as jfx
from jax.nn import Module, Param
from jax.optim import Optimizer, GradientInference

# JAX configuration
jax.config.update_default(tracker="simple")
jax.auto_gpu()

# Define transforms
transform = jax_transform.Compose([
    jax_transform.ToTensor(),
    jax_transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets (example; actual JAX dataset loading differs)
# train_dataset =... (similar to Torchvision CIFAR10 but JAX-compatible)
# test_dataset =...

train_loader = jax.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = jax.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define model
class VanillaCNNModel(jax.nn.Module):
    @jax.nn.local_argument_scope(name="model")
    def __init__(self):
        super().__init__()
        self.conv1 = jax.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = jax.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = jax.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = jax.nn.Linear(64 * 16 * 16, 128)
        self.fc2 = jax.nn.Linear(128, 10)
        self.relu = jax.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model with different configs
models = []
vanilla = VanillaCNNModel()
models.append((vanilla, "Vanilla"))

init_type_dict = {"kaiming": jax.nn.init.kaiming_normal,
                  "xavier": jax.nn.init.xavier_normal,
                  "zeros": jax.nn.init.zeros,
                  "random": jax.nn.init.normal}
for t in ["kaiming", "xavier", "zeros", "random"]:
    m = vanilla.apply(init_type_dict[t])
    models.append((m, f"{t}_Config"))

# Training function (simplified, using GradientInference for autograd)
def train_test_loop(model, train_loader, test_loader, epochs=10):
    model.train()
    criterion = jax.nn.functional.cross_entropy
    optimizer = GradientInference(
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        init_func=jax.random_uniform_initializer(),
        opt_state=None
    )
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            with optimizer.begin_state():
                loss = criterion(model(images), labels)
            loss.update(complete=True)
            
        print(f"Training loss at epoch {epoch} = {loss.item()}")
    
    model.eval()
    correct = 0
    total = 0
    for images_test, labels_test in test_loader:
        predictions = model(images_test)
        _, preds = jfx.max(predictions, axis=-1)
        total += labels_test.size(0)
        correct += (preds == labels_test).sum().item()
    print(f"Test Accuracy = {(correct * 100)/total}")

# Initialize and configure models
for model, name in models:
    jax.config.update_all_default(jax.config.DEFAULT_JAX_CONFIG, 
                                {"model": {"initialized": False}})
    if not hasattr(model, "initialized"):  # Ensure initialization if not already done
        model.initialize()
    train_test_loop(model, train_loader, test_loader)