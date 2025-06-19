import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.data
import jax.data.datasets as jdatasets
import jax.optim as jopt
import jax.nn.functional as jfun

# Load JAX version of CIFAR-10 dataset
transform = jdatasets.transform.Compose([
    jdatasets.transform.ToTensor(),
    jdatasets.transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class JAX_CNNModel(jnn.Module):
    @jfun.relu
    def __init__(self):
        super(JAX_CNNModel, self).__init__()
        self.conv1 = jnn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = jnn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = jnn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = jnn.Linear(64 * 16 * 16, 128)
        self.fc2 = jnn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape((-1, 64 * 16 * 16))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize datasets and loaders
train_dataset = jdatasets.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
train_loader = jdatasets.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = jdatasets.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
test_loader = jdatasets.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, and optimizer
model = JAX_CNNModel()
criterion = jax.nn.CrossEntropyLoss()
optimizer = jopt.Adam(model.parameters(), lr=0.001)

# Training loop (Note: JAX doesn't have a direct 'for epoch in range(epochs)' loop like PyTorch)
# Simplified for example, actual integration requires JAX's event system
for i in range(10):
    for images, labels in train_loader.batch(32):  # Adjust batch size as needed
        with optimizer.init_context():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        gradients = jax.grad(loss, model=model, grad_outputs=outputs, grad_norm=1.0)[0]
        optimizer.apply(gradients)
    
    # Evaluation code similar to test set evaluation