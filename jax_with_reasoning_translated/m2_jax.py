import jax
from jax import numpy as jnp
from jax.vjp import prange
from jax.experimental.build_lib import custom_jax_library
import jax.nn as jnn
import jax.optim as jopt
import jax.data
import jax.data.datasets as jdatasets
import jax.data.preprocessing as jpreproc

# Custom JAX imports
custom_jax_library.migrate()

# Define JAX transforms
transform = jpreproc.Compose([
    jpreproc.ToTensor(),
    jpreproc.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load JAX CIFAR-10 dataset
train_dataset = jdatasets.cifar10(root='data', train=True, download=True, transform=transform)
train_loader = jdata.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = jdatasets.cifar10(root='data', train=False, download=True, transform=transform)
test_loader = jdata.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class CNNModel(jnn.Module):
    @jax.jit_pandas
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = jnn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = jnn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = jn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = jnn.Linear(64 * 16 * 16, 128)
        self.fc2 = jnn.Linear(128, 10)
        self.relu = jax.nn.relu

    @jax.jit_pandas
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = jax.jax.nn.CrossEntropyLoss()
optimizer = jopt.Adam(model.parameters(), lr=0.001)

# Training loop with JAX
@prange(1000)
def train_step(data):
    jimages, jlabels = data
    joutputs = model(jimages)
    jloss = criterion(joutputs, jlabels)

    joptimizer = optimizer
    joptimizer.zero_grad()
    jloss.backward()
    joptimizer.step()

train_loader = jdata.Iterators([train_dataset], batch_size=64)[0]

for epoch in range(10):
    for data in train_loader:
        train_step(data)
    print(f"Epoch [{epoch + 1}/{10}], Loss: {jloss.item():.4f}")

# Evaluate on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        jimages, jlabels = data
        joutputs = model(jimages)
        _, predicted = joutputs.argmax()
        total += jlabels.size()
        correct += (predicted == jlabels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")