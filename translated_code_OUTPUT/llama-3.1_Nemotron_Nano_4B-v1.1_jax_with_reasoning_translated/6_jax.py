__import__('jax')__.jax.__version__
</think>
import jax
import jax.numpy as jnp
from jax import devices
from jax.vmap import vmap
from jax.nn import layers as jnn
from jax.optim import root_inertia
from jax.data import Dataset, DataLoader

def seed(args):
    jax.random.seed(args.seed)
    if devices.JAX(device_type='cpu'):
        jnp.set_seed(args.seed)
    if devices.cuda():
        jax.device_set(device_type='cuda')

def keywords(args):
    return {'num_workers': 2, 'pin_memory': True}

def load_train(args, kwargs):
    class MNISTDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return jnp.array(self.data)[idx].astype('float32') / 255.0

        def __len__(self):
            return len(self.data)

    train_data, val_data = jnp.array_of_arrays_from_iterating_list(
        datasets.MNIST(args.train_path, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), split=jnp.array([True, False]))

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    return train_loader, val_loader

class Net(jnn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = layers.Conv2d(1, 10, kernel_size=5)
        self.conv2 = layers.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = layers.Dropout2d()
        self.fc1 = layers.Linear(320, 50)
        self.fc2 = layers.Linear(50, 10)

    def forward(self, x):
        x = jnp.max_pool2d(self.conv1(x), 2)
        x = jnp.relu(jnp.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape((-1, 320))
        x = jnp.relu(self.fc1(x))
        x = jnp.dropout(x, training=True)
        x = self.fc2(x)
        return jnp.log_softmax(x, axis=1)

def get_model(args):
    model = Net()
    if devices.cuda():
        jax.device_set(device_type='cuda')
        model.set_device('cuda')
    optimizer = root_inertia(root_config={
        'device': 'cpu' if devices.cpu() else 'cuda',
        'initializers': {'weights': 'he_initializers'},
        'updates': {'accumulators': model.parameters()}
    })
    return model, optimizer

def train(args, epoch, model, train_loader, optimizer):
    model.set_innitrain(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        model.set_device('cpu' if devices.cpu() else 'cuda')
        data, target = jnp.array_of_arrays_from_iterating_list(
            [data.to(model.device if devices.cuda() else 'cpu'), target.to(model.device if devices.cuda() else 'cpu')])
        loss = jnp.mean(jnp.square(jax.grad(model)(data, target)).sum())
        loss.backward()
        optimizer.apply_jit()
        optimizer.step()

def test(args, epoch, model, test_loader, optimizer):
    model.set_innitrain(False)
    test_loss = 0.0
    correct = 0
    for data, target in test_loader:
        data, target = jnp.array_of_arrays_from_iterating_list(
            [data.to(model.device if devices.cuda() else 'cpu'), target.to(model.device if devices.cuda() else 'cpu')])
        output = model(data)
        test_loss += jnp.mean(jnp.square(jax.grad(model)(data, target)).sum())
        pred = output.argmax(axis=1)
        correct += (pred == target).sum()
    test_loss /= jnp.array(len(test_loader.dataset))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct/len(test_loader.dataset):.0f}%')

def main():
    jax.macrodevice('auto')
    parser = jax.command.CommandParser(description='PyTorch MNIST Example in JAX')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                     help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10