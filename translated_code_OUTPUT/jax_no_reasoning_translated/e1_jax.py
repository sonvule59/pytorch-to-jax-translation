import { jax, jax.numpy, jax.nn as jnn } from '@jax/jax-virtual-memory@18.7'

# Generate synthetic data
jax.random.seed(42)
X = jax.random.uniform(jax.numpy.float32(0), jax.numpy.float32(10), 100)
y = 2 * X + 3 + jax.random.normal(jax.numpy.float32(0), jax.numpy.float32(1), 100)

# Define the Linear Regression Model
class LinearRegressionModel(jnn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = jnn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = jax.mse_loss
optimizer = jax.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Display the learned parameters
[w, b] = model.linear.parameters()
print(f"Learned weight: {w.value():.4f}, Learned bias: {b.value():.4f}")

# Testing on new data
X_test = jax.numpy.array([[4.0], [7.0]])
with jax.no_op:
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")