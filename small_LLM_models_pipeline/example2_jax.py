import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + np.random.randn(100, 1)  # Linear relationship with noise

class LearnedSiLUFunction(jax.api.JaxFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x, slope):
        # Save the input tensor and slope for backward computation
        # ctx.save_for_backward(x)
        # ctx.slope = slope
        # return slope * x * torch.sigmoid(x)
        return slope * x * jnp.sigmoid(x)

    def backward(self, grad_output):
        # Retrieve the input and slope saved in the forward pass
        # x, = ctx.saved_tensors
        # slope = ctx.slope
        # sigmoid_x = torch.sigmoid(x)
        # Compute the gradient with respect to input (x)
        # grad_input = grad_output * slope * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
        # Compute the gradient with respect to slope
        # grad_slope = grad_output * x * sigmoid_x
        # return grad_input, grad_slope
        x = jnp.array([1.0])
        slope = jnp.array([1.0])
        sigmoid_x = jnp.sigmoid(x)
        grad_input = grad_output * slope * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
        grad_slope = grad_output * x * sigmoid_x
        return grad_input, grad_slope

# Define the Linear Regression Model
class LinearRegressionModel(jax.api.JaxModule):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = jnp.array(slope)

    def forward(self, x):
        # Use the custom LearnedSiLUFunction
        return LearnedSiLUFunction.apply(x, self.slope)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = lambda y_pred, y_true: jnp.mean((y_pred - y_true) ** 2)
optimizer = jax.experimental.optimizers.Adam(learning_rate=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    grads = jax.grad(loss)(model, X)
    # grads = jax.grad(loss)(model, X)
# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    grads = jax.grad(loss)(model, X)
    # grads = jax.grad(loss)(model, X)
    # optimizer.update(grads, model)
    # model = optimizer.update(grads, model)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Display the learned parameters
[w, b] = model.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.disable_jit():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

