import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt

key = random.PRNGKey(42)

X = jnp.array([jnp.random.uniform(0, 10) for _ in range(100)])
y = 3 * X + 5 + jnp.random.normal(0, 1, (100, 1))

def model(params, x):
    return jnp.dot(x, params) + params[1]

def loss(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

initial_params = jnp.array([0., 0.])

optimizer = optax.sgd(learning_rate=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    current_loss = loss(initial_params, X, y)
    grads = jax.grad(loss)(initial_params, X, y)
    updated_params = optimizer.update(grads, initial_params)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {current_loss:.4f}")
    initial_params = updated_params

plt.scatter(X, y, label='Data points')
plt.plot(X, model(initial_params, X), color='red', label='Fitted line')
plt.legend()
plt.show()