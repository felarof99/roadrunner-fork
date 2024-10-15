import jax
import jax.numpy as jnp
from jax import random, jit, grad, pmap
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np

# Initialize JAX's distributed system
jax.distributed.initialize()

# Simple neural network
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

# Create fake data
def create_fake_data(num_rows, input_dim, key):
    x = random.normal(key, (num_rows, input_dim))
    y = jnp.sum(x, axis=1, keepdims=True) + random.normal(key, (num_rows, 1)) * 0.1
    return x, y

# Training step
@jit
def train_step(state, batch):
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, batch['x'])
        return jnp.mean((pred - batch['y']) ** 2)
    
    grad_fn = grad(loss_fn)
    grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads)

# Parallel training step
p_train_step = pmap(train_step)

# Main training loop
def train(num_epochs, batch_size, learning_rate, num_devices):
    # Initialize model and optimizer
    key = random.PRNGKey(0)
    model = SimpleNN()
    dummy_input = jnp.ones((1, 10))
    params = model.init(key, dummy_input)['params']
    
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    # Create fake data
    x, y = create_fake_data(128, 10, key)
    
    # Reshape data for multi-device training
    x = x.reshape(num_devices, -1, 10)
    y = y.reshape(num_devices, -1, 1)
    
    # Replicate state for each device
    state = jax.device_put_replicated(state, jax.local_devices()[:num_devices])

    # Training loop
    for epoch in range(num_epochs):
        state = p_train_step(state, {'x': x, 'y': y})
        
        if epoch % 10 == 0:
            loss = jnp.mean((state.apply_fn({'params': state.params}, x[0]) - y[0]) ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    # For 8 chips on a single host
    num_devices = 8
    
    # Uncomment the line below for 32 chips across multiple hosts
    # num_devices = jax.device_count()
    
    train(num_epochs, batch_size, learning_rate, num_devices)

