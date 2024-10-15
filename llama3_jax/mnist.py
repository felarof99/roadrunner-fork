import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
from absl import app, flags
from jax.sharding import Mesh, PartitionSpec as PS
from jax.sharding import NamedSharding

# Initialize JAX's distributed computing
jax.distributed.initialize()

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_steps", 5000, "Number of training steps")
flags.DEFINE_integer("batch_size", 64, "Batch size for training")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")

# Define the mesh for multi-TPU training
devices = jax.devices()
mesh = Mesh(devices, ('dp',))

class Model(nn.Module):
    def setup(self):
        self.dense1 = nn.Dense(1024)
        self.dense2 = nn.Dense(1024)
        self.dense3 = nn.Dense(1024)
        self.final_layer = nn.Dense(10)

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = nn.relu(self.dense3(x))
        x = self.final_layer(x)
        return x

def create_dataset(x, y, batch_size):
    dataset = (
        tf.data.Dataset
        .from_tensor_slices((x.reshape(-1, 28*28), y))  # Flatten the images
        .shuffle(buffer_size=5000)
        .batch(batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator())
    return dataset

def get_batch(dataset):
    images, labels = next(dataset)
    return jnp.array(images), jnp.array(labels, dtype=jnp.int32)

def create_train_state(rng, learning_rate):
    model = Model()
    params = model.init(rng, jnp.ones([1, 28*28]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def cross_entropy_loss(logits, labels):
        return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch[0])
        loss = cross_entropy_loss(logits, batch[1])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main(_):
    # Load and preprocess the MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0  # Flatten and normalize
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Create dataset
    train_ds = create_dataset(x_train, y_train, FLAGS.batch_size)

    # Initialize the training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, FLAGS.learning_rate)

    # Shard the initial state
    state = jax.device_put(state, NamedSharding(mesh, PS()))

    # Train
    for step in range(FLAGS.num_steps):
        batch = get_batch(train_ds)
        sharded_batch = jax.device_put(batch, NamedSharding(mesh, PS('dp')))
        state, loss = train_step(state, sharded_batch)

        if step % 100 == 0:
            print(f"Step: {step}, Loss: {loss:.4f}")

if __name__ == "__main__":
    app.run(main)
