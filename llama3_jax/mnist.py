import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
from absl import app, flags

# Initialize JAX's distributed computing
jax.distributed.initialize()

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_steps", 5000, "Number of training steps")
flags.DEFINE_integer("batch_size", 64, "Batch size for training")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")

class Model(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(16, kernel_size=(3, 3), strides=(2, 2))
        self.conv2 = nn.Conv(8, kernel_size=(3, 3), strides=(2, 2))
        self.conv3 = nn.Conv(4, kernel_size=(3, 3), strides=(2, 2))
        self.conv4 = nn.Conv(1, kernel_size=(3, 3), strides=(2, 2))
        self.final_layer = nn.Dense(10)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = nn.relu(self.conv4(x))
        x = x.reshape((x.shape[0], -1))
        x = self.final_layer(x)
        return x

def create_dataset(x, y, batch_size):
    dataset = (
        tf.data.Dataset
        .from_tensor_slices((x, y))
        .shuffle(buffer_size=5000)
        .batch(batch_size)
        .repeat()  # This makes the dataset repeat indefinitely
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator())
    return dataset

def get_batch(dataset):
    images, labels = next(dataset)
    return jnp.expand_dims(jnp.array(images), axis=-1), jnp.array(labels, dtype=jnp.int32)

def create_train_state(rng, learning_rate):
    model = Model()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch[0])
        loss = optax.softmax_cross_entropy(logits=logits, labels=batch[1]).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train(state, train_ds, num_steps):
    for step in range(num_steps):
        train_batch = get_batch(train_ds)
        state, loss = train_step(state, train_batch)
        if step % 500 == 0:
            print(f"Step: {step}, Loss: {loss:.4f}")

def main(_):
    # Load and preprocess the MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Create dataset
    train_ds = create_dataset(x_train, y_train, FLAGS.batch_size)

    # Initialize the training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, FLAGS.learning_rate)

    # Train
    train(state, train_ds, FLAGS.num_steps)

if __name__ == "__main__":
    app.run(main)
