import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
from absl import app, flags
from jax.sharding import Mesh, PartitionSpec as PS
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils

# Initialize JAX's distributed computing
jax.distributed.initialize()

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_steps", 5000, "Number of training steps")
flags.DEFINE_integer("batch_size", 64, "Batch size for training")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")

# Define the mesh for multi-TPU training
devices = jax.devices()
device_count = len(devices)

if device_count == 1:
    device_mesh = mesh_utils.create_device_mesh((1, 1))
elif device_count == 4:
    device_mesh = mesh_utils.create_device_mesh((2, 2))
elif device_count == 8:
    device_mesh = mesh_utils.create_device_mesh((2, 4))
else:
    device_mesh = mesh_utils.create_device_mesh((1, device_count))

mesh = Mesh(devices=device_mesh, axis_names=('dp', 'mp'))

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1024, name='dense1')(x)
        x = nn.relu(x)
        x = nn.Dense(1024, name='dense2')(x)
        x = nn.relu(x)
        x = nn.Dense(10, name='final_layer')(x)
        return x

def create_train_state(rng, learning_rate):
    model = Model()
    params = model.init(rng, jnp.ones([1, 28*28]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

def get_param_sharding_rules():
    return {
        'dense1': PS('mp', None),
        'dense2': PS('mp', None),
        'final_layer': PS('mp', None)
    }

def shard_params(params):
    rules = get_param_sharding_rules()
    return jax.tree_map(
        lambda p, rule: jax.device_put(p, NamedSharding(mesh, rule)),
        params, rules
    )

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

def create_dataset(x, y, batch_size):
    dataset = (
        tf.data.Dataset
        .from_tensor_slices((x.reshape(-1, 28*28), y))
        .shuffle(buffer_size=5000)
        .batch(batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator())
    return dataset

def get_batch(dataset):
    images, labels = next(dataset)
    return jnp.array(images), jnp.array(labels, dtype=jnp.int32)

def main(_):
    # Load and preprocess the MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    y_train = jax.nn.one_hot(y_train, 10)

    # Create dataset
    train_ds = create_dataset(x_train, y_train, FLAGS.batch_size)

    # Initialize the training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, FLAGS.learning_rate)

    # Shard the parameters
    state = state.replace(params=shard_params(state.params))

    # Train
    for step in range(FLAGS.num_steps):
        batch = get_batch(train_ds)
        sharded_batch = jax.device_put(batch,
                                       NamedSharding(mesh, PS('dp', None)))
        state, loss = train_step(state, sharded_batch)
        
        if step % 100 == 0:
            print(f"Step: {step}, Loss: {loss:.4f}")

if __name__ == "__main__":
    app.run(main)
