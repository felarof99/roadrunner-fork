import jax

jax.distributed.initialize()

import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
from absl import app, flags
from jax.sharding import Mesh, PartitionSpec as PS
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import llama3_jax
from llama3_jax.trainer_engine import jax_utils, trainer_lib

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_steps", 5000, "Number of training steps")
flags.DEFINE_integer("batch_size", 64, "Batch size for training")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")

# Use the MESH from jax_utils
MESH = jax_utils.MESH


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
    params = model.init(rng, jnp.ones([1, 28 * 28]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params,
                                         tx=tx)


def get_param_sharding_rules():
    return [
        ("dense1/kernel", PS("fsdp", "mp")),
        ("dense1/bias", PS(None)),
        ("dense2/kernel", PS("fsdp", "mp")),
        ("dense2/bias", PS(None)),
        ("final_layer/kernel", PS("fsdp", "mp")),
        ("final_layer/bias", PS(None)),
        # Add a catch-all rule at the end
        (".*", PS(None)),
    ]


def shard_params(params):
    rules = get_param_sharding_rules()
    return jax_utils.match_partition_rules(rules, params)


def train_step(state, batch, rng):

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch[0])
        loss = optax.softmax_cross_entropy(logits=logits,
                                           labels=batch[1]).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, rng, {'loss': loss}


def create_dataset(x, y, batch_size):
    # yapf: disable
    dataset = (
        tf.data.Dataset.from_tensor_slices((x.reshape(-1, 28 * 28),y))
        .shuffle(buffer_size=5000)
        .batch(batch_size).repeat()
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator())
    # yapf: enable
    return dataset

def get_batch(dataset):
    images, labels = next(dataset)
    return jnp.array(images), jnp.array(labels, dtype=jnp.int32)


def main(_):
    # Load and preprocess the MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    y_train = jax.nn.one_hot(y_train, 10)

    # Create dataset
    train_ds = create_dataset(x_train, y_train, FLAGS.batch_size)

    # Initialize the training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, FLAGS.learning_rate)

    # Shard the parameters
    state = state.replace(params=shard_params(state.params))

    # Get a batch for jitted_train_step setup
    batch = get_batch(train_ds)
    sharded_batch = jax.device_put(batch, NamedSharding(MESH, PS("dp", None)))

    # Setup jitted train step
    jitted_train_step = setup_jitted_train_step(state, sharded_batch)

    # Train
    for step in range(FLAGS.num_steps):
        batch = get_batch(train_ds)
        sharded_batch = jax.device_put(batch,
                                       NamedSharding(MESH, PS("dp", None)))
        state, _, metrics = jitted_train_step(state, sharded_batch,
                                              jax.random.PRNGKey(step))

        if step % 100 == 0:
            print(f"Step: {step}, Loss: {metrics['loss']:.4f}")


def setup_jitted_train_step(state, batch):
    dummy_rng = jax.random.PRNGKey(0)

    in_shardings = jax.eval_shape(lambda: (state, batch, dummy_rng))

    out_shardings = jax.eval_shape(lambda: train_step(state, batch, dummy_rng))

    return jax.jit(
        train_step,
        in_shardings=(
            jax_utils.match_partition_rules(get_param_sharding_rules(),
                                            in_shardings[0]),
            NamedSharding(MESH, PS("dp", None)),
            NamedSharding(MESH, PS()),
        ),
        out_shardings=(
            jax_utils.match_partition_rules(get_param_sharding_rules(),
                                            out_shardings[0]),
            NamedSharding(MESH, PS()),
            NamedSharding(MESH, PS()),
        ))


if __name__ == "__main__":
    app.run(main)
