import os
import sys
import jax
import numpy as np
import re
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

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import llama3_jax
from llama3_jax.trainer_engine import jax_utils, trainer_lib


class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1024, name='dense1')(x)
        x = nn.relu(x)
        x = nn.Dense(1024, name='dense2')(x)
        x = nn.relu(x)
        x = nn.Dense(10, name='final_layer')(x)
        return x


def train_step(state, batch):
    images, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {'loss': loss}


def create_dataset(x, y, batch_size):
    # yapf: disable
    dataset = (
        tf.data.Dataset.from_tensor_slices((x.reshape(-1, 28 * 28),y))
        .shuffle(buffer_size=5000)
        .batch(batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator())
    # yapf: enable
    return dataset


def get_batch(dataset):
    images, labels = next(dataset)
    return jnp.array(images), jnp.array(labels, dtype=jnp.int32)


model = Model()
params = model.init(jax.random.PRNGKey(0), jnp.ones([1, 28 * 28]))['params']
tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=tx)


# Load and preprocess the MNIST dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
y_train = jax.nn.one_hot(y_train, 10)

train_ds = create_dataset(x_train, y_train, batch_size=64)

# create mesh
devices = np.array(jax.devices()).reshape((8, 1))
mesh = Mesh(devices, axis_names=('x', 'y'))

# state sharding
state_sharding_rules = [
    ("dense1/kernel", PS("x", "y")),
    ("dense1/bias", PS(None)),
    ("dense2/kernel", PS("x", "y")),
    ("dense2/bias", PS(None)),
    ("final_layer/kernel", PS("x", "y")),
    ("final_layer/bias", PS(None)),
    # Add a catch-all rule at the end
    (".*", PS(None)),
]
state.params = jax_utils.match_partition_rules(state_sharding_rules,
                                               state.params, mesh)

images, labels = get_batch(train_ds)

jitted_train_step = jax.jit(train_step,
                            in_shardings=(PS('x', 'y'), PS('x', 'y')),
                            out_shardings=PS('x', 'y'))
