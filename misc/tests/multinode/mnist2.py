import os
import functools
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
from jax.experimental import multihost_utils

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import llama3_jax
from llama3_jax.trainer_engine import jax_utils, trainer_lib


def create_dataset(x, y, batch_size):
    # yapf: disable
    dataset = (
        tf.data.Dataset.from_tensor_slices((x.reshape(-1, 28 * 28),y))
        .batch(batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator())
    # yapf: enable
    return dataset


def get_batch(dataset):
    images, labels = next(dataset)
    return jnp.array(images), jnp.array(labels, dtype=jnp.int32)


class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
        # x = nn.Dense(1024, name='dense1', use_bias=False)(x)
        # x = nn.relu(x)
        # x = nn.Dense(1024, name='dense2', use_bias=False)(x)
        # x = nn.relu(x)
        x = nn.Dense(10, name='final_layer', use_bias=False)(x)
        return x


def train_step(state, images, labels):

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {'loss': loss}


def create_train_state(model):
    params = model.init(jax.random.PRNGKey(0), jnp.ones([1,
                                                         28 * 28]))['params']
    tx = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(apply_fn=model.apply,
                                          params=params,
                                          tx=tx)
    return state


STATE_SHARDING_RULES = [
    # ("dense1/kernel", PS("x", "y")),
    # ("dense2/kernel", PS("x", "y")),
    ("final_layer/kernel", PS("batch", "mp")),
    # Add a catch-all rule at the end
    (".*", PS(None)),
]
DEVICES = np.array(jax.devices()).reshape((8, 1))
MESH = Mesh(DEVICES, axis_names=('batch', 'mp'))

model = Model()

# Get state shapes and partition them -- you'll use that partitioned shapes in jax.jit decorator.
state_shapes = jax.eval_shape(
    functools.partial(create_train_state, model=model))
state_shapes_partitioned = jax_utils.match_partition_rules(
    STATE_SHARDING_RULES, state_shapes.params, MESH)

# Now, actually create the train state and partition it.
# (even if you don't partition it, when passing to jax.jit, it'll get partitioned)
state = create_train_state(model)
state = state.replace(params=jax_utils.match_partition_rules(
    STATE_SHARDING_RULES, state.params, MESH))

# Create the dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
y_train = jax.nn.one_hot(y_train, 10)

# The same train_ds would exist on all the hosts
train_ds = create_dataset(x_train, y_train, batch_size=16)

images, labels = get_batch(train_ds)
images_global = multihost_utils.host_local_array_to_global_array(
    images, MESH, PS('batch', 'mp'))
labels_global = multihost_utils.host_local_array_to_global_array(
    labels, MESH, PS('batch', 'mp'))

# def train_step(state, images, labels):
# return state, {'loss': loss}

jitted_train_step = jax.jit(
    train_step,
    in_shardings=(state_shapes_partitioned,
                  NamedSharding(MESH, PS('batch', 'mp')),
                  NamedSharding(MESH, PS('batch', 'mp'))),
    out_shardings=(state_shapes_partitioned, NamedSharding(MESH, PS())))

state, metrics = jitted_train_step(state, images_global, labels_global)
print(metrics)
