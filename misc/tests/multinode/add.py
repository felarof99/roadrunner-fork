from functools import partial
import jax

jax.distributed.initialize()
print(
    f"Process index: {jax.process_index()}, Device count: {jax.device_count()}"
)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('x', 'y'))


def add_basic(a, b):
    c = a + b
    return c


add_basic_jitted = jax.jit(
    add_basic,
    in_shardings=(NamedSharding(mesh,
                                P('x', 'y')),
                  NamedSharding(mesh, P('x', 'y'))),
)

# Create and shard data on all processes
a = jnp.arange(8 * 16).reshape(8, 16)
b = jnp.arange(8 * 16).reshape(8, 16)

a_sharding = NamedSharding(mesh, P('x', 'y'))
b_sharding = NamedSharding(mesh, P('x', 'y'))

a = jax.device_put(a, a_sharding)
b = jax.device_put(b, b_sharding)

c = add_basic_jitted(a, b)
print("add_basic_jitted", c)
