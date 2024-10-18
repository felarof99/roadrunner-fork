from functools import partial
import jax

jax.distributed.initialize()
print(
    f"Process index: {jax.process_index()}, Device count: {jax.device_count()}"
)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.experimental import multihost_utils


devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('x', 'y'))


def add_basic(a, b):
    c = a + b
    return c

@partial(shard_map,
         mesh=mesh,
         in_specs=(P('x', 'y'), P('x', 'y')),
         out_specs=P('x', 'y'))
def add_basic(a_block, b_block):
    c_partialsum = a_block + b_block
    return c_partialsum

# Create and shard data on all processes
a = jnp.arange(8 * 16).reshape(8, 16)
b = jnp.arange(8 * 16).reshape(8, 16)

c = add_basic(a, b)
print("add_basic", c)
