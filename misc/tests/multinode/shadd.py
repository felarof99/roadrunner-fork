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


devices = mesh_utils.create_device_mesh((8,))
mesh = Mesh(devices, axis_names=('i',))


def add_basic(a, b):
    c = a + b
    return c

@partial(shard_map,
         mesh=mesh,
         in_specs=(P('i'), P('i')),
         out_specs=P('i'))
def add_basic(a_block, b_block):
    c_partialsum = a_block + b_block
    return c_partialsum

@partial(jax.jit,
         in_shardings=NamedSharding(mesh, P('i')),
         out_shardings=NamedSharding(mesh, P()))
def get_c(c):
    return c


# Create and shard data on all processes
a = jnp.arange(8 * 8).reshape(8, 8)
b = jnp.arange(8 * 8).reshape(8, 8)
c = add_basic(a, b)
print("add done")
c = get_c(c)
print(c)
print(jax.debug.visualize_array_sharding(c))
