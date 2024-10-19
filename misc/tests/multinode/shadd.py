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
from jax.experimental.multihost_utils import broadcast_one_to_all


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

if jax.process_index() == 0:
    # Create arrays only on process 0
    a = jnp.arange(8 * 8).reshape(8, 8)
    b = jnp.arange(8 * 8).reshape(8, 8)
else:
    # On other processes, initialize `a` and `b` as None
    a = None
    b = None

# Broadcast the arrays from process 0 to all processes
a = broadcast_one_to_all(a)
b = broadcast_one_to_all(b)

# Proceed with the computation
c = add_basic(a, b)

if jax.process_index() == 0:
    print("add done")

# Gather the sharded array from all processes
c = get_c(c)
c_gathered = multihost_utils.process_allgather(c)

# Only process 0 prints the gathered array
if jax.process_index() == 0:
    print(c_gathered)
    print(jax.debug.visualize_array_sharding(c_gathered))
