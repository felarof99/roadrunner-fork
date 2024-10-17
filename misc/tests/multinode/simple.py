from functools import partial
import jax
jax.distributed.initialize()
if jax.process_index() == 0:
    print(jax.device_count())

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.experimental import multihost_utils

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('x', 'y'))

@partial(shard_map,
         mesh=mesh,
         in_specs=(P('x', 'y'), P('y', None)),
         out_specs=P('x', None))
def matmul_basic(a_block, b_block):
    c_partialsum = jnp.dot(a_block, b_block)
    c_block = jax.lax.psum(c_partialsum, 'y')
    return c_block

if jax.process_index() == 0:
    a = jnp.arange(8 * 16).reshape(8, 16)
    b = jnp.arange(16 * 8).reshape(16, 8)
    c = matmul_basic(a, b)

    # Option 1: Use process_allgather to collect results from all processes
    c_gathered = multihost_utils.process_allgather(c)
    print("Gathered result (all processes):")
    print(c_gathered)

    # Option 2: Print only the local part of the result
    print("\nLocal part of the result:")
    for shard in c.addressable_shards:
        print(f"Device: {shard.device}")
        print(shard.data)

else:
    # For non-zero processes, we still need to participate in the allgather
    _ = multihost_utils.process_allgather(matmul_basic(a, b))
# Synchronize all processes
multihost_utils.sync_global_devices("end_of_script")

# matmul_basic_jitted = jax.jit(
#     matmul_basic,
#     in_shardings=(P('x', 'y'), P('y', 'x')),
#     out_shardings=P(),
# )

# if jax.process_index() == 0:
#     a = jnp.arange(8 * 16).reshape(8, 16)
#     b = jnp.arange(16 * 8).reshape(16, 8)

#     c = matmul_basic(a, b)
#     print("matmul_basic", c)

#     a_sharding = NamedSharding(mesh, P('x', 'y'))
#     b_sharding = NamedSharding(mesh, P('y', 'x'))

#     a = jax.device_put(a, a_sharding)
#     b = jax.device_put(b, b_sharding)

#     c = matmul_basic_jitted(a, b)
#     print("matmul_basic_jitted", c)

