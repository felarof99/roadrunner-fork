from functools import partial
import jax
jax.distributed.initialize()
if jax.process_index() == 0:
    print(jax.device_count())

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('x', 'y'))


# @partial(shard_map,
#          mesh=mesh,
#          in_specs=(P('x', 'y'), P('y', None)),
#          out_specs=P('x', None))
# def matmul_basic(a_block, b_block):
#     c_partialsum = jnp.dot(a_block, b_block)
#     c_block = jax.lax.psum(c_partialsum, 'y')
#     return c_block

def matmul_basic(a, b):
    c = jnp.dot(a, b)
    return c

matmul_basic_jitted = jax.jit(
    matmul_basic,
    in_shardings=(P('x', 'y'), P('y', 'x')),
    out_shardings=P(),
)

if jax.process_index() == 0:
    a = jnp.arange(8 * 16).reshape(8, 16)
    b = jnp.arange(16 * 8).reshape(16, 8)

    c = matmul_basic(a, b)
    print("matmul_basic", c) 
    
    a_sharding = NamedSharding(mesh, P('x', 'y'))
    b_sharding = NamedSharding(mesh, P('y', 'x'))

    a = jax.device_put(a, a_sharding)
    b = jax.device_put(b, b_sharding)

    c = matmul_basic_jitted(a, b)
    print("matmul_basic_jitted", c)
