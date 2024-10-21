from functools import partial
import jax

jax.distributed.initialize()
print(f"Process index: {jax.process_index()}, Device count: {jax.device_count()}")

import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import multihost_utils

num_hosts = jax.process_count()
devices_per_host = jax.local_device_count()

assert num_hosts == 2, "This script requires exactly 2 hosts."
assert devices_per_host == 4, "Each host must have exactly 4 devices."

devices = np.array(jax.devices()).reshape((8, 1))
mesh = Mesh(devices, axis_names=('host', 'device'))

host_id = jax.process_index()

pspec = P('host')

a_host_local = np.arange(8 * 8).reshape(8, 8)
b_host_local = np.arange(8 * 8).reshape(8, 8)

# Attempt to convert host-local arrays to global arrays
a_global = multihost_utils.host_local_array_to_global_array(a_host_local, mesh, pspec)
b_global = multihost_utils.host_local_array_to_global_array(b_host_local, mesh, pspec)

@partial(
    jax.jit,
    in_shardings=(NamedSharding(mesh, P('host')), NamedSharding(mesh, P('host'))),
    out_shardings=NamedSharding(mesh, P('host'))
)
def add_arrays(a, b):
    return a + b

c_global = add_arrays(a_global, b_global)
print(jax.debug.visualize_array_sharding(a_global))

c_host_local = multihost_utils.global_array_to_host_local_array(c_global, mesh, pspec)
print(jax.debug.visualize_array_sharding(c_host_local))

print(f"Host {host_id} local output:")
print(c_host_local)
