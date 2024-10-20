from functools import partial
import jax

# Initialize distributed computation
jax.distributed.initialize()
print(f"Process index: {jax.process_index()}, Device count: {jax.device_count()}")

import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import multihost_utils

# Get the total number of hosts and devices per host
num_hosts = jax.process_count()
devices_per_host = jax.local_device_count()

# Ensure we have 2 hosts with 4 devices each
assert num_hosts == 2, "This script requires exactly 2 hosts."
assert devices_per_host == 4, "Each host must have exactly 4 devices."

# Get all devices and reshape them into (num_hosts, devices_per_host)
devices = np.array(jax.devices()).reshape((1, 8))

# Create a mesh with 'host' and 'device' axes
mesh = Mesh(devices, axis_names=('host', 'device'))

# Create host-local data on each host
host_id = jax.process_index()
a_host_local = np.arange(8 * 8).reshape(8, 8) 
b_host_local = np.arange(8 * 8).reshape(8, 8)  # Same data on all hosts

# Define PartitionSpec to partition over the 'host' axis
pspec = P('host')

# Convert host-local arrays to global arrays
a_global = multihost_utils.host_local_array_to_global_array(a_host_local, mesh, pspec)
b_global = multihost_utils.host_local_array_to_global_array(b_host_local, mesh, pspec)

# Define the computation using jax.jit with in_shardings and out_shardings
@partial(
    jax.jit,
    in_shardings=(NamedSharding(mesh, P('host')), NamedSharding(mesh, P('host'))),
    out_shardings=NamedSharding(mesh, P('host'))
)
def add_arrays(a, b):
    return a + b

# Perform the computation within the mesh context
with mesh:
    c_global = add_arrays(a_global, b_global)

# Convert the global array back to host-local arrays
c_host_local = multihost_utils.global_array_to_host_local_array(c_global, mesh, pspec)

# Print the host-local output
print(f"Host {host_id} local output:")
print(c_host_local)
