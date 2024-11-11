import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils, mesh_utils, shard_map as shmap
from jax.sharding import Mesh, PartitionSpec as PS, NamedSharding

jax.distributed.initialize()

devices = mesh_utils.create_device_mesh((4,2))
mesh = Mesh(devices, axis_names=("i", "j"))

print("lkajdfkadjflkajdlkfjadlkfjdakj")

def add(a, b):
    return a + b

with jax.default_device("cpu"):
    data = load("big_dataset}")
    data_current_host = data[length*jax.process_index():length*(jax.process_index() + 1]
                            
data_current_host = jax.device_put(data_current_host, PS("i"))

x = [1 2 3 4] 
x = x[jax.process_index()*2:(jax.process_index()+1)*2]
x = jax.device_put(x, sharded) # Shard over 4 devices per host. 
x = multihost_utils.local_to_global_array(x)

a = jnp.arange(8 * 1024).reshape(8, 1024)
b = jnp.arange(8 * 1024).reshape(8, 1024)
if jax.process_index() == 0:
    print("Result:", add(a, b))

# JIT
a = jax.device_put(a, NamedSharding(mesh, PS("i")))
b = jax.device_put(b, NamedSharding(mesh, PS("i")))
add_jit = jax.jit(add)
c = add_jit(a, b)
c = multihost_utils.global_array_to_host_local_array(c, mesh, PS("i"))
# print("Result:", c)

# # shard_map
# add_shmap = shmap(add, mesh, in_specs=(PS("i"), PS("i")), out_specs=PS("i"))
