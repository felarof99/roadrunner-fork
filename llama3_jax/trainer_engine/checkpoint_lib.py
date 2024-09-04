"""Checkpoints JAX models efficiently, will replace with Orbax soon!"""
import asyncio
import os
import shutil

import aiofiles
import flax
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from flax.serialization import (from_bytes, from_state_dict, to_bytes,
                                to_state_dict)
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS
from ml_collections import ConfigDict

from . import jax_utils, utils


async def async_copy_directory(src, dst):
    """Asynchronously copy a directory from src to dst."""
    async def copy_file(src, dst):
        async with aiofiles.open(src, 'rb') as fsrc:
            async with aiofiles.open(dst, 'wb') as fdst:
                await fdst.write(await fsrc.read())

    async def copy_item(src, dst):
        # print(f"Copying {src} to {dst}")
        if os.path.isfile(src):
            await copy_file(src, dst)
        elif os.path.isdir(src):
            print("skipping directory ", src)

    tasks = []
    for item in os.listdir(src):
        if item == "tmp":
            continue  # Skip the tmp directory
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        tasks.append(copy_item(s, d))

    await asyncio.gather(*tasks)

def copy_directory(src, dst):
    """Synchronous wrapper for async_copy_directory."""
    print("copying directory ", src, " to ", dst)
    loop = asyncio.get_event_loop()  # Get the current event loop
    if loop.is_running():
        loop.create_task(async_copy_directory(src, dst))  # Schedule the coroutine on the running loop
    else:
        loop.run_until_complete(async_copy_directory(src, dst))  # This is for non-interactive environments
    print(f"Copied all files from {src} to {dst}")

def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'bfloat16': jnp.bfloat16,
        'fp16': jnp.float16,
        'float16': jnp.float16,
        'fp32': jnp.float32,
        'float32': jnp.float32,
        'fp64': jnp.float64,
        'float64': jnp.float64,
    }[dtype]


def tree_apply(fns, tree):
    """ Apply a pytree of functions to the pytree. """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def float_tensor_to_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = get_float_dtype_by_name(dtype)
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
    """Creates pytree of sharding and gathering functions from pytree of partition specs."""

    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):

        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype',
                                                       None) in float_dtypes:
                # Convert all float tensors to the same dtype
                return tensor.astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return tensor.astype(dtype_spec.dtype)
            return tensor

        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        # Check if partition_spec is already a NamedSharding
        if isinstance(partition_spec, NamedSharding):
            out_sharding = partition_spec
        else:
            out_sharding = NamedSharding(jax_utils.MESH, partition_spec)

        jax_shard_function = jax.jit(make_to_dtype_fn(dtype_spec),
                                     in_shardings=None,
                                     out_shardings=out_sharding)

        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()

        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        # Check if partition_spec is already a NamedSharding
        if isinstance(partition_spec, NamedSharding):
            in_sharding = partition_spec
        else:
            in_sharding = NamedSharding(jax_utils.MESH, partition_spec)

        jax_gather_fn = jax.jit(make_to_dtype_fn(dtype_spec),
                                in_shardings=in_sharding,
                                out_shardings=None)

        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))

        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs,
                                           dtype_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs,
                                            dtype_specs)
    return shard_fns, gather_fns


class Checkpointer(object):
    """Checkpoints JAX models efficiently."""

    @staticmethod
    def get_default_config(updates=None):
        config = utils.create_config_dict(float_dtype="bf16",
                                          save_optimizer_state=False)

        if updates is not None:
            config = utils.update_config_dict(config, updates)
        return config

    def __init__(self, config, checkpoint_dir, enable_checkpointer=True):
        self.config = self.get_default_config(config)
        self.checkpoint_dir = checkpoint_dir
        self.enable_checkpointer = enable_checkpointer

    def save_checkpoint_simple(self, train_state, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        with utils.open_file(path, "wb") as fout:
            fout.write(
                flax.serialization.msgpack_serialize(
                    train_state.params["params"], in_place=True))

    @staticmethod
    def save_train_state_to_file(train_state, path, gather_fns=None, float_dtype=None):
        train_state = to_state_dict(train_state)
        packer = msgpack.Packer()
        flattend_train_state = flatten_dict(train_state)
        if gather_fns is not None:
            gather_fns = flatten_dict(to_state_dict(gather_fns))

        with utils.open_file(path, "wb") as fout:
            for key, value in flattend_train_state.items():
                if gather_fns is not None:
                    value = gather_fns[key](value)
                value = float_tensor_to_dtype(value, float_dtype)
                fout.write(packer.pack((key, to_bytes(value))))

    @staticmethod
    def load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None):
        if shard_fns is not None:
            shard_fns = flatten_dict(to_state_dict(shard_fns))
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)
        flattend_train_state = {}
        with utils.open_file(path) as fin:
            # 83886080 bytes = 80 MB, which is 16 blocks on GCS
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=0)
            for key, value in unpacker:
                key = tuple(key)
                if remove_dict_prefix is not None:
                    if key[:len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix):]
                    else:
                        continue

                tensor = from_bytes(None, value)
                if shard_fns is not None:
                    # Adjust the key to match the structure in shard_fns
                    # tensor = shard_fns[key](tensor)
                    adjusted_key = ('params', 'params') + key
                    if adjusted_key in shard_fns:
                        tensor = shard_fns[adjusted_key](tensor)
                    else:
                        print(f"Warning: No shard function found for key {adjusted_key}")
                flattend_train_state[key] = tensor

        if target is not None:
            flattened_target = flatten_dict(to_state_dict(target), keep_empty_nodes=True)
            for key, value in flattened_target.items():
                if key not in flattend_train_state and value == empty_node:
                    flattend_train_state[key] = value

        train_state = unflatten_dict(flattend_train_state)
        if target is None:
            return train_state

        return from_state_dict(target, train_state)

    @staticmethod
    def _load_flax_checkpoint(path, param_shapes=None, shard_fns=None):
        """Load a standard flax checkpoint that's not saved with the
        msgpack streaming format.

        Args:
            path (str): Path to the checkpoint file.
            param_shapes (Any, optional): Template object to restore the state into.
            shard_fns (dict, optional): Functions to apply to each shard of the loaded state.

        Returns:
            The loaded and potentially reshaped state dictionary.
        """
        with utils.open_file(path) as fin:
            encoded_bytes = fin.read()

        # Deserialize the checkpoint data using msgpack
        state_dict = flax.serialization.msgpack_restore(encoded_bytes)

        # If shard functions are provided, apply them to reshape the loaded state
        if shard_fns is not None:
            shard_fns = to_state_dict(shard_fns)
            state_dict = tree_apply(shard_fns, state_dict)

        if param_shapes is None:
            return state_dict

        # Restore the state into the provided param_shapes structure
        return from_state_dict(param_shapes, state_dict)

    @classmethod
    def load_trainstate_checkpoint(
        cls,
        load_from,
        state_shapes=None,
        shard_fns=None,
    ):
        """
        Load a checkpoint for the training state.

        Args:
            load_from (str): String specifying the checkpoint type and path.
            state_shapes (Any, optional): Template of the expected training state structure.
            shard_fns (dict, optional): Functions to reshape the loaded state.

        Returns:
            tuple: (train_state, restored_params)
        """
        # Extract the parameters param_shapes from the state_shapes if provided
        if state_shapes is not None:
            param_shapes = state_shapes.params["params"]
        else:
            param_shapes = None

        # Extract the parameter shard functions if provided
        if shard_fns is not None:
            params_shard_fns = shard_fns.params["params"]
        else:
            params_shard_fns = None

        # Split the load_from string into type and path
        load_type, load_path = load_from.split("::", 1)

        train_state = None
        restored_params = None

        if load_type == "flax_params":
            # Load parameters in standard flax format (non-streaming)
            restored_params = cls._load_flax_checkpoint(
                path=load_path,
                param_shapes=param_shapes,
                shard_fns=params_shard_fns)
            restored_params = {"params": restored_params}
        elif load_type == "trainstate":
            # Load the entire train state in the streaming format
            train_state = cls.load_checkpoint(
                path=load_path,
                target=state_shapes,
                shard_fns=shard_fns,
            )
        elif load_type == "trainstate_params":
            # Load the params part of the train state in the streaming format
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=param_shapes,
                shard_fns=params_shard_fns,
                remove_dict_prefix=('params', 'params'),
            )
            restored_params = {'params': restored_params}
        elif load_type == "params":
            # Load the params in the streaming format
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=param_shapes,
                shard_fns=params_shard_fns,
            )
            restored_params = {'params': restored_params}
        else:
            raise ValueError(f"Invalid load_from type: {load_type}")

        return train_state, restored_params
