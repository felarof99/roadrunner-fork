import os
import asyncio
import aiofiles
from typing import Any, Dict, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_state_dict, to_state_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, PartitionSpec as PS
from ml_collections import ConfigDict
from orbax import checkpoint

from . import jax_utils, utils

def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'fp16': jnp.float16,
        'fp32': jnp.float32,
        'fp64': jnp.float64,
    }[dtype]

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
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):
        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype', None) in float_dtypes:
                return tensor.astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return tensor.astype(dtype_spec.dtype)
            return tensor
        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        out_sharding = partition_spec if isinstance(partition_spec, jax.sharding.NamedSharding) else jax.sharding.NamedSharding(jax_utils.MESH, partition_spec)
        jax_shard_function = jax.jit(make_to_dtype_fn(dtype_spec), out_shardings=out_sharding)
        return lambda tensor: jax_shard_function(tensor).block_until_ready()

    def make_gather_fn(partition_spec, dtype_spec=None):
        in_sharding = partition_spec if isinstance(partition_spec, jax.sharding.NamedSharding) else jax.sharding.NamedSharding(jax_utils.MESH, partition_spec)
        jax_gather_fn = jax.jit(make_to_dtype_fn(dtype_spec), in_shardings=in_sharding)
        return lambda tensor: jax.device_get(jax_gather_fn(tensor))

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs, dtype_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs, dtype_specs)
    
    return shard_fns, gather_fns

class OrbaxCheckpointer:
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.float_dtype = 'bf16'
        config.save_optimizer_state = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config: ConfigDict, checkpoint_dir: str, enable: bool = True):
        self.config = self.get_default_config(config)
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable
        self.checkpointer = checkpoint.PyTreeCheckpointer()
        self.options = checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True)
        self.manager = checkpoint.CheckpointManager(
            self.checkpoint_dir,
            self.checkpointer,
            self.options
        )

    async def _async_save_checkpoint(self, train_state: Any, step: int) -> None:
        save_args = checkpoint.SaveArgs(aggregate_filename='checkpoint')
        await self.manager.async_save(step, train_state, save_kwargs={'save_args': save_args})

    def save_checkpoint(self, train_state: Any, step: int) -> None:
        if self.enable:
            asyncio.run(self._async_save_checkpoint(train_state, step))

    async def _async_restore_checkpoint(self, target: Optional[Any] = None) -> Any:
        return await self.manager.async_restore(self.manager.latest_step(), items=target)

    def restore_checkpoint(self, target: Optional[Any] = None) -> Any:
        return asyncio.run(self._async_restore_checkpoint(target))

    @staticmethod
    async def async_copy_directory(src: str, dst: str) -> None:
        async def copy_file(src: str, dst: str) -> None:
            async with aiofiles.open(src, 'rb') as fsrc:
                async with aiofiles.open(dst, 'wb') as fdst:
                    await fdst.write(await fsrc.read())

        async def copy_item(src: str, dst: str) -> None:
            if os.path.isfile(src):
                await copy_file(src, dst)
            elif os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                tasks = [copy_item(os.path.join(src, item), os.path.join(dst, item)) for item in os.listdir(src) if item != "tmp"]
                await asyncio.gather(*tasks)

        await copy_item(src, dst)

    @classmethod
    def copy_directory(cls, src: str, dst: str) -> None:
        asyncio.run(cls.async_copy_directory(src, dst))

    @classmethod
    def load_checkpoint(cls, 
                        load_dir: str, 
                        target: Optional[Any] = None, 
                        step: Optional[int] = None, 
                        shard_fns: Optional[Dict] = None) -> Any:
        checkpointer = checkpoint.PyTreeCheckpointer()
        if step is None:
            ckpt_paths = [f for f in os.listdir(load_dir) if f.startswith('checkpoint_') and f.endswith('.ckpt')]
            if not ckpt_paths:
                raise ValueError(f"No checkpoints found in {load_dir}")
            latest_ckpt = max(ckpt_paths, key=lambda x: int(x.split('_')[1].split('.')[0]))
            step = int(latest_ckpt.split('_')[1].split('.')[0])
        
        ckpt = checkpointer.restore(os.path.join(load_dir, f'checkpoint_{step}'))
        
        if shard_fns is not None:
            ckpt = jax.tree_util.tree_map(lambda fn, x: fn(x), shard_fns, ckpt)
        
        if target is not None:
            return from_state_dict(target, ckpt)
        return ckpt

    @classmethod
    def load_trainstate_checkpoint(cls,
                                   load_from: str,
                                   state_shapes: Optional[Any] = None,
                                   shard_fns: Optional[Dict] = None) -> tuple:
        load_type, load_path = load_from.split('::', 1)

        if load_type == 'orbax':
            train_state = cls.load_checkpoint(load_path, target=state_shapes, shard_fns=shard_fns)
            return train_state, None
        elif load_type == 'flax_params':
            params = cls.load_checkpoint(load_path, target=state_shapes.params['params'] if state_shapes else None, shard_fns=shard_fns.params['params'] if shard_fns else None)
            return None, {'params': params}
        else:
            raise ValueError(f'Invalid load_from type: {load_type}')