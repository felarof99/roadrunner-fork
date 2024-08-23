import functools
import gc
import re
import warnings
import jax
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from easydel.etils.partition_module import PartitionAxis
from easydel.modules.modeling_utils import FelafaxPretrainedConfig

logger = print


class AutoJAXModelForCausalLM:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: jax.Device = jax.devices("cpu")[0],
        dtype: jax.numpy.dtype = jax.numpy.float32,
        param_dtype: jax.numpy.dtype = jax.numpy.float32,
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
        sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
        sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
        partition_axis: PartitionAxis = PartitionAxis(),
        shard_attention_computation: bool = True,
        input_shape: Tuple[int, int] = (1, 1),
        shard_fns: Optional[Mapping[tuple, Callable] | dict] = None,
        backend: Optional[str] = None,
        config_kwargs: Optional[Mapping[str, Any]] = None,
        auto_shard_params: bool = False,
        partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]] = None,
        quantization_method: Optional[Literal["4bit", "8bit"]] = None,
        bit_targeted_params: Optional[List[str]] = None,
        verbose_params: bool = False,
        safe: bool = True,
        from_torch: bool = True,
        **kwargs,
    ) -> Tuple[FelafaxPretrainedModel, dict]:

        with jax.default_device(device):
            from easydel.modules.modeling_utils import FelafaxPretrainedModel

            return FelafaxPretrainedModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                input_shape=input_shape,
                dtype=dtype,
                precision=precision,
                param_dtype=param_dtype,
                partition_axis=partition_axis,
                auto_shard_params=auto_shard_params,
                shard_fns=shard_fns,
                sharding_axis_dims=sharding_axis_dims,
                sharding_axis_names=sharding_axis_names,
                config_kwargs=config_kwargs,
                partition_rules=partition_rules,
                safe=safe,
            )


class AutoJAXModelConfig:    

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
        **kwargs,
    ) -> FelafaxPretrainedConfig:
        """
        Load a pretrained model configuration from the Hugging Face model hub.

        Args:
            pretrained_model_name_or_path (str): Name or path of the pretrained model.
            sharding_axis_dims (Sequence[int]): Dimensions for each sharding axis.
            sharding_axis_names (Sequence[str]): Names of sharding axes (e.g., "dp", "fsdp", "tp", "sp").
            partition_axis (PartitionAxis): Module for array partitioning in EasyDeL.
            shard_attention_computation (bool): Whether to use shard_map for attention computation.
            backend (Optional[str]): Computation backend for the model.
            from_torch (bool): Whether to load config from PyTorch models.
            **kwargs: Additional arguments for model and config initialization.

        Returns:
            A Model Config
        """
        from transformers import AutoConfig

        cls_main = AutoConfig if from_torch else FelafaxPretrainedConfig
        config = cls_main.from_pretrained(pretrained_model_name_or_path)
        model_type: str = config.model_type

        cfg, module, trf = get_modules_by_type(model_type)
        cfg = cfg.from_pretrained(pretrained_model_name_or_path)
        if hasattr(cfg, "add_jax_args"):
            cfg.add_jax_args()
        cfg.add_basic_configurations(
            axis_dims=sharding_axis_dims,
            axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            shard_attention_computation=shard_attention_computation,
        )

        return cfg