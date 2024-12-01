
    

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator
from dataclasses import dataclass
from typing import Optional, Tuple


def get_mesh(num_tpus: int, mesh_shape: Optional[Tuple[int, int, int]] = None):
    if mesh_shape is None:
        if num_tpus == 1:
            mesh_shape = (1, 1, 1)
        elif num_tpus == 2:
            mesh_shape = (1, 2, 1)
        elif num_tpus == 4:
            mesh_shape = (1, 2, 2)
        else:
            raise ValueError(f"Invalid number of TPUs: {num_tpus}")

    print(f"Creating TPU device mesh with shape {mesh_shape}...")
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(device_mesh, axis_names=("batch", "fsdp", "mp"))
    return mesh

@dataclass
class TrainerConfig:
    """Configuration for the Llama trainer"""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    param_dtype: str = "float32"
    compute_dtype: str = "float32"

    # Training configuration
    num_epochs: int = 1
    num_steps: Optional[int] = None
    num_tpus: int = 2
    mesh_shape: Optional[Tuple[int, int, int]] = None

    learning_rate: float = 1e-3

    # lora configuration
    lora_rank: int = 4  # Rank for lora matrices
    use_lora: bool = False  # Enable or disable lora training

    # Environment configuration
    base_dir: str = "/mnt/persistent-disk"
    hf_token: Optional[str] = None

    # Logging configuration
    log_interval: int = 10
    eval_interval: int = 10
    eval_steps: int = 10

    # Restore checkpoint
    restore_checkpoint: bool = False

    use_optimized_decoder: bool = True




# def main():
#     t = torch.randn(2, 2, device=xm.xla_device())
#     print(t.device)
#     print(t)
    
    

if __name__ == "__main__":
    main()
    