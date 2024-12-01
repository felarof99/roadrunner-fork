import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)
from peft import LoraConfig, TaskType, get_peft_model
from .dataset import create_med_qa_loaders

from src.felafax.trainer_engine.data.data import (
    DatasetConfig,
)


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


# Select a supported model from above list to use!
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HUGGINGFACE_TOKEN = input(
    "Please provide your HUGGINGFACE_TOKEN: "
)  # YOUR_HF_TOKEN


def apply_lora(*, model, lora_rank=None, lora_alpha=None, lora_dropout=None):
    """Applies LoRA configuration to the model."""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8 if not lora_rank else lora_rank,
        lora_alpha=32 if not lora_alpha else lora_alpha,
        lora_dropout=0.1 if not lora_dropout else lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def init_model(*, model_name, hugging_face_token):
    """Downloads and initializes the model."""
    config = AutoConfig.from_pretrained(model_name, token=hugging_face_token)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hugging_face_token
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hugging_face_token, low_cpu_mem_usage=True
    )

    # model = apply_lora(
    #     model=model,
    #     lora_rank=TRAINER_CONFIG["lora_rank"],
    #     lora_alpha=TRAINER_CONFIG["lora_alpha"],
    #     lora_dropout=TRAINER_CONFIG["lora_dropout"],
    # )

    return model, tokenizer


def main():
    model, tokenizer = init_model(
        model_name=MODEL_NAME, hugging_face_token=HUGGINGFACE_TOKEN
    )
    
    # Create dataset configuration for MedQA
    medqa_config = DatasetConfig(
        # Data loading parameters
        data_source="ngram/medchat-qa",
        max_examples=None,
        # Batching parameters
        batch_size=8,
        max_seq_length=4096,
        num_workers=8,
        ignore_index=-100,
        mask_prompt=False,
        pad_id=0,
    )
    train_dataloader, val_dataloader = create_med_qa_loaders(
        config=medqa_config, tokenizer=tokenizer
    )
    # Print first batch from train_dataloader
    for batch in train_dataloader:
        print("Sample batch from train_dataloader:")
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
            print(f"Content: {value}")
        break  # Only print first batch

    t = torch.randn(2, 2, device=xm.xla_device())
    print(t.device)
    print(t)


if __name__ == "__main__":
    main()
