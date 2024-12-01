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
    num_steps: Optional[int] = 10
    num_tpus: int = 2
    mesh_shape: Optional[Tuple[int, int, int]] = None

    learning_rate: float = 1e-4

    # lora configuration
    lora_rank: int = 4  # Rank for lora matrices
    use_lora: bool = False  # Enable or disable lora training

    # Environment configuration
    base_dir: str = "/home/ubuntu/trainer_data/"
    hf_token: Optional[str] = "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"

    # Logging configuration
    log_interval: int = 1
    eval_interval: int = 10
    eval_steps: int = 10

    # Restore checkpoint
    restore_checkpoint: bool = False
    use_optimized_decoder: bool = True


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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hugging_face_token
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

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
    trainer_config = TrainerConfig()

    model, tokenizer = init_model(
        model_name=trainer_config.model_name,
        hugging_face_token=trainer_config.hf_token,
    )

    # Create dataset configuration for MedQA
    medqa_config = DatasetConfig(
        # Data loading parameters
        data_source="ngram/medchat-qa",
        max_examples=None,
        # Batching parameters
        batch_size=8,
        max_seq_length=32,
        num_workers=8,
        ignore_index=-100,
        mask_prompt=False,
        pad_id=0,
    )
    train_dataloader, val_dataloader = create_med_qa_loaders(
        config=medqa_config, tokenizer=tokenizer
    )

    torch.manual_seed(99)
    device = xm.xla_device()
    model = model.to(device)

    # # Create a mesh for the model partitioning.
    # num_devices = xr.global_runtime_device_count()
    # mesh_shape = (1, num_devices, 1)
    # device_ids = np.array(range(num_devices))
    # mesh = Mesh(device_ids, mesh_shape, ("dp", "fsdp", "mp"))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=trainer_config.learning_rate
    )

    max_steps = trainer_config.num_steps or float("inf")
    step = 0
    prev_step = -1
    prev_loss = 0.0

    for epoch in range(trainer_config.num_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step > max_steps:
                break
            if (prev_step + 1) % trainer_config.log_interval == 0:
                xm.master_print(
                    f"Step {prev_step} loss: {prev_loss}"
                )

            optimizer.zero_grad()
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
                if "attention_mask" in batch
                else None,
                batch["labels"].to(device),
            )

            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            xm.mark_step()

            prev_step = step
            prev_loss = loss.detach().cpu().item()
            step = step + 1
            
    print(f"Training complete! Final loss: {loss.detach().to('cpu')}")


if __name__ == "__main__":
    main()
