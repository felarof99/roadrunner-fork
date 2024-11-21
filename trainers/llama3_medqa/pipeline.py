import jax
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer
from src.felafax.trainer_engine.trainer import Trainer, TrainerConfig
from src.felafax.trainer_engine.setup import setup_environment
from src.felafax.trainer_engine.checkpoint import (
    Checkpointer,
    CheckpointerConfig,
)
from src.felafax.trainer_engine.data.base import (
    DatasetConfig,
    load_data,
    create_dataloader,
    SFTDataset,
)
from src.felafax.trainer_engine import utils
from .dataset import create_med_qa_loaders, DatasetConfig


load_dotenv()
TEST_MODE = False
HF_TOKEN = os.getenv("HF_TOKEN") or input(
    "Please enter your HuggingFace token: "
)
BASE_DIR = os.getenv("BASE_DIR") or input(
    "Please enter the base directory for the training run: "
)
# Tip: To avoid entering these values manually, create a .env file in the `llama3_medqa` folder with:
#   HF_TOKEN=your_huggingface_token
#   BASE_DIR=path_to_base_directory

########################################################
# Configure the dataset pipeline
########################################################
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN
)

# Create dataset configuration for MedQA
medqa_config = DatasetConfig(
    # Data loading parameters
    data_source="ngram/medchat-qa",
    max_examples=None,
    # Batching parameters
    batch_size=8,
    max_seq_length=4096,  # 8192 fails, 4096 works
    num_workers=8,
    ignore_index=-100,
    mask_prompt=False,
    pad_id=0,
)

# Create dataloaders for SFT
train_dataloader, val_dataloader = create_med_qa_loaders(
    config=medqa_config, tokenizer=tokenizer
)

########################################################
# Configure the trainer pipeline
########################################################
trainer_config = TrainerConfig(
    # Model configuration
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    param_dtype="bfloat16",
    output_dtype="bfloat16",
    # Training configuration
    num_epochs=1,
    num_steps=20,  # set to None to run through the entire dataset
    num_tpus=jax.device_count(),
    mesh_shape=(1, 1, 4),  # (batch, fsdp, mp)
    # lora configuration
    lora_rank=16,
    use_lora=False,
    learning_rate=1e-5,
    # Environment configuration
    base_dir=BASE_DIR,
    hf_token=HF_TOKEN,
    # Logging configuration
    log_interval=5,
    eval_interval=100,
    eval_steps=10,
    # Restore checkpoint
    restore_checkpoint=False,
)

# Set up the training environment using trainer_config
setup_environment(trainer_config)

# Configure the checkpointer
checkpointer_config = CheckpointerConfig(
    checkpoint_dir=f"{trainer_config.base_dir}/checkpoints/",
    max_to_keep=2,
    save_interval_steps=50,
)
checkpointer = Checkpointer(config=checkpointer_config)

# Put everything together and initialize the trainer
trainer = Trainer(
    trainer_config=trainer_config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    checkpointer=checkpointer,
)

# Run training
trainer.train()

export_dir = f"{trainer_config.base_dir}/hf_export/"

# # Export the model in HF format
# trainer.export(export_dir=export_dir)

# # Upload exported model to HF
# utils.upload_dir_to_hf(
#     dir_path=export_dir, 
#     repo_name="felarof01/test-llama3.1-8b-medqa-finetuned-2048-no-mask",
#     token=HF_TOKEN,
# )
