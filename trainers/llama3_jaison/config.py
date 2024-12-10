from dataclasses import dataclass, field
from pathlib import Path
import jax

from src.felafax.trainer_engine.trainer import TrainerConfig
from src.felafax.trainer_engine.data.data import DatasetConfig
from src.felafax.trainer_engine.checkpoint import CheckpointerConfig


@dataclass
class PipelineConfig:
    """Main configuration for the training pipeline"""

    trainer_dir: str = ""
    export_dir: str = ""
    hf_token: str = ""
    hf_model_download_token: str = ""
    hf_repo: str = ""
    test_mode: bool = False

    data_config: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            data_source="ngram/medchat-qa",
            batch_size=8,
            max_seq_length=4096,
            num_workers=8,
            mask_prompt=False,
            dataset_input_field="instruction",
            dataset_output_field="output",
            split="train",
        )
    )

    trainer_config: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            param_dtype="bfloat16",
            compute_dtype="bfloat16",
            num_epochs=1,
            num_steps=100,
            num_tpus=jax.device_count(),
            learning_rate=1e-3,
            lora_rank=16,
            use_lora=True,
            log_interval=5,
            eval_interval=5,
            eval_steps=10,
            restore_checkpoint=False,
            base_dir="/mnt/persistent-disk",
        )
    )

    checkpointer_config: CheckpointerConfig = field(
        default_factory=lambda: CheckpointerConfig(
            checkpoint_dir="",
            max_to_keep=2,
            save_interval_steps=50,
            enable_async_checkpointing=True,
            erase_existing_checkpoints=True,
        )
    )

    def __post_init__(self):
        trainer_dir = Path(self.trainer_dir)
        self.trainer_config.base_dir = str(trainer_dir)
        self.trainer_config.hf_token = self.hf_model_download_token
        self.checkpointer_config.checkpoint_dir = str(
            trainer_dir / "checkpoints"
        )

        if isinstance(self.trainer_config.mesh_shape, list):
            self.trainer_config.mesh_shape = tuple(
                self.trainer_config.mesh_shape
            )
