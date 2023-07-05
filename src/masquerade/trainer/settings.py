from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional, Union

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import SaveConfigCallback
from pydantic import BaseModel, Field

from masquerade.trainer.callbacks import (
    ImageLogger,
    SetupCallback,
    get_checkpoint_logger,
    get_image_logger,
    get_lr_monitor,
    get_wandb_logger,
)

NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


class SetupCallbackConfig(BaseModel):
    resume: bool = Field(False)
    now: str = Field(NOW, description="Datetime string for logging (default: now)")
    logdir: Path = Field(..., description="Path to save logs")
    ckptdir: Path = Field(..., description="Path to save checkpoints")
    cfgdir: Path = Field(..., description="Path to save config files")


class WandbConfig(BaseModel):
    project: str = Field("masquerade", description="WandB project name")
    save_dir: Optional[PathLike] = Field(
        None, description="Path to save WandB logs (default: project_dir/wandb)"
    )
    group_name: Optional[str] = Field(None, description="WandB group name (optional)")
    run_name: Optional[str] = Field(None, description="WandB run name (default: autogenerate)")
    run_version: Optional[str] = Field(None, description="WandB run version/ID for resume (default: none)")
    prefix: str = Field("", description="WandB metric key prefix (default: empty string)")
    log_model: Union[str, bool] = Field(
        False, description="Log checkpoint files to WandB (bandwidth intensive)"
    )
    checkpoint_name: Optional[str] = Field(
        None, description="Name of checkpoint file (if log_model is not False)"
    )

    def __post_init_post_parse__(self) -> None:
        if self.save_dir is not None:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)


class CallbackConfig(BaseModel):
    wandb: WandbConfig = Field(
        default_factory=get_wandb_logger,
        description="WandB configuration (default: project=masquerade, log_model=False)",
    )
    image_logger: ImageLogger = Field(
        default_factory=get_image_logger,
        description="Image logger configuration (default: batch_frequency=1000, max_images=4, clamp=True)",
    )
    learning_rate_logger: LearningRateMonitor = Field(
        default_factory=get_lr_monitor,
        description="Learning rate logger configuration (default: logging_interval=step)",
    )
    checkpoint: ModelCheckpoint = Field(
        default_factory=get_checkpoint_logger,
        description="Checkpoint configuration",
    )
    setup: SetupCallback = Field(
        ...,
        description="Setup callback configuration (default: log_env=True, log_gpu=True)",
    )
