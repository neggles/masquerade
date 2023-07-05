from os import PathLike
from pathlib import Path
from time import sleep
from typing import Any, Optional, Union

import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
import torchvision
import wandb
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image

from masquerade.trainer.settings import WandbConfig
from masquerade.utils import isheatmap

MULTINODE_HACKS = True


class SetupCallback(Callback):
    def __init__(
        self,
        resume: bool = False,
        now: str = ...,
        logdir: PathLike = ...,
        ckptdir: PathLike = ...,
        cfgdir: PathLike = ...,
        config: OmegaConf = ...,
        lightning_config: OmegaConf = ...,
        debug: bool = False,
        ckpt_name: Optional[str] = None,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir: Path = Path(logdir)
        self.ckptdir: Path = Path(ckptdir)
        self.cfgdir: Path = Path(cfgdir)
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: pl.Trainer, pl_module: L.LightningModule, exception) -> None:
        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            if self.ckpt_name is None:
                ckpt_path = self.ckptdir.joinpath("last.ckpt")
            else:
                ckpt_path = self.ckptdir.joinpath(self.ckpt_name)
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            self.logdir.mkdir(exist_ok=True, parents=True)
            self.ckptdir.mkdir(exist_ok=True, parents=True)
            self.cfgdir.mkdir(exist_ok=True, parents=True)

            if "callbacks" in self.lightning_config:
                if "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                    self.ckptdir.joinpath("trainstep_checkpoints").mkdir(exist_ok=True, parents=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                sleep(5)
            OmegaConf.save(
                self.config,
                self.cfgdir.joinpath(f"{self.now}-project.yaml"),
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                self.cfgdir.joinpath(f"{self.now}-lightning.yaml"),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and self.logdir.exists():
                dst = self.logdir.parent.joinpath("child_runs", self.logdir.name)
                dst.parent.mkdir(exist_ok=True, parents=True)
                try:
                    self.logdir.rename(dst.absolute())
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency=1000,
        max_images=4,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_before_first_step=False,
        enable_autocast=True,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = Path(save_dir) / "images" / split
        root.mkdir(exist_ok=True, parents=True)

        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(images[k].cpu().numpy(), cmap="hot", interpolation="lanczos")
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
                path = root / filename

                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
                path = root / filename

                img = Image.fromarray(grid)
                img.save(path)
                if pl_module is not None:
                    if not isinstance(pl_module.logger, WandbLogger):
                        raise ValueError("logger_log_image only supports WandbLogger currently")
                    pl_module.logger.log_image(
                        key=f"{split}/{k}",
                        images=[img],
                        step=pl_module.global_step,
                    )

    @rank_zero_only
    def log_img(
        self,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
        split: str = "train",
    ):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()
                    if self.clamp and not isheatmap(images[k]):
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module if isinstance(pl_module.logger, WandbLogger) else None,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx: int):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx,
    ):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
        *args,
        **kwargs,
    ):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


def get_wandb_logger(
    config: Optional[WandbConfig] = None,
    run_name: Optional[str] = None,
    run_version: Optional[str] = None,
) -> WandbLogger:
    if config is None:
        config = WandbConfig()

    if config.save_dir is not None:
        save_dir = Path(config.save_dir)
    else:
        save_dir = Path.cwd().joinpath("data", "wandb")
    # make sure save_dir exists, don't need to but it's polite
    save_dir.mkdir(exist_ok=True, parents=True)
    # make logger
    return WandbLogger(
        project=config.project,
        save_dir=config.save_dir,
        group_name=config.group_name,
        name=run_name or config.run_name,
        version=run_version or config.run_version,
        prefix=config.prefix,
        log_model=config.log_model,
        checkpoint_name=config.checkpoint_name,
        settings=wandb.Settings(
            code_dir=Path(__file__).parent.parent,
        ),
    )


def get_checkpoint_logger(ckpt_dir: PathLike, monitor: Optional[str] = None) -> ModelCheckpoint:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    save_top_k = 3 if monitor is not None else 1

    return ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:06d}-{step:06d}",
        verbose=True,
        save_last=True,
        save_on_train_epoch_end=True,
        monitor=monitor,
        save_top_k=save_top_k,
    )


def get_image_logger(**kwargs) -> ImageLogger:
    return ImageLogger(**kwargs)


def get_lr_monitor(**kwargs) -> LearningRateMonitor:
    kwargs.setdefault("logging_interval", "step")
    return LearningRateMonitor(**kwargs)
