from pathlib import Path
from typing import Annotated, List, NoReturn, Optional

import torch
import typer
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from lightning_utilities.core.imports import module_available
from torch.optim import Adam

import masquerade.dataset
import masquerade.models
from masquerade import __version__, console
from masquerade.dataset import HFDatasetModule
from masquerade.models import BaseAutoencoder, VQAutoEncoder

train_app: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
)


class VAETrainerCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.CosineAnnealingLR)


@train_app.command(add_help_option=False)
def main(
    args: Annotated[
        Optional[ArgsType],
        typer.Argument(help="Arguments to pass to the trainer."),
    ] = None,
) -> NoReturn:
    """
    Main entrypoint for training VQGAN VAE.
    """
    if Path.cwd().joinpath("configs/lightning/defaults.yaml").exists():
        default_config_files: List[str] = ["configs/lightning/defaults.yaml"]
    else:
        default_config_files: List[str] = []

    cli = VAETrainerCLI(
        datamodule_class=HFDatasetModule,
        model_class=BaseAutoencoder,
        subclass_mode_data=True,
        subclass_mode_model=True,
        auto_configure_optimizers=False,
        args=args,
        parser_kwargs=dict(
            default_config_files=default_config_files,
        ),
    )


if __name__ == "__main__":
    main()
