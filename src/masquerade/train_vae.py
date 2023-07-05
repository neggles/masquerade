import sys
from typing import Annotated, List, NoReturn, Optional

import typer
from lightning.pytorch.cli import LightningCLI
from rich.traceback import install as traceback_install

import masquerade.dataset
import masquerade.models
from masquerade import __version__, console
from masquerade.dataset import HFDatasetModule
from masquerade.models.autoencoder import BaseAutoencoder, VQAutoEncoder

_ = traceback_install()

train_app: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
)


cli = LightningCLI(
    datamodule_class=HFDatasetModule,
    model_class=BaseAutoencoder,
    subclass_mode_data=True,
    subclass_mode_model=True,
)


@train_app.command(add_help_option=False)
def main(
    args: Annotated[
        Optional[list[str]],
        typer.Argument(help="Arguments to pass to the LightningCLI"),
    ] = None,
) -> NoReturn:
    """
    Main entrypoint for training VQGAN VAE.
    """
    sys.argv = sys.argv[:2] + args

    cli()


if __name__ == "__main__":
    cli()
