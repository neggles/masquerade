from enum import Enum
from pathlib import Path
from typing import Annotated, List, NoReturn, Optional

import typer
from lightning.pytorch.cli import LightningCLI
from rich.traceback import install as traceback_install

from masquerade import __version__, console
from masquerade.dataset import HFDatasetModule
from masquerade.train_vae import train_app as train_vae_app

_ = traceback_install()

app: typer.Typer = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(
    train_vae_app,
    name="train-vae",
)


def version_callback(value: bool) -> NoReturn:
    if value is True:
        console.print(f"accelmon v{__version__}")
        raise typer.Exit()


@app.command(no_args_is_help=True)
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", "-v", callback=version_callback, is_eager=True, is_flag=True, help="Show version"
        ),
    ] = None,
    args: Annotated[
        List[str],
        typer.Argument(help="Arguments to pass to the LightningCLI"),
    ] = ...,
) -> NoReturn:
    """
    Main entrypoint for training.
    """
    cli = LightningCLI(
        datamodule_class=HFDatasetModule,
        subclass_mode_data=True,
        subclass_mode_model=True,
        args=args,
    )
