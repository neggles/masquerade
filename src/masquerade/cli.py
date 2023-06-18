from pathlib import Path
from typing import Annotated, List, NoReturn, Optional, Union

import typer

from masquerade import __version__, console

cli: typer.Typer = typer.Typer(rich_markup_mode="rich")


def version_callback(value: bool) -> NoReturn:
    if value is True:
        console.print(f"accelmon v{__version__}")
        raise typer.Exit()


def int_list_callback(value: str) -> list[int]:
    return [int(node) for node in value.split(",")]


@cli.command()
def main(
    verbose: bool = typer.Option(
        False, "-v", "--verbose", is_flag=True, help="Enable verbose console output."
    ),
    option: Path = typer.Option(
        "./option.ini", "-o", "--option", exists=True, dir_okay=False, help="Path to a file"
    ),
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            is_flag=True,
            help="Show version",
        ),
    ] = None,
    argument: str = typer.Argument(..., help="An argument"),
) -> NoReturn:
    """
    Main entrypoint for your application.
    """
    raise typer.Exit()
