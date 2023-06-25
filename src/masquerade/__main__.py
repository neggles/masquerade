from enum import Enum
from pathlib import Path
from typing import Annotated, List, NoReturn, Optional

import typer
from lightning.pytorch.cli import LightningCLI

import masquerade.dataset  # noqa: F401
import masquerade.models  # noqa: F401
from masquerade import __version__, console


cli = LightningCLI(
    subclass_mode_data=True,
    subclass_mode_model=True,
    parser_kwargs={"prog": "masquerade"},
)
