from inspect import Parameter, signature
from typing import Any

from lightning.pytorch import Trainer


def default_trainer_args() -> dict[str, Any]:
    argspec = dict(signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {param: argspec[param].default for param in argspec if argspec[param] != Parameter.empty}
    return default_args
