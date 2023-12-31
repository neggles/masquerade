from contextlib import contextmanager
from importlib import resources

import torch

DATA_PACKAGE = __package__

LPIPS_PACKAGE = DATA_PACKAGE + ".lpips"


@contextmanager
def package_file(dir: str, name: str):
    file = resources.files(f"{__package__}.{dir}").joinpath(name)
    if not file.exists():
        raise FileNotFoundError(f"File {file} not found in {resources.files(__package__)}")
    try:
        yield file
    finally:
        pass


@contextmanager
def lpips_checkpoint(name: str = "vgg_lpips"):
    lpips_file = resources.files(LPIPS_PACKAGE).joinpath(f"{name}.pth")
    if not lpips_file.exists():
        raise FileNotFoundError(f"File {lpips_file} not found in {LPIPS_PACKAGE}")
    try:
        yield lpips_file.open("rb")
    finally:
        pass
