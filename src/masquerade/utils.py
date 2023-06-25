import functools
import importlib
from contextlib import contextmanager
from functools import partial
from importlib import resources
from pathlib import Path
from typing import TypeVar

import fsspec
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file as load_safetensors
from torch import Tensor
from torch.nn import Module

from masquerade.constants import PACKAGE_ROOT

# used for overriding nn.Module methods while retaining type information
T = TypeVar("T", bound=Module)


def disabled_train(self: T, mode: bool = True) -> T:
    """No-op method to disable training mode"""
    return self


def get_data_path(file: str = ..., module: str = f"{__name__.split('.')[0]}.data") -> Path:
    """Get the path to a module data file"""
    module_data = resources.files(module)
    file_path = module_data.joinpath(file)

    if not module_data.joinpath(file).exists():
        raise FileNotFoundError(f"File {file} not found in {module_data}")

    return file_path


def get_string_from_tuple(s: str):
    try:
        # Check if the string starts and ends with parentheses
        if s[0] == "(" and s[-1] == ")":
            # Convert the string to a tuple
            t = eval(s)
            # Check if the type of t is tuple
            if type(t) == tuple:
                return t[0]
            else:
                pass
    except Exception:
        pass
    return s


def is_power_of_two(n) -> bool:
    if n <= 0:
        return False
    return bool((n & (n - 1)) == 0)


def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


def log_txt_as_img(wh, xc, size=10) -> Tensor:
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        if isinstance(xc[bi], list):
            text_seq = xc[bi][0]
        else:
            text_seq = xc[bi]
        lines = "\n".join(text_seq[start : start + nc] for start in range(0, len(text_seq), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return str(Path(p).absolute())
    return path


def ismap(x) -> bool:
    if not isinstance(x, Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x) -> bool:
    if not isinstance(x, Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def isheatmap(x) -> bool:
    if not isinstance(x, Tensor):
        return False

    return x.ndim == 2


def isneighbors(x) -> bool:
    if not isinstance(x, Tensor):
        return False
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


def expand_dims_like(x: Tensor, y: Tensor) -> Tensor:
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def mean_flat(tensor: Tensor) -> Tensor:
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x: Tensor) -> Tensor:
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x: Tensor, target_dims) -> Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def load_model_from_config(config, ckpt, verbose=True, freeze=True) -> Module:
    print(f"Loading model from {ckpt}")
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError

    model: Module = instantiate_from_config(config.model)
    sd = pl_sd["state_dict"]

    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.eval()
    return model


MODEL_EXTNS = [".pt", ".ckpt", ".pth", ".safetensors"]


@contextmanager
def package_data_file(name: str, package: str = __package__, mode: str = "rb") -> str:
    package_files = resources.files(package)
    target_file = package_files.joinpath(name)
    if not target_file.exists():
        for ext in MODEL_EXTNS:
            target_file = package_files.joinpath(name + ext)
            if target_file.exists():
                break

    if not target_file.exists():
        raise FileNotFoundError(f"Could not find file {name} in package {package}")
    try:
        yield target_file.open(mode)
    finally:
        pass
