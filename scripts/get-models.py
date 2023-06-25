#!/usr/bin/env python3
from dataclasses import dataclass
from os import environ
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

HF_HOME = Path(environ.get("HF_HOME", "~/.cache/huggingface"))
HF_HUB_CACHE = Path(environ.get("HUGGINGFACE_HUB_CACHE", HF_HOME.joinpath("hub")))

MODELS_DIR = Path(__file__).parent.parent.joinpath("models")


@dataclass
class ModelMeta:
    repo_id: str
    type_dir: str = "models"
    get: bool = True

    @property
    def name(self):
        return self.repo_id.split("/")[-1]


GET_MODELS = [
    ModelMeta("google/t5-v1_1-large", "tenc", True),
    ModelMeta("google/t5-v1_1-xl", "tenc", True),
]

ALLOW_PATTERNS = ["*.json", "*.md", "*.txt", "pytorch_*.bin", "pytorch_*.safetensors", "spiece.model"]


class DownloadTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "nrows": 20,
                "ncols": 100,
                "dynamic_ncols": False,
                "disable": None,
            }
        )
        super().__init__(*args, **kwargs)


def download_model(model: ModelMeta, models_dir: Path):
    if model.get is False:
        print(f"Skipping {model.repo_id} (get={model.get})")
        return

    target_dir = models_dir.joinpath(model.type_dir, model.name)
    if target_dir.joinpath("config.json").exists() or target_dir.joinpath("model_index.json").exists():
        print(f"Skipping {model.repo_id} (delete {target_dir.relative_to(Path.cwd())} to re-download)")
        return

    target_dir.mkdir(exist_ok=True, parents=True)
    print(f"Downloading {model.repo_id} to {target_dir}...")
    snapshot_download(
        repo_id=model.repo_id,
        revision="main",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERNS,
        cache_dir=HF_HUB_CACHE,
        tqdm_class=DownloadTqdm,
        max_workers=4,
        token=True,
        resume_download=True,
    )
    print(f"Downloaded {model.repo_id} to {target_dir}.")


if __name__ == "__main__":
    for model in GET_MODELS:
        download_model(model, MODELS_DIR)
