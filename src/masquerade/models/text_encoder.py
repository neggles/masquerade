from os import PathLike
from typing import Union

import torch
from torch.amp import autocast
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.byt5 import ByT5Tokenizer
from transformers.models.t5 import T5EncoderModel, T5Tokenizer
from transformers.tokenization_utils import BatchEncoding

from masquerade.modules.encoders import AbstractEmbModel


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self,
        model_name_or_path: Union[str, PathLike] = "",
        max_length: int = 256,
        model_kwargs: dict = {},
        freeze: bool = True,
        apply_mask: bool = True,
    ):
        super().__init__()
        self.model: T5EncoderModel = T5EncoderModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, **model_kwargs)

        self.device = self.model.device
        self.dtype = self.model.dtype
        self.max_length = max_length
        self.apply_mask = apply_mask

        if freeze:
            self.freeze()

    def forward(self, text: Union[str, list[str]]) -> torch.Tensor:
        batch_encoding: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = batch_encoding["input_ids"].to(self.device)

        with autocast("cuda", enabled=False):
            output: BaseModelOutputWithPastAndCrossAttentions = self.model(input_ids=input_ids)
        z = output.last_hidden_state
        return z

    def encode(self, text: Union[str, list[str]]) -> torch.Tensor:
        """Encode text into a latent representation."""
        return self(text)


class FrozenByT5Embedder(AbstractEmbModel):
    """
    Uses the ByT5 transformer encoder for text. Is character-aware.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, PathLike] = "",
        max_length: int = 256,
        model_kwargs: dict = {},
        freeze: bool = True,
    ):
        super().__init__()
        self.model: T5EncoderModel = T5EncoderModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.tokenizer: ByT5Tokenizer = ByT5Tokenizer.from_pretrained(model_name_or_path, **model_kwargs)

        self.device = self.model.device
        self.dtype = self.model.dtype
        self.max_length = max_length

        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text: Union[str, list[str]]) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with autocast("cuda", enabled=False):
            output: BaseModelOutputWithPastAndCrossAttentions = self.model(tokens)
        z = output.last_hidden_state
        return z

    def encode(self, text: Union[str, list[str]]) -> torch.Tensor:
        """Encode text into a latent representation."""
        return self(text)
