import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from packaging import version
from safetensors.torch import load_file as load_safetensors
from torch import nn

from masquerade.modules.ema import LitEma
from masquerade.modules.vqvae import ConvDecoder, ConvEncoder, VectorQuantizer


class BaseAutoencoder(L.LightningModule):
    """
    Base class for autoencoders.
    """

    def __init__(
        self,
        ema_decay: Optional[float] = None,
        monitor: Optional[str] = None,
        input_key: str = "jpg",
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple, list, ListConfig] = (),
    ):
        super().__init__()
        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    def init_from_ckpt(self, path: str, ignore_keys: Union[Tuple, list, ListConfig] = tuple()) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError(f"get_input() not implemented in class {self.__class__.__name__}")

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(f"encode() not implemented in {self.__class__}")

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(f"decode() not implemented in {self.__class__}")

    def configure_optimizers(self) -> Any:
        raise NotImplementedError(f"configure_optimizers() not implemented in {self.__class__}")


class VQAutoEncoder(BaseAutoencoder):
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        latent_channels: int = 3,
        num_vq_embeddings: int = 256,
        vq_embed_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels

        self.encoder = ConvEncoder(**encoder_config)

        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)

        self.decoder = ConvDecoder(**decoder_config)

    def get_input(self, batch: Dict) -> torch.Tensor:
        return batch[self.input_key]

    def encode(self, x: torch.FloatTensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        h = self.quantize(h)

        return (h,)

    def decode(self, h: torch.FloatTensor):
        # also go through quantization layer
        quant, emb_loss, info = self.quantize(h)
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2)

        return (dec,)

    def forward(self, sample: torch.FloatTensor):
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        h = self.encode(x)
        dec = self.decode(h)

        return (dec,)
