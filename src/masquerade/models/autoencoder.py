import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from omegaconf import ListConfig
from packaging import version
from safetensors.torch import load_file as load_safetensors
from torch import nn
from vector_quantize_pytorch import VectorQuantize

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
        input_key: str = "image",
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
        encoder: ConvEncoder,
        decoder: ConvDecoder,
        quantizer: Union[VectorQuantizer, VectorQuantize],
        loss: nn.Module,
        opt_ae: OptimizerCallable,
        opt_disc: OptimizerCallable,
        lr_g_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.quantize = quantizer
        self.decoder = decoder
        self.loss = loss
        self.opt_ae = opt_ae
        self.opt_disc = opt_disc
        self.lr_g_factor = lr_g_factor

    def get_input(self, batch: Dict) -> torch.Tensor:
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.get_trainable_parameters())
            + list(self.loss.get_trainable_autoencoder_parameters())
        )
        return params

    def get_discriminator_params(self) -> list:
        params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        z = self.encoder(x)
        z, reg_log = self.quantize(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: Any) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z)
        return z, dec, reg_log

    def training_step(self, batch, batch_idx: int, optimizer_idx: int) -> Any:
        x = self.get_input(batch)
        z, xrec, regularization_log = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(
                regularization_log,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(
                regularization_log,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx: int) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch, batch_idx: int, postfix="") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        aeloss, log_dict_ae = self.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        discloss, log_dict_disc = self.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )
        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        log_dict_ae.update(log_dict_disc)
        self.log_dict(log_dict_ae)
        return log_dict_ae

    def configure_optimizers(self) -> Any:
        ae_params = self.get_autoencoder_params()
        opt_ae = self.opt_ae(
            ae_params,
            lr=(self.lr_g_factor or 1.0) * self.learning_rate,
        )
        disc_params = self.get_discriminator_params()
        opt_disc = self.opt_disc(
            disc_params,
            lr=self.learning_rate,
        )

        return [opt_ae, opt_disc]

    @torch.no_grad()
    def log_images(self, batch: Dict, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch)
        _, xrec, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        with self.ema_scope():
            _, xrec_ema, _ = self(x)
            log["reconstructions_ema"] = xrec_ema
        return log
