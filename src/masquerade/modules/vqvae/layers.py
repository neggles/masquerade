from math import ceil
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


# Conv2D with same padding
class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        return super().forward(x)


# AvgPool2D with same padding
class AvgPool2dSame(nn.AvgPool2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        return super().forward(x)


# Convolutional ResNet block as per MaskGit
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        groups: int = 32,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.act1 = nn.SiLU()
        self.conv1 = Conv2dSame(self.in_channels, self.out_channels, kernel_size=3, bias=False)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=self.out_channels, eps=eps, affine=True)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = Conv2dSame(self.out_channels, self.out_channels, kernel_size=3, bias=False)

        if self.in_channels != self.out_channels:
            self.resample = Conv2dSame(
                self.out_channels, self.out_channels, kernel_size=3 if use_conv_shortcut else 1, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)

        if self.in_channels != self.out_channels:
            residual = self.resample(residual)

        return x + residual


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        use_conv_shortcut: bool = False,
        add_downsample: bool = False,
        conv_downsample: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.layers.append(
                ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    use_conv_shortcut=use_conv_shortcut,
                )
            )

        if add_downsample:
            if conv_downsample:
                self.downsample = Conv2dSame(out_channels, out_channels, kernel_size=4, stride=2)
            else:
                self.downsample = AvgPool2dSame(kernel_size=2, stride=2)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        use_conv_shortcut: bool = False,
        add_upsample: bool = False,
    ):
        super().__init__()
        layers = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            layers.append(
                ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    use_conv_shortcut=use_conv_shortcut,
                )
            )

        self.layers = nn.ModuleList(layers)
        if add_upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                Conv2dSame(out_channels, out_channels, kernel_size=3, stride=1, bias=False),
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
