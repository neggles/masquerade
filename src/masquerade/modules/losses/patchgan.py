from torch import Tensor, nn

from masquerade.modules.utils import ActNorm


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    else:
        raise NotImplementedError(f"Unknown module type {classname}")


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_actnorm: bool = False,
    ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if use_actnorm:
            norm_layer = ActNorm
            use_bias = True
        else:
            norm_layer = nn.BatchNorm2d
            use_bias = False

        k_size = 4
        pad_size = 1
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(input_nc, ndf, kernel_size=k_size, stride=2, padding=pad_size),
                nn.LeakyReLU(0.2, True),
            ]
        )

        layer_mult = 1
        prev_layer_mult = 1
        for n in range(n_layers):  # gradually increase the number of filters
            layer_num = n + 1  # range() starts at 0, but we want to start at 1

            prev_layer_mult = layer_mult
            layer_mult = min(2**layer_num, 8)

            in_channels = ndf * prev_layer_mult
            out_channels = ndf * layer_mult

            self.layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=k_size,
                        stride=2 if layer_num < n_layers else 1,  # last layer has stride 1
                        padding=pad_size,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * layer_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            )

        # output 1 channel prediction map
        self.layers.append(nn.Conv2d(ndf * layer_mult, 1, kernel_size=k_size, stride=1, padding=pad_size))

    def initialize_weights(self):
        return self.apply(weights_init)

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward."""
        for layer in self.layers:
            x = layer(x)
        return x
