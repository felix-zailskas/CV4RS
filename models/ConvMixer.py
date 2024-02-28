import timm
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    """
    Implementation of the ConvMixer architecture as found in https://doi.org/10.48550/arXiv.2201.09792
    """

    def __init__(
        self, dim, depth, channels, kernel_size=9, patch_size=7, n_classes=1000
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim, dim, kernel_size, groups=dim, padding="same"
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
                for i in range(depth)
            ],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.output = nn.Linear(dim, n_classes)

    def forward(self, x, features_only=False):
        y = self.features(x)
        if not features_only:
            y = self.output(y)
        return y


def create_convmixer(
    channels: int,
    num_classes: int,
    pretrained: bool = False,
    version: str = "convmixer_768_32",
) -> ConvMixer:
    """
    Initializes and returns an instance of the ConvMixer architecture.

    Args:
        channels (int): Amount of channels in the input images.
        num_classes (int): Amount of classes to predict from.
        pretrained (bool, optional): Use model pretrained on ImageNet1K. Defaults to False.
        version (str, optional): Chose model type as available on timm. Defaults to "convmixer_768_32".

    Returns:
        ConvMixer: The ConvMixer as specified.
    """
    convmixer = timm.create_model(
        version, pretrained=pretrained, in_chans=channels, num_classes=num_classes
    )
    return convmixer
