import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 150 x 150
            nn.Conv2d(
                channels_img, features_d * 8, kernel_size=4, stride=2, padding=1
            ),  # img: 150x150 -> 75x75
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d * 8, features_d * 4, 3, 2, 1),  # img: 75x75 -> 38x38
            self._block(features_d * 4, features_d * 2, 4, 2, 1),  # img: 38x38 -> 19x19
            self._block(features_d * 2, features_d, 3, 2, 1),  # img: 19x19 -> 10x10
            self._block(features_d, features_d, 4, 2, 1),  # img: 10x10 -> 5x5
            # After all _block img output is 1x1 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d, 1, kernel_size=4, stride=2, padding=0),  # img: 5x5 -> 1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 5, 1, 0),  # img: 1x1 -> 5x5
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 5x5 -> 10x10
            self._block(features_g * 8, features_g * 4, 3, 2, 1),  # img: 10x10 -> 19x19
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 19x19 -> 38x38
            self._block(features_g * 2, features_g, 3, 2, 1),  # img: 38x38 -> 75x75
            nn.ConvTranspose2d(
                features_g, 1, kernel_size=4, stride=2, padding=1
            ),  # img: 75x75 -> 150x150
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)