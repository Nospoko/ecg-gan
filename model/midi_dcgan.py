import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # This line calculates the padding to ensure the size remains consistent
            nn.ConstantPad1d((0, 1), 0),
            nn.Conv1d(out_channels, out_channels, 4, 1, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module("conv_shortcut", nn.Conv1d(in_channels, out_channels, 4, 2, 1, bias=False))
            self.shortcut.add_module("bn_shortcut", nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return nn.LeakyReLU(0.2, inplace=True)(self.residual(x) + self.shortcut(x))


class Discriminator(nn.Module):
    def __init__(self, input_channels, input_size, neurons):
        super().__init__()

        layers = []
        prev_channels = input_channels
        for n in neurons:
            layers.append(ResidualBlock(prev_channels, n))
            prev_channels = n

        layers.append(nn.Conv1d(prev_channels, 1, input_size // (2 ** len(neurons)), 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, noise_size, output_size):
        super(Generator, self).__init__()
        self.output_size = output_size
        self.main = nn.Sequential(
            nn.ConvTranspose1d(noise_size, 512, 3, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 4, 4, 2, 1, bias=False),
            # Ensure channels in positive range
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)

        # Ensure 2nd channel has integers in 0-127 range
        x[:, 2, :] = torch.round(x[:, 2, :] * 127)

        # Ensure 3rd channel has integers in 0-127 range
        x[:, 3, :] = torch.round(x[:, 3, :] * 127)

        x = x[:, :, : self.output_size]
        return x
