import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, input_channels, input_size, neurons):
        super().__init__()
        layers = []
        prev_channels = input_channels
        for n in neurons:
            layers.extend(
                [
                    nn.Conv1d(prev_channels, n, 4, 2, 1, bias=False),
                    nn.BatchNorm1d(n) if prev_channels != input_channels else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            prev_channels = n
        layers.append(nn.Conv1d(prev_channels, 1, input_size // (2 ** len(neurons)), 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz, output_size):
        super(Generator, self).__init__()
        self.output_size = output_size
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, 16, 1, 0, bias=False),
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
            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.main(x)

        x = x[:, :, : self.output_size]
        return x
