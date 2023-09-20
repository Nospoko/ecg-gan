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
    def __init__(self, nz, output_channels, output_size, neurons):
        super(Generator, self).__init__()

        self.output_channels = output_channels
        self.output_size = output_size

        layers = []

        # Using a specific kernel_size for the first ConvTranspose layer
        first_kernel_size = output_size // 16

        layers.extend(
            [
                nn.ConvTranspose1d(nz, neurons[-1], first_kernel_size, 1, 0, bias=False),
                nn.BatchNorm1d(neurons[-1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        prev_channels = neurons[-1]

        for n in reversed(neurons[:-1]):
            layers.extend(
                [nn.ConvTranspose1d(prev_channels, n, 4, 2, 1, bias=False), nn.BatchNorm1d(n), nn.LeakyReLU(0.2, inplace=True)]
            )
            prev_channels = n

        layers.extend([nn.ConvTranspose1d(prev_channels, output_channels, 4, 2, 1, bias=False), nn.Tanh()])

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        # return only output_size samples
        x = x[:, :, : self.output_size]
        return x
