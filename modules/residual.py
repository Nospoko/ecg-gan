import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal):
        super(Residual, self).__init__()

        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(
            in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False
        )

        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)

        if use_kaiming_normal:
            conv_1 = nn.utils.weight_norm(conv_1)
            nn.init.kaiming_normal_(conv_1.weight, nonlinearity="relu")
            conv_2 = nn.utils.weight_norm(conv_2)
            nn.init.kaiming_normal_(conv_2.weight, nonlinearity="relu")

        # All parameters same as specified in the paper
        self._block = nn.Sequential(relu_1, conv_1, relu_2, conv_2)

    def forward(self, x):
        return x + self._block(x)
