import torch.nn as nn
import torch.nn.functional as F

from modules.conv1d_builder import Conv1DBuilder
from modules.residual_stack import ResidualStack


class ConvolutionalEncoder(nn.Module):
    def __init__(
        self,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        use_kaiming_normal,
        input_features_type,
        features_filters,
        device,
        verbose=False,
    ):
        super(ConvolutionalEncoder, self).__init__()

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """
        self._batch_norm1 = nn.BatchNorm1d(num_hiddens)
        self._batch_norm2 = nn.BatchNorm1d(num_hiddens)
        self._batch_norm3 = nn.BatchNorm1d(num_hiddens)
        self._batch_norm4 = nn.BatchNorm1d(num_hiddens)
        self._batch_norm5 = nn.BatchNorm1d(num_hiddens)

        self._conv_1 = Conv1DBuilder.build(
            in_channels=features_filters,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1,
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1,
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=2,
        )

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1,
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1,
        )

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._input_features_type = input_features_type
        self._features_filters = features_filters
        self._device = device
        self._verbose = verbose

    def forward(self, inputs):
        if self._verbose:
            print("inputs size: {}".format(inputs.size()))

        x_conv_1 = F.relu(self._batch_norm1(self._conv_1(inputs)))
        if self._verbose:
            print("x_conv_1 output size: {}".format(x_conv_1.size()))

        x_conv_2 = F.relu(self._batch_norm2(self._conv_2(x_conv_1))) + x_conv_1
        if self._verbose:
            print("_conv_2 output size: {}".format(x_conv_2.size()))

        x_conv_3 = F.relu(self._batch_norm3(self._conv_3(x_conv_2)))
        if self._verbose:
            print("_conv_3 output size: {}".format(x_conv_3.size()))

        x_conv_4 = F.relu(self._batch_norm4(self._conv_4(x_conv_3))) + x_conv_3
        if self._verbose:
            print("_conv_4 output size: {}".format(x_conv_4.size()))

        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        if self._verbose:
            print("x_conv_5 output size: {}".format(x_conv_5.size()))

        x = self._residual_stack(x_conv_5) + x_conv_5
        if self._verbose:
            print("_residual_stack output size: {}".format(x.size()))

        return x
