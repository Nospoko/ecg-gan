import torch.nn as nn
import torch.nn.functional as F

from modules.conv1d_builder import Conv1DBuilder
from modules.residual_stack import ResidualStack
from modules.conv_transpose1d_builder import ConvTranspose1DBuilder


class DeconvolutionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        use_kaiming_normal,
        device,
        verbose=False,
    ):
        super(DeconvolutionalDecoder, self).__init__()

        self._device = device
        self._verbose = verbose

        # self.batch_norm_trans_1 = nn.BatchNorm1d(num_hiddens)
        # self.batch_norm_trans_2 = nn.BatchNorm1d(num_hiddens)

        self._conv_1 = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._upsample = nn.Upsample(scale_factor=2)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._conv_trans_1 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._conv_trans_2 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=0,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._conv_trans_3 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=out_channels,
            kernel_size=2,
            padding=0,
            use_kaiming_normal=use_kaiming_normal,
        )

    def forward(self, inputs):
        x = inputs
        if self._verbose:
            print("[FEATURES_DEC] input size: {}".format(x.size()))

        x = self._conv_1(x)
        if self._verbose:
            print("[FEATURES_DEC] _conv_1 output size: {}".format(x.size()))

        x = self._upsample(x)
        if self._verbose:
            print("[FEATURES_DEC] _upsample output size: {}".format(x.size()))

        x = self._residual_stack(x)
        if self._verbose:
            print("[FEATURES_DEC] _residual_stack output size: {}".format(x.size()))

        # x = self.batch_norm_trans_1(x)
        x = F.relu(self._conv_trans_1(x))
        if self._verbose:
            print("[FEATURES_DEC] _conv_trans_1 output size: {}".format(x.size()))

        # x = self.batch_norm_trans_2(x)
        x = F.relu(self._conv_trans_2(x))
        if self._verbose:
            print("[FEATURES_DEC] _conv_trans_2 output size: {}".format(x.size()))

        x = self._conv_trans_3(x)
        if self._verbose:
            print("[FEATURES_DEC] _conv_trans_3 output size: {}".format(x.size()))

        return x
