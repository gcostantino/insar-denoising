from collections import OrderedDict

import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)),
            ('relu1', nn.ReLU(inplace=False)),
            ('dropout', nn.Dropout2d(dropout_rate)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)),
            ('relu2', nn.ReLU(inplace=False)),
        ]))
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.residual_conv = None

    def forward(self, x):
        # return self.conv(x)
        residual = x
        out = self.conv(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out = out + residual  # out += residual  # needs to be out-of-place
        return out
