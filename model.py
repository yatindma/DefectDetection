import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.module):
    def __init__(self, no_in_channels =3 , no_output_classes=2):
        super().__init()
        self.en
        pass


class ResBlock(nn.module):
    def __init__(self, in_channels, out_channels, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        super().__init__()
        self.n_n = nn.sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input_tensor):
        res_out = self.res(input_tensor)
        input_tensor = self.one_by_one(input_tensor)
        input_tensor = self.batchnorm(input_tensor)
        output_tensor = res_out + input_tensor
        return output_tensor
