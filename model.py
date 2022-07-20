import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, no_in_channels =3 , no_out_channels=2):
        super().__init__()
        self.encoder = ResEncoder(in_channels=no_in_channels, out_channels=64)
        self.res1 = ResBlock(in_channels=1, out_channels=2, stride= 2)
        self.res2 = ResBlock(in_channels=1, out_channels=2, stride=2)
        self.res3 = ResBlock(in_channels=1, out_channels=2, stride=2)
        self.res4 = ResBlock(in_channels=1, out_channels=2, stride=2)
        self.res5 = ResBlock(in_channels=1, out_channels=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, no_out_channels)
        pass

    def forward(self, input_tensor):
        encoded_result = self.encoder(input_tensor)
        res1_result = self.res1(encoded_result)
        res2_result = self.res2(res1_result)
        res3_result = self.res3(res2_result)
        res4_result = self.res4(res3_result)
        res5_result = self.res5(res4_result)

        avg = self.avgpool(res5_result)
        flattened = self.flatten(avg)
        fully_connected = self.fc(flattened)
        return fully_connected


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.n_n = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.one_cross_one_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, input_tensor):
        res_out = self.res(input_tensor)
        input_tensor = self.one_by_one(input_tensor)
        input_tensor = self.batchnorm(input_tensor)
        output_tensor = res_out + input_tensor
        return output_tensor


class ResEncoder(nn.Module):
    # Takes the image tensor first and encode it
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, input_tensor):
        return self.encoder(input_tensor)
