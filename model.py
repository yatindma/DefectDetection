# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ResNet(nn.Module):
#     def __init__(self, no_in_channels =3 , no_out_channels=2):
#         super().__init__()
#         self.encoder = ResEncoder(in_channels=no_in_channels, out_channels=64)
#         self.res1 = ResBlock(in_channels=64, out_channels=64, stride=1)
#         self.res2 = ResBlock(in_channels=64, out_channels=128, stride=2)
#         self.res3 = ResBlock(in_channels=128, out_channels=256, stride=2)
#         self.res4 = ResBlock(in_channels=256, out_channels=512, stride=2)
#         self.decoder = ResnetDecoder(in_ch=512, out_ch=no_out_channels)
#         pass
#
#     def forward(self, input_tensor):
#         print("input result ", input_tensor.shape)
#         encoded_result = self.encoder(input_tensor)
#         print("encoded result ", encoded_result.shape)
#         res1_result = self.res1(encoded_result)
#         print("RES1  ", res1_result.shape)
#         res2_result = self.res2(res1_result)
#         res3_result = self.res3(res2_result)
#         res4_result = self.res4(res3_result)
#         output_tensor = self.decoder(res4_result)
#         print("XxxxxXXXXxxxxXXXXXxxxxxXXXXX")
#         return output_tensor
#
#
# class ResnetDecoder(nn.Module):
#     """
#     This class represents the tail of ResNet. It performs a global pooling and maps the output to the
#     correct class by using a fully connected layer.
#     """
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d((1, 1))
#         self.fully = nn.Linear(in_ch, out_ch)
#
#     def forward(self, input_tensor):
#         averaged = self.avg(input_tensor)
#         flattened = torch.flatten(averaged, 1)
#         output_tensor = self.fully(flattened)
#         return output_tensor
#
#
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super().__init__()
#         self.n_n = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#         self.one_cross_one_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#         self.batch_norm = nn.BatchNorm2d(out_channels)
#
#     def forward(self, input_tensor):
#         res_out = self.n_n(input_tensor)
#         print("RES RESULT", res_out.shape)
#         one_x_one_input_tensor = self.one_cross_one_conv(input_tensor)
#         batched_norm_input_tensor = self.batch_norm(one_x_one_input_tensor)
#         output_tensor = res_out + batched_norm_input_tensor
#         return output_tensor
#
#
# class ResEncoder(nn.Module):
#     # Takes the image tensor first and encode it
#     def __init__(self, in_channels=3, out_channels=64):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
#
#     def forward(self, input_tensor):
#         return self.encoder(input_tensor)




import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, n_in_channels=3, n_out_classes=2):
        super().__init__()
        self.encoder = ResNetGate(in_channels=n_in_channels, out_channels=64)
        self.resblock1 = ResBlock(in_ch=64, out_ch=64, stride=1)
        self.resblock2 = ResBlock(in_ch=64, out_ch=128, stride=2)
        self.resblock3 = ResBlock(in_ch=128, out_ch=256, stride=2)
        self.resblock4 = ResBlock(in_ch=256, out_ch=512, stride=2)
        self.decoder = ResnetDecoder(in_ch=512, out_ch=n_out_classes)

    def forward(self, input_tensor):
        first_output = self.encoder(input_tensor)
        res_out1 = self.resblock1(first_output)
        res_out2 = self.resblock2(res_out1)
        res_out3 = self.resblock3(res_out2)
        res_out4 = self.resblock4(res_out3)
        output_tensor = self.decoder(res_out4)
        return output_tensor


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())
        self.one_by_one = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_ch)

    def forward(self, input_tensor):
        res_out = self.res(input_tensor)
        input_tensor = self.one_by_one(input_tensor)
        input_tensor = self.batchnorm(input_tensor)
        output_tensor = res_out + input_tensor
        return output_tensor


class ResNetGate(nn.Module):
    """ResNet encoder before the resblocks."""
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, input_tensor):
        output_tensor = self.gate(input_tensor)
        return output_tensor


#Out Block
class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fully = nn.Linear(in_ch, out_ch)

    def forward(self, input_tensor):
        averaged = self.avg(input_tensor)
        flattened = torch.flatten(averaged, 1)
        output_tensor = self.fully(flattened)
        return output_tensor