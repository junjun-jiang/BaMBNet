import torch.nn as nn
from models.rcab import RCAB, ResidualGroup, default_conv
import math
import torch
import torch.nn.functional as F


def kernel_conv(kernel_size, input_dim, reduction, max_pool=False, upsample=False):
    res_conv = []

    if kernel_size <= 1:
        res_conv = [nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(True)]
        return nn.Sequential(*res_conv)
    else:
        res_conv.append(ResidualGroup(default_conv, input_dim, 3, reduction, n_resblocks=math.floor(kernel_size/3)))

    if max_pool:
        res_conv.append(nn.MaxPool2d(kernel_size=2, stride=2))

    if upsample:
        res_conv.append(nn.Upsample(scale_factor=2))

    return nn.Sequential(*res_conv)


def connect_conv(input_dim, output_dim, kernel_size, stride, padding, bias=True, dilation=1):
    conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, dilation=dilation)
    relu = nn.ReLU(True)

    return nn.Sequential(*[conv, relu])


class KernelEDNet(nn.Module):
    def __init__(self):
        super(KernelEDNet, self).__init__()
        kernel_size = [1, 4, 7, 10]
        self.kernel_size = kernel_size

        self.head = connect_conv(6, 64, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        convk_tail = nn.Conv2d(64 * 1, 3, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        reluk_tail = nn.Sigmoid()
        self.tail_hard = nn.Sequential(*[convk_tail, reluk_tail])

        self.layer1 = nn.ModuleList()
        for k in kernel_size:
            self.layer1.append(kernel_conv(k, 64, 16, max_pool=True))

        self.layer2 = nn.ModuleList()
        for k in kernel_size:
            self.layer2.append(kernel_conv(k, 64, 16, max_pool=True))

        self.layer3 = nn.ModuleList()
        for k in kernel_size:
            self.layer3.append(kernel_conv(k, 64, 16, max_pool=True))

        self.layer4 = nn.ModuleList()
        for k in kernel_size:
            self.layer4.append(kernel_conv(k, 64, 16, max_pool=True))

        self.layer5 = nn.ModuleList()
        for k in kernel_size:
            self.layer5.append(kernel_conv(k, 64, 16, upsample=True))

        self.layer6 = nn.ModuleList()
        for k in kernel_size:
            self.layer6.append(kernel_conv(k, 64, 16, upsample=True))

        self.layer7 = nn.ModuleList()
        for k in kernel_size:
            self.layer7.append(kernel_conv(k, 64, 16, upsample=True))

        self.layer8 = nn.ModuleList()
        for k in kernel_size:
            self.layer8.append(kernel_conv(k, 64, 16, upsample=True))

        self.MAX_TRAINNUM = 2e4
        self.iter_num = 0

    def forward(self, x, gt=None):
        blur, _ = x[:, 6:, :, :].abs().max(dim=1, keepdim=True)
        x = x[:, :6, :, :]

        x = self.head(x)
        blur_mask = []

        static_kernel_size = [0.0, 1.9, 4.2, 6.2]
        for kernel_bound, kernel_up in zip(static_kernel_size, static_kernel_size[1:]):
            mask = ((blur >= kernel_bound) & (blur < kernel_up)).float()
            blur_mask.append(mask)

        mask = (blur >= static_kernel_size[-1]).float()
        blur_mask.append(mask)

        layer_output1 = []
        for i in range(len(self.kernel_size)):
            layer_output1.append(self.layer1[i](x))

        layer_output2 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output1[i].size()[2:])
            layer_output2.append(self.layer2[i](res_x+layer_output1[i]))

        layer_output3 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output2[i].size()[2:])
            layer_output3.append(self.layer3[i](res_x+layer_output2[i]))

        layer_output4 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output3[i].size()[2:])
            layer_output4.append(self.layer4[i](res_x+layer_output3[i]))

        layer_output5 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output4[i].size()[2:])
            layer_output5.append(self.layer5[i](res_x + layer_output4[i]))

        layer_output6 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output5[i].size()[2:])
            layer_output6.append(self.layer6[i](res_x+layer_output5[i] + layer_output3[i]))

        layer_output7 = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output6[i].size()[2:])
            layer_output7.append(self.layer7[i](res_x+layer_output6[i] + layer_output2[i]))

        layer_output = []
        for i in range(len(self.kernel_size)):
            res_x = F.adaptive_avg_pool2d(x, layer_output7[i].size()[2:])
            layer_output.append(self.layer8[i](res_x+layer_output7[i] + layer_output1[i]))

        feature_layer = []

        if gt is not None:
            self.iter_num += 1

        if self.iter_num < self.MAX_TRAINNUM:
            iter_weight = torch.exp(torch.tensor(- (self.iter_num * 2 / self.MAX_TRAINNUM) ** 2))
            for layer_i, blur_i in zip(layer_output, blur_mask):
                feature_layer.append((layer_i * blur_i * iter_weight + (1-iter_weight) * layer_i).unsqueeze(0))
        else:
            feature_layer = [layer_i.unsqueeze(0) for layer_i in layer_output]
        layer_res = torch.cat(feature_layer, dim=0).sum(dim=0)
        x = x + layer_res
        out = self.tail_hard(x)
        return [out]

