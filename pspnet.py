from turtle import forward
import torch
import torch.nn as nn
from torch.nn import functional as F

from backbone import model_factory

class PSPModule(nn.Module):
    def __init__(self,features, out_features=1024, sizes=(1, 2, 3, 6)) -> None:
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)
        

    def forward(self, features):
        h, w = features.size(2), features.size(3)
        priors = [F.interpolate(input=stage(features), size=(h, w), mode='bilinear') for stage in self.stages] + [features]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, backbone_type='resnet34', pretrained=True, pool_scale=[1, 2, 3, 6], backbone_out_features=512, n_classes=21) -> None:
        super(PSPNet,self).__init__()
        self.backbone = model_factory(backbone_type, pretrained)
        self.psp = PSPModule(backbone_out_features, 1024, pool_scale)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 512)
        self.up_2 = PSPUpsample(512, 256)
        self.up_3 = PSPUpsample(256, 128)
        self.up_4 = PSPUpsample(128, 64)
        self.up_5 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )
        self.network = nn.Sequential(self.backbone, self.psp, self.drop_1, self.up_1, self.up_2, self.up_3, self.up_4, self.up_5, self.drop_2, self.final)

    def forward(self, x):
        return self.network(x)

