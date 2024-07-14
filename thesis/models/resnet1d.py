import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import Flatten
from models.basic_conv1d import create_head1d

###############################################################################################
# Standard resnet


def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False,
    )


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=[3, 3], downsample=None):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size // 2 + 1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Sequential):
    """1d adaptation of the torchvision resnet"""

    def __init__(
        self,
        block,
        layers,
        kernel_size=3,
        num_classes=2,
        input_channels=3,
        inplanes=64,
        fix_feature_dim=True,
        kernel_size_stem=None,
        stride_stem=2,
        pooling_stem=True,
        stride=2,
        lin_ftrs_head=None,
        ps_head=0.5,
        bn_final_head=False,
        bn_head=True,
        act_head="relu",
        concat_pooling=True,
    ):
        self.inplanes = inplanes

        layers_tmp = []

        if kernel_size_stem is None:
            kernel_size_stem = (
                kernel_size[0] if isinstance(kernel_size, list) else kernel_size
            )
        # stem
        layers_tmp.append(
            nn.Conv1d(
                input_channels,
                inplanes,
                kernel_size=kernel_size_stem,
                stride=stride_stem,
                padding=(kernel_size_stem - 1) // 2,
                bias=False,
            )
        )
        layers_tmp.append(nn.BatchNorm1d(inplanes))
        layers_tmp.append(nn.ReLU(inplace=True))
        if pooling_stem is True:
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        # backbone
        for i, l in enumerate(layers):
            if i == 0:
                layers_tmp.append(
                    self._make_layer(
                        block, inplanes, layers[0], kernel_size=kernel_size
                    )
                )
            else:
                layers_tmp.append(
                    self._make_layer(
                        block,
                        inplanes if fix_feature_dim else (2**i) * inplanes,
                        layers[i],
                        stride=stride,
                        kernel_size=kernel_size,
                    )
                )

        # head
        # layers_tmp.append(nn.AdaptiveAvgPool1d(1))
        # layers_tmp.append(Flatten())
        # layers_tmp.append(nn.Linear((inplanes if fix_feature_dim else (2**len(layers)*inplanes)) * block.expansion, num_classes))

        head = create_head1d(
            (inplanes if fix_feature_dim else (2 ** len(layers) * inplanes))
            * block.expansion,
            nc=num_classes,
            lin_ftrs=lin_ftrs_head,
            ps=ps_head,
            bn_final=bn_final_head,
            bn=bn_head,
            act=act_head,
            concat_pooling=concat_pooling,
        )
        layers_tmp.append(head)

        super().__init__(*layers_tmp)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel_size, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_layer_groups(self):
        return (self[6], self[-1])

    def get_output_layer(self):
        return self[-1][-1]

    def set_output_layer(self, x):
        self[-1][-1] = x

# original used kernel_size_stem = 8
def resnet1d_wang(**kwargs):

    if not ("kernel_size" in kwargs.keys()):
        kwargs["kernel_size"] = [5, 3]
    if not ("kernel_size_stem" in kwargs.keys()):
        kwargs["kernel_size_stem"] = 7
    if not ("stride_stem" in kwargs.keys()):
        kwargs["stride_stem"] = 1
    if not ("pooling_stem" in kwargs.keys()):
        kwargs["pooling_stem"] = False
    if not ("inplanes" in kwargs.keys()):
        kwargs["inplanes"] = 128

    return ResNet1d(BasicBlock1d, [1, 1, 1], **kwargs)


def resnet1d(**kwargs):
    """Constructs a custom ResNet model."""
    return ResNet1d(BasicBlock1d, **kwargs)

