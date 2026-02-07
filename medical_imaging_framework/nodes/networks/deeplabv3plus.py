"""
DeepLabV3+ implementation with Atrous Spatial Pyramid Pooling (ASPP).

Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
Paper: https://arxiv.org/abs/1802.02611
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class ASPPConv(nn.Module):
    """Atrous convolution with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPooling(nn.Module):
    """Global average pooling with 1x1 convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()

        modules = []

        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Project
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res = torch.cat(res, dim=1)
        return self.project(res)


class Decoder(nn.Module):
    """DeepLabV3+ decoder."""

    def __init__(self, low_level_channels, num_classes):
        super().__init__()

        # Low-level feature processing
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Decoder convolutions
        self.conv2 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 256 from ASPP + 48 from low-level
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        # Upsample and concatenate
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.conv2(x)

        return x


class SimpleBackbone(nn.Module):
    """Simple ResNet-like backbone."""

    def __init__(self, in_channels=1):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer2 = self._make_layer(64, 64, 2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.layer5 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        low_level_feat = self.layer2(x)
        x = self.layer3(low_level_feat)
        x = self.layer4(x)
        x = self.layer5(x)

        return x, low_level_feat


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for semantic segmentation.

    Features:
    - ASPP (Atrous Spatial Pyramid Pooling)
    - Encoder-decoder architecture
    - Low-level feature fusion
    - Multi-scale processing
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        atrous_rates: tuple = (6, 12, 18)
    ):
        super().__init__()

        self.backbone = SimpleBackbone(in_channels)
        self.aspp = ASPP(512, 256, atrous_rates)
        self.decoder = Decoder(64, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Backbone
        features, low_level_feat = self.backbone(x)

        # ASPP
        x = self.aspp(features)

        # Decoder
        x = self.decoder(x, low_level_feat)

        # Upsample to input size
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        # Classifier
        x = self.decoder.classifier(x)

        return x


@NodeRegistry.register('networks', 'DeepLabV3Plus',
                      description='DeepLabV3+ with ASPP for segmentation')
class DeepLabV3PlusNode(PyTorchModuleNode):
    """DeepLabV3+ node for semantic segmentation."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)
        self.add_output('model', DataType.MODEL)    # For training

    def build_module(self) -> nn.Module:
        return DeepLabV3Plus(
            in_channels=self.get_config('in_channels', 1),
            num_classes=self.get_config('num_classes', 2)
        )

    def execute(self) -> bool:
        try:
            if self.module is None:
                self.initialize_module()

            # Always output the model itself (for trainer/optimizer)
            self.set_output_value('model', self.module)

            # If there's input data, do forward pass
            x = self.get_input_value('input')
            if x is not None:
                with torch.set_grad_enabled(self.training_mode):
                    output = self.module(x)
                self.set_output_value('output', output)
            return True

        except Exception as e:
            print(f"Error in DeepLabV3PlusNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'in_channels': {
                'type': 'text',
                'label': 'Input Channels',
                'default': '1'
            },
            'num_classes': {
                'type': 'text',
                'label': 'Number of Classes',
                'default': '2'
            }
        }
