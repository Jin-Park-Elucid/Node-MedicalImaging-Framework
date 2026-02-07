"""
V-Net implementation for 3D volumetric medical image segmentation.

V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
Paper: https://arxiv.org/abs/1606.04797
"""

import torch
import torch.nn as nn
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class ConvBlock(nn.Module):
    """Convolutional block with residual connection."""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.convs = nn.ModuleList()

        for i in range(num_convs):
            if i == 0:
                self.convs.append(nn.Conv3d(in_channels, out_channels, 5, padding=2))
            else:
                self.convs.append(nn.Conv3d(out_channels, out_channels, 5, padding=2))

        self.activation = nn.PReLU(out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = x

        for conv in self.convs:
            out = conv(out)
            out = self.activation(out)

        return out + residual


class DownBlock(nn.Module):
    """Downsampling block."""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.down = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        self.conv_block = ConvBlock(out_channels, out_channels, num_convs)
        self.activation = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.activation(x)
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels, num_convs)
        self.activation = nn.PReLU(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.activation(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class VNet(nn.Module):
    """
    V-Net for 3D volumetric segmentation.

    Features:
    - Residual connections within blocks
    - Skip connections between encoder and decoder
    - PReLU activation
    - Designed for 3D medical images
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 16
    ):
        super().__init__()

        # Initial convolution
        self.input_conv = ConvBlock(in_channels, base_channels, num_convs=1)

        # Encoder
        self.down1 = DownBlock(base_channels, base_channels * 2, num_convs=2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, num_convs=3)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, num_convs=3)

        # Bottleneck
        self.bottom = DownBlock(base_channels * 8, base_channels * 16, num_convs=3)

        # Decoder
        self.up1 = UpBlock(base_channels * 16, base_channels * 8, num_convs=3)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, num_convs=3)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, num_convs=2)
        self.up4 = UpBlock(base_channels * 2, base_channels, num_convs=1)

        # Output
        self.output_conv = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.bottom(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        x = self.output_conv(x)

        return x


@NodeRegistry.register('networks', 'VNet',
                      description='V-Net for 3D volumetric segmentation')
class VNetNode(PyTorchModuleNode):
    """V-Net node for 3D medical image segmentation."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)
        self.add_output('model', DataType.MODEL)    # For training

    def build_module(self) -> nn.Module:
        return VNet(
            in_channels=self.get_config('in_channels', 1),
            out_channels=self.get_config('out_channels', 2),
            base_channels=self.get_config('base_channels', 16)
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
            print(f"Error in VNetNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'in_channels': {
                'type': 'text',
                'label': 'Input Channels',
                'default': '1'
            },
            'out_channels': {
                'type': 'text',
                'label': 'Output Channels',
                'default': '2'
            },
            'base_channels': {
                'type': 'text',
                'label': 'Base Channels',
                'default': '16'
            }
        }
