"""
SegResNet implementation - Efficient segmentation with residual blocks.

Based on 3D MRI brain tumor segmentation using autoencoder regularization
Paper: https://arxiv.org/abs/1810.11654
"""

import torch
import torch.nn as nn
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class ResBlock(nn.Module):
    """Residual block with optional normalization."""

    def __init__(self, in_channels, norm_type='group', dimension='3d'):
        super().__init__()

        conv_layer = nn.Conv3d if dimension == '3d' else nn.Conv2d

        self.conv1 = conv_layer(in_channels, in_channels, 3, padding=1)
        self.conv2 = conv_layer(in_channels, in_channels, 3, padding=1)

        if norm_type == 'group':
            num_groups = min(32, in_channels)
            self.norm1 = nn.GroupNorm(num_groups, in_channels)
            self.norm2 = nn.GroupNorm(num_groups, in_channels)
        elif norm_type == 'instance':
            norm_class = nn.InstanceNorm3d if dimension == '3d' else nn.InstanceNorm2d
            self.norm1 = norm_class(in_channels)
            self.norm2 = norm_class(in_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual
        out = self.activation(out)

        return out


class DownBlock(nn.Module):
    """Downsampling block with residual blocks."""

    def __init__(self, in_channels, out_channels, num_res_blocks=1, dimension='3d'):
        super().__init__()

        conv_layer = nn.Conv3d if dimension == '3d' else nn.Conv2d

        self.down = conv_layer(in_channels, out_channels, 3, stride=2, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResBlock(out_channels, dimension=dimension) for _ in range(num_res_blocks)]
        )

    def forward(self, x):
        x = self.down(x)
        x = self.res_blocks(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with residual blocks."""

    def __init__(self, in_channels, out_channels, num_res_blocks=1, dimension='3d'):
        super().__init__()

        conv_transpose = nn.ConvTranspose3d if dimension == '3d' else nn.ConvTranspose2d

        self.up = conv_transpose(in_channels, out_channels, 2, stride=2)

        self.res_blocks = nn.Sequential(
            *[ResBlock(out_channels, dimension=dimension) for _ in range(num_res_blocks)]
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = x + skip  # Addition instead of concatenation

        x = self.res_blocks(x)
        return x


class SegResNet(nn.Module):
    """
    SegResNet - Efficient segmentation network with residual blocks.

    Features:
    - Residual blocks throughout
    - Addition-based skip connections (not concatenation)
    - Group normalization
    - Efficient and lightweight
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        init_filters: int = 8,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        dimension: str = '3d'
    ):
        super().__init__()

        conv_layer = nn.Conv3d if dimension == '3d' else nn.Conv2d

        # Initial convolution
        self.input_conv = conv_layer(in_channels, init_filters, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        channels = init_filters
        for num_blocks in blocks_down:
            self.encoders.append(
                DownBlock(channels, channels * 2, num_blocks, dimension)
            )
            channels *= 2

        # Decoder
        self.decoders = nn.ModuleList()
        for num_blocks in blocks_up:
            self.decoders.append(
                UpBlock(channels, channels // 2, num_blocks, dimension)
            )
            channels //= 2

        # Output
        self.output_conv = conv_layer(channels, out_channels, 1)

    def forward(self, x):
        # Initial convolution
        x = self.input_conv(x)

        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(x)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)

        # Output
        x = self.output_conv(x)

        return x


@NodeRegistry.register('networks', 'SegResNet',
                      description='Efficient segmentation with residual blocks')
class SegResNetNode(PyTorchModuleNode):
    """SegResNet node for efficient medical image segmentation."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)
        self.add_output('model', DataType.MODEL)    # For training

    def build_module(self) -> nn.Module:
        return SegResNet(
            in_channels=self.get_config('in_channels', 1),
            out_channels=self.get_config('out_channels', 2),
            init_filters=self.get_config('init_filters', 8),
            dimension=self.get_config('dimension', '3d')
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
            print(f"Error in SegResNetNode: {e}")
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
            'init_filters': {
                'type': 'text',
                'label': 'Initial Filters',
                'default': '8'
            },
            'dimension': {
                'type': 'choice',
                'label': 'Dimension',
                'choices': ['2d', '3d'],
                'default': '3d'
            }
        }
