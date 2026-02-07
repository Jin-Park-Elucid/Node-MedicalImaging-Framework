"""
U-Net architecture for medical image segmentation.

Implements 2D and 3D U-Net with various configurations.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: Literal['2d', '3d'] = '2d'
    ):
        super().__init__()

        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if dimension == '2d' else nn.BatchNorm3d

        self.double_conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: Literal['2d', '3d'] = '2d'
    ):
        super().__init__()

        MaxPool = nn.MaxPool2d if dimension == '2d' else nn.MaxPool3d

        self.maxpool_conv = nn.Sequential(
            MaxPool(2),
            DoubleConv(in_channels, out_channels, dimension)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: Literal['2d', '3d'] = '2d',
        bilinear: bool = False
    ):
        super().__init__()

        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d

        if bilinear:
            mode = 'bilinear' if dimension == '2d' else 'trilinear'
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dimension)
        else:
            ConvTranspose = nn.ConvTranspose2d if dimension == '2d' else nn.ConvTranspose3d
            self.up = ConvTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dimension)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch
        if x1.shape[2:] != x2.shape[2:]:
            # Calculate padding
            diff_dims = [x2.size(i) - x1.size(i) for i in range(2, len(x1.shape))]
            padding = []
            for diff in reversed(diff_dims):
                padding.extend([diff // 2, diff - diff // 2])
            x1 = nn.functional.pad(x1, padding)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        base_channels: Base number of channels (doubled at each level)
        depth: Depth of the U-Net
        dimension: '2d' or '3d'
        bilinear: Use bilinear upsampling instead of transposed conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        dimension: Literal['2d', '3d'] = '2d',
        bilinear: bool = False
    ):
        super().__init__()
        self.depth = depth
        self.dimension = dimension

        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d

        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels, dimension)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            self.down_blocks.append(Down(channels, channels * 2, dimension))
            channels *= 2

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(Up(channels, channels // 2, dimension, bilinear))
            channels //= 2

        # Final convolution
        self.outc = Conv(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.inc(x)
        skips = [x]

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        # Remove last skip (bottom of U)
        skips.pop()

        # Decoder
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            x = up(x, skip)

        # Final output
        logits = self.outc(x)
        return logits


class AttentionUNet(nn.Module):
    """
    U-Net with attention gates.

    Attention gates help the model focus on relevant features.
    """

    class AttentionGate(nn.Module):
        """Attention gate module."""

        def __init__(
            self,
            gate_channels: int,
            skip_channels: int,
            inter_channels: int,
            dimension: Literal['2d', '3d'] = '2d'
        ):
            super().__init__()

            Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d

            self.W_g = nn.Sequential(
                Conv(gate_channels, inter_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(inter_channels) if dimension == '2d' else nn.BatchNorm3d(inter_channels)
            )

            self.W_x = nn.Sequential(
                Conv(skip_channels, inter_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(inter_channels) if dimension == '2d' else nn.BatchNorm3d(inter_channels)
            )

            self.psi = nn.Sequential(
                Conv(inter_channels, 1, kernel_size=1, bias=True),
                nn.BatchNorm2d(1) if dimension == '2d' else nn.BatchNorm3d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, gate, skip):
            g1 = self.W_g(gate)
            x1 = self.W_x(skip)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return skip * psi

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        dimension: Literal['2d', '3d'] = '2d'
    ):
        super().__init__()
        self.depth = depth

        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d

        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels, dimension)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            self.down_blocks.append(Down(channels, channels * 2, dimension))
            channels *= 2

        # Attention gates
        self.attention_gates = nn.ModuleList()
        for i in range(depth):
            self.attention_gates.append(
                self.AttentionGate(channels, channels // 2, channels // 4, dimension)
            )

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(Up(channels, channels // 2, dimension, bilinear=False))
            channels //= 2

        # Output
        self.outc = Conv(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.inc(x)
        skips = [x]

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        skips.pop()

        # Decoder with attention
        for i, (attention, up) in enumerate(zip(self.attention_gates, self.up_blocks)):
            skip = skips[-(i + 1)]
            skip = attention(x, skip)
            x = up(x, skip)

        logits = self.outc(x)
        return logits


# Node wrappers

@NodeRegistry.register('networks', 'UNet2D',
                      description='2D U-Net for image segmentation')
class UNet2DNode(PyTorchModuleNode):
    """2D U-Net node."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)  # For inference/forward pass
        self.add_output('model', DataType.MODEL)    # For training (pass model to trainer/optimizer)

    def build_module(self) -> nn.Module:
        return UNet(
            in_channels=self.get_config('in_channels', 1),
            out_channels=self.get_config('out_channels', 2),
            base_channels=self.get_config('base_channels', 64),
            depth=self.get_config('depth', 4),
            dimension='2d',
            bilinear=self.get_config('bilinear', False)
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
            print(f"Error in UNet2DNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'in_channels': {'type': 'text', 'label': 'Input Channels', 'default': '1'},
            'out_channels': {'type': 'text', 'label': 'Output Channels', 'default': '2'},
            'base_channels': {'type': 'text', 'label': 'Base Channels', 'default': '64'},
            'depth': {'type': 'text', 'label': 'Network Depth', 'default': '4'},
            'bilinear': {
                'type': 'choice',
                'label': 'Bilinear Upsampling',
                'choices': ['False', 'True'],
                'default': 'False'
            }
        }


@NodeRegistry.register('networks', 'UNet3D',
                      description='3D U-Net for volumetric segmentation')
class UNet3DNode(PyTorchModuleNode):
    """3D U-Net node."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)  # For inference
        self.add_output('model', DataType.MODEL)    # For training

    def build_module(self) -> nn.Module:
        return UNet(
            in_channels=self.get_config('in_channels', 1),
            out_channels=self.get_config('out_channels', 2),
            base_channels=self.get_config('base_channels', 32),
            depth=self.get_config('depth', 3),
            dimension='3d',
            bilinear=self.get_config('bilinear', False)
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
            print(f"Error in UNet3DNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'in_channels': {'type': 'text', 'label': 'Input Channels', 'default': '1'},
            'out_channels': {'type': 'text', 'label': 'Output Channels', 'default': '2'},
            'base_channels': {'type': 'text', 'label': 'Base Channels', 'default': '32'},
            'depth': {'type': 'text', 'label': 'Network Depth', 'default': '3'},
            'bilinear': {
                'type': 'choice',
                'label': 'Bilinear Upsampling',
                'choices': ['False', 'True'],
                'default': 'False'
            }
        }


@NodeRegistry.register('networks', 'AttentionUNet2D',
                      description='2D Attention U-Net with attention gates')
class AttentionUNet2DNode(PyTorchModuleNode):
    """2D Attention U-Net node."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)  # For inference
        self.add_output('model', DataType.MODEL)    # For training

    def build_module(self) -> nn.Module:
        return AttentionUNet(
            in_channels=self.get_config('in_channels', 1),
            out_channels=self.get_config('out_channels', 2),
            base_channels=self.get_config('base_channels', 64),
            depth=self.get_config('depth', 4),
            dimension='2d'
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
            print(f"Error in AttentionUNet2DNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'in_channels': {'type': 'text', 'label': 'Input Channels', 'default': '1'},
            'out_channels': {'type': 'text', 'label': 'Output Channels', 'default': '2'},
            'base_channels': {'type': 'text', 'label': 'Base Channels', 'default': '64'},
            'depth': {'type': 'text', 'label': 'Network Depth', 'default': '4'}
        }
