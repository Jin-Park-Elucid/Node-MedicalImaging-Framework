"""
ResNet building blocks for 2D and 3D medical imaging.

Provides encoder and decoder blocks based on ResNet architecture,
supporting both 2D and 3D convolutions for various medical imaging tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class ResNetBlock(nn.Module):
    """
    Basic ResNet block with skip connection.

    Supports both 2D and 3D convolutions with optional downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dimension: Literal['2d', '3d'] = '2d',
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if dimension == '2d' else nn.BatchNorm3d

        self.conv1 = Conv(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Conv(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = BatchNorm(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    """
    ResNet bottleneck block (1x1 -> 3x3 -> 1x1).

    More efficient for deeper networks.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        stride: int = 1,
        dimension: Literal['2d', '3d'] = '2d',
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if dimension == '2d' else nn.BatchNorm3d

        out_channels = base_channels * self.expansion

        self.conv1 = Conv(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(base_channels)

        self.conv2 = Conv(
            base_channels, base_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm(base_channels)

        self.conv3 = Conv(base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder.

    Progressively downsamples spatial dimensions while increasing channels.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        layers: list = [2, 2, 2, 2],
        dimension: Literal['2d', '3d'] = '2d',
        use_bottleneck: bool = False
    ):
        super().__init__()

        self.dimension = dimension
        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if dimension == '2d' else nn.BatchNorm3d
        MaxPool = nn.MaxPool2d if dimension == '2d' else nn.MaxPool3d

        block = ResNetBottleneck if use_bottleneck else ResNetBlock
        self.expansion = block.expansion if use_bottleneck else 1

        self.in_channels = base_channels

        # Initial convolution
        self.conv1 = Conv(
            in_channels, base_channels, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = MaxPool(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

    def _make_layer(
        self,
        block: nn.Module,
        channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a layer with multiple blocks."""
        Conv = nn.Conv2d if self.dimension == '2d' else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if self.dimension == '2d' else nn.BatchNorm3d

        downsample = None
        out_channels = channels * self.expansion

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                Conv(self.in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False),
                BatchNorm(out_channels)
            )

        layers = []
        layers.append(
            block(self.in_channels, channels, stride, self.dimension, downsample)
        )
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, dimension=self.dimension))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning multi-scale features.

        Returns:
            Dictionary with keys: 'out', 'skip1', 'skip2', 'skip3', 'skip4'
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip0 = x

        x = self.maxpool(x)
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        skip4 = self.layer4(skip3)

        return {
            'out': skip4,
            'skip0': skip0,
            'skip1': skip1,
            'skip2': skip2,
            'skip3': skip3,
            'skip4': skip4
        }


class ResNetDecoder(nn.Module):
    """
    ResNet-based decoder with skip connections.

    Upsamples spatial dimensions while reducing channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: list = [256, 128, 64, 64],
        dimension: Literal['2d', '3d'] = '2d'
    ):
        super().__init__()

        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if dimension == '2d' else nn.ConvTranspose3d
        BatchNorm = nn.BatchNorm2d if dimension == '2d' else nn.BatchNorm3d

        # Upsampling layers
        self.up1 = ConvTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            Conv(in_channels // 2 + skip_channels[0], in_channels // 2,
                 kernel_size=3, padding=1),
            BatchNorm(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.up2 = ConvTranspose(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            Conv(in_channels // 4 + skip_channels[1], in_channels // 4,
                 kernel_size=3, padding=1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.up3 = ConvTranspose(in_channels // 4, in_channels // 8, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            Conv(in_channels // 8 + skip_channels[2], in_channels // 8,
                 kernel_size=3, padding=1),
            BatchNorm(in_channels // 8),
            nn.ReLU(inplace=True)
        )

        self.up4 = ConvTranspose(in_channels // 8, in_channels // 16, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            Conv(in_channels // 16 + skip_channels[3], in_channels // 16,
                 kernel_size=3, padding=1),
            BatchNorm(in_channels // 16),
            nn.ReLU(inplace=True)
        )

        # Final output
        self.final = Conv(in_channels // 16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: dict) -> torch.Tensor:
        """
        Forward pass with skip connections.

        Args:
            x: Input tensor
            skips: Dictionary of skip connection tensors

        Returns:
            Decoded output
        """
        x = self.up1(x)
        x = torch.cat([x, skips['skip3']], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, skips['skip2']], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, skips['skip1']], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, skips['skip0']], dim=1)
        x = self.conv4(x)

        x = self.final(x)

        return x


# Node wrappers for the framework

@NodeRegistry.register('networks', 'ResNetEncoder2D',
                      description='2D ResNet encoder with skip connections')
class ResNetEncoder2DNode(PyTorchModuleNode):
    """Node wrapper for 2D ResNet encoder."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('features', DataType.TENSOR)
        self.add_output('skip_connections', DataType.ANY)

    def build_module(self) -> nn.Module:
        return ResNetEncoder(
            in_channels=self.get_config('in_channels', 1),
            base_channels=self.get_config('base_channels', 64),
            layers=self.get_config('layers', [2, 2, 2, 2]),
            dimension='2d',
            use_bottleneck=self.get_config('use_bottleneck', False)
        )

    def execute(self) -> bool:
        try:
            if self.module is None:
                self.initialize_module()

            x = self.get_input_value('input')
            if x is None:
                return False

            with torch.set_grad_enabled(self.training_mode):
                output = self.module(x)

            self.set_output_value('features', output['out'])
            self.set_output_value('skip_connections', output)

            return True
        except Exception as e:
            print(f"Error in ResNetEncoder2DNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'in_channels': {'type': 'text', 'label': 'Input Channels', 'default': '1'},
            'base_channels': {'type': 'text', 'label': 'Base Channels', 'default': '64'},
            'use_bottleneck': {
                'type': 'choice',
                'label': 'Use Bottleneck',
                'choices': ['False', 'True'],
                'default': 'False'
            }
        }


@NodeRegistry.register('networks', 'ResNetEncoder3D',
                      description='3D ResNet encoder for volumetric data')
class ResNetEncoder3DNode(PyTorchModuleNode):
    """Node wrapper for 3D ResNet encoder."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('features', DataType.TENSOR)
        self.add_output('skip_connections', DataType.ANY)

    def build_module(self) -> nn.Module:
        return ResNetEncoder(
            in_channels=self.get_config('in_channels', 1),
            base_channels=self.get_config('base_channels', 32),
            layers=self.get_config('layers', [1, 2, 2, 2]),
            dimension='3d',
            use_bottleneck=self.get_config('use_bottleneck', False)
        )

    def execute(self) -> bool:
        try:
            if self.module is None:
                self.initialize_module()

            x = self.get_input_value('input')
            if x is None:
                return False

            with torch.set_grad_enabled(self.training_mode):
                output = self.module(x)

            self.set_output_value('features', output['out'])
            self.set_output_value('skip_connections', output)

            return True
        except Exception as e:
            print(f"Error in ResNetEncoder3DNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'in_channels': {'type': 'text', 'label': 'Input Channels', 'default': '1'},
            'base_channels': {'type': 'text', 'label': 'Base Channels', 'default': '32'},
            'use_bottleneck': {
                'type': 'choice',
                'label': 'Use Bottleneck',
                'choices': ['False', 'True'],
                'default': 'False'
            }
        }


@NodeRegistry.register('networks', 'ResNetDecoder2D',
                      description='2D ResNet decoder with skip connections')
class ResNetDecoder2DNode(PyTorchModuleNode):
    """Node wrapper for 2D ResNet decoder."""

    def _setup_ports(self):
        self.add_input('features', DataType.TENSOR)
        self.add_input('skip_connections', DataType.ANY)
        self.add_output('output', DataType.TENSOR)

    def build_module(self) -> nn.Module:
        return ResNetDecoder(
            in_channels=self.get_config('in_channels', 512),
            out_channels=self.get_config('out_channels', 2),
            skip_channels=self.get_config('skip_channels', [256, 128, 64, 64]),
            dimension='2d'
        )

    def execute(self) -> bool:
        try:
            if self.module is None:
                self.initialize_module()

            features = self.get_input_value('features')
            skips = self.get_input_value('skip_connections')

            if features is None or skips is None:
                return False

            with torch.set_grad_enabled(self.training_mode):
                output = self.module(features, skips)

            self.set_output_value('output', output)

            return True
        except Exception as e:
            print(f"Error in ResNetDecoder2DNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'in_channels': {'type': 'text', 'label': 'Input Channels', 'default': '512'},
            'out_channels': {'type': 'text', 'label': 'Output Channels', 'default': '2'}
        }
