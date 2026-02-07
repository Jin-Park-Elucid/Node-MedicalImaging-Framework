"""
TransUNet - Transformers Make Strong Encoders for Medical Image Segmentation.

Paper: https://arxiv.org/abs/2102.04306
Combines CNN encoder with Transformer and U-Net decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class PatchEmbed(nn.Module):
    """Convert image to patches and embed them."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and FFN."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x


class CNNEncoder(nn.Module):
    """CNN encoder for extracting multi-scale features."""

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x = self.pool1(x1)

        x2 = self.layer2(x)
        x = self.pool2(x2)

        x3 = self.layer3(x)

        return [x1, x2, x3]


class DecoderBlock(nn.Module):
    """Decoder block with skip connection."""

    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class TransUNet(nn.Module):
    """
    TransUNet - Hybrid CNN-Transformer architecture.

    Features:
    - CNN encoder for low-level features
    - Transformer for global context
    - U-Net style decoder with skip connections
    - State-of-the-art for medical image segmentation
    """

    def __init__(
        self,
        img_size=224,
        in_channels=1,
        out_channels=2,
        base_channels=64,
        embed_dim=512,
        num_heads=8,
        num_layers=12,
        patch_size=16
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        # CNN Encoder
        self.cnn_encoder = CNNEncoder(in_channels, base_channels)

        # Patch embedding for transformer
        self.patch_embed = PatchEmbed(
            img_size=img_size // 4,  # After CNN encoder
            patch_size=patch_size,
            in_channels=base_channels * 4,
            embed_dim=embed_dim
        )

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        # Decoder
        self.decoder3 = DecoderBlock(embed_dim, base_channels * 4, base_channels * 4)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels)

        # Output
        self.output = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        # CNN encoder
        skip_features = self.cnn_encoder(x)  # [x1, x2, x3]

        # Patch embedding
        x = self.patch_embed(skip_features[-1])

        # Transformer
        x = self.transformer(x)

        # Reshape back to spatial
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Decoder with skip connections
        x = self.decoder3(x, skip_features[2])
        x = self.decoder2(x, skip_features[1])
        x = self.decoder1(x, skip_features[0])

        # Output
        x = self.output(x)

        return x


@NodeRegistry.register('networks', 'TransUNet',
                      description='TransUNet - Transformer + U-Net for segmentation')
class TransUNetNode(PyTorchModuleNode):
    """TransUNet node combining transformers with U-Net."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)
        self.add_output('model', DataType.MODEL)    # For training

    def build_module(self) -> nn.Module:
        return TransUNet(
            img_size=self.get_config('img_size', 224),
            in_channels=self.get_config('in_channels', 1),
            out_channels=self.get_config('out_channels', 2),
            base_channels=self.get_config('base_channels', 64),
            embed_dim=self.get_config('embed_dim', 512),
            num_heads=self.get_config('num_heads', 8),
            num_layers=self.get_config('num_layers', 12)
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
            print(f"Error in TransUNetNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'img_size': {
                'type': 'text',
                'label': 'Image Size',
                'default': '224'
            },
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
                'default': '64'
            },
            'embed_dim': {
                'type': 'text',
                'label': 'Embedding Dimension',
                'default': '512'
            },
            'num_heads': {
                'type': 'text',
                'label': 'Number of Attention Heads',
                'default': '8'
            },
            'num_layers': {
                'type': 'text',
                'label': 'Number of Transformer Layers',
                'default': '12'
            }
        }
