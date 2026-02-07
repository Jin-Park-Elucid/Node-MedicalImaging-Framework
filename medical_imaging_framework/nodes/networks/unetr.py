"""
UNETR - UNEt TRansformers for Medical Image Segmentation.

Paper: https://arxiv.org/abs/2103.10504
Pure transformer encoder with CNN decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class PatchEmbedding3D(nn.Module):
    """3D patch embedding for volumetric images."""

    def __init__(self, img_size=96, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 3

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P, D/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder block."""

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
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock3D(nn.Module):
    """3D decoder block with skip connection."""

    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNETR(nn.Module):
    """
    UNETR - Pure transformer encoder with CNN decoder.

    Features:
    - Vision Transformer encoder
    - Multi-scale skip connections from transformer layers
    - CNN decoder
    - Designed for 3D medical images
    """

    def __init__(
        self,
        img_size=96,
        in_channels=1,
        out_channels=2,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        patch_size=16,
        feature_size=16
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)

        # Transformer encoder
        self.transformers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Projection layers for skip connections (at layers 3, 6, 9, 12)
        self.proj_layers = nn.ModuleDict({
            'z3': nn.Conv3d(embed_dim, feature_size * 2, 1),
            'z6': nn.Conv3d(embed_dim, feature_size * 4, 1),
            'z9': nn.Conv3d(embed_dim, feature_size * 8, 1),
            'z12': nn.Conv3d(embed_dim, feature_size * 16, 1)
        })

        # CNN encoder for fine details
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(in_channels, feature_size, 3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, feature_size, 3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder4 = DecoderBlock3D(feature_size * 16, feature_size * 8, feature_size * 8)
        self.decoder3 = DecoderBlock3D(feature_size * 8, feature_size * 4, feature_size * 4)
        self.decoder2 = DecoderBlock3D(feature_size * 4, feature_size * 2, feature_size * 2)
        self.decoder1 = DecoderBlock3D(feature_size * 2, feature_size, feature_size)

        # Output
        self.output = nn.Conv3d(feature_size, out_channels, 1)

    def forward(self, x):
        B, C, H, W, D = x.shape

        # CNN encoder for skip connection
        z0 = self.cnn_encoder(x)

        # Patch embedding
        x_embed = self.patch_embed(x)

        # Transformer encoder with intermediate outputs
        hidden_states = []
        for i, transformer in enumerate(self.transformers):
            x_embed = transformer(x_embed)
            if (i + 1) in [3, 6, 9, 12]:
                hidden_states.append(x_embed)

        # Reshape hidden states back to 3D
        patch_dim = self.img_size // self.patch_size
        z3 = hidden_states[0].transpose(1, 2).reshape(B, self.embed_dim, patch_dim, patch_dim, patch_dim)
        z6 = hidden_states[1].transpose(1, 2).reshape(B, self.embed_dim, patch_dim, patch_dim, patch_dim)
        z9 = hidden_states[2].transpose(1, 2).reshape(B, self.embed_dim, patch_dim, patch_dim, patch_dim)
        z12 = hidden_states[3].transpose(1, 2).reshape(B, self.embed_dim, patch_dim, patch_dim, patch_dim)

        # Project transformer features
        z3 = self.proj_layers['z3'](z3)
        z6 = self.proj_layers['z6'](z6)
        z9 = self.proj_layers['z9'](z9)
        z12 = self.proj_layers['z12'](z12)

        # Decoder with skip connections
        x = self.decoder4(z12, z9)
        x = self.decoder3(x, z6)
        x = self.decoder2(x, z3)
        x = self.decoder1(x, z0)

        # Output
        x = self.output(x)

        return x


@NodeRegistry.register('networks', 'UNETR',
                      description='UNETR - Pure Transformer encoder for 3D segmentation')
class UNETRNode(PyTorchModuleNode):
    """UNETR node for 3D medical image segmentation."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)
        self.add_output('model', DataType.MODEL)    # For training

    def build_module(self) -> nn.Module:
        return UNETR(
            img_size=self.get_config('img_size', 96),
            in_channels=self.get_config('in_channels', 1),
            out_channels=self.get_config('out_channels', 2),
            embed_dim=self.get_config('embed_dim', 768),
            num_heads=self.get_config('num_heads', 12),
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
            print(f"Error in UNETRNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'img_size': {
                'type': 'text',
                'label': 'Image Size',
                'default': '96'
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
            'embed_dim': {
                'type': 'text',
                'label': 'Embedding Dimension',
                'default': '768'
            },
            'num_heads': {
                'type': 'text',
                'label': 'Number of Attention Heads',
                'default': '12'
            },
            'num_layers': {
                'type': 'text',
                'label': 'Number of Transformer Layers',
                'default': '12'
            }
        }
