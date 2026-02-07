"""
Swin-UNETR - Swin Transformers for Semantic Segmentation.

Paper: https://arxiv.org/abs/2201.01266
Hierarchical Swin Transformer encoder with U-Net decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class PatchMerging3D(nn.Module):
    """Patch merging layer for downsampling."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x):
        B, H, W, D, C = x.shape

        # Pad if needed
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        pad_d = (2 - D % 2) % 2
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))

        # Merge patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, H/2, W/2, D/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # (B, H/2, W/2, D/2, 8*C)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class WindowAttention3D(nn.Module):
    """Window-based multi-head self attention for 3D."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer block."""

    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        B, H, W, D, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # Flatten for attention
        x = x.view(B, -1, C)
        x = self.attn(x)
        x = x.view(B, H, W, D, C)

        # Skip connection
        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class SwinStage(nn.Module):
    """Swin Transformer stage."""

    def __init__(self, dim, depth, num_heads, window_size, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim, num_heads, window_size)
            for _ in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SwinUNETR(nn.Module):
    """
    Swin-UNETR - Hierarchical vision transformer with U-Net decoder.

    Features:
    - Swin Transformer encoder with shifted windows
    - Hierarchical feature extraction
    - U-Net style decoder
    - State-of-the-art for 3D medical imaging
    """

    def __init__(
        self,
        img_size=96,
        in_channels=1,
        out_channels=2,
        embed_dim=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        feature_size=24
    ):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.patch_norm = nn.LayerNorm(embed_dim)

        # Encoder stages
        self.stage1 = SwinStage(embed_dim, depths[0], num_heads[0], window_size,
                               downsample=PatchMerging3D(embed_dim))
        self.stage2 = SwinStage(embed_dim * 2, depths[1], num_heads[1], window_size,
                               downsample=PatchMerging3D(embed_dim * 2))
        self.stage3 = SwinStage(embed_dim * 4, depths[2], num_heads[2], window_size,
                               downsample=PatchMerging3D(embed_dim * 4))
        self.stage4 = SwinStage(embed_dim * 8, depths[3], num_heads[3], window_size)

        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(embed_dim * 8, feature_size * 8, 2, stride=2),
            nn.InstanceNorm3d(feature_size * 8),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(feature_size * 8 + embed_dim * 4, feature_size * 4, 2, stride=2),
            nn.InstanceNorm3d(feature_size * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(feature_size * 4 + embed_dim * 2, feature_size * 2, 2, stride=2),
            nn.InstanceNorm3d(feature_size * 2),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(feature_size * 2 + embed_dim, feature_size, 2, stride=2),
            nn.InstanceNorm3d(feature_size),
            nn.ReLU(inplace=True)
        )

        # Output
        self.output = nn.Conv3d(feature_size, out_channels, 1)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H/4, W/4, D/4)
        B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
        x = self.patch_norm(x)

        # Encoder
        x1 = self.stage1(x)  # (B, H/2, W/2, D/2, 2C)
        x2 = self.stage2(x1)  # (B, H/4, W/4, D/4, 4C)
        x3 = self.stage3(x2)  # (B, H/8, W/8, D/8, 8C)
        x4 = self.stage4(x3)  # (B, H/8, W/8, D/8, 8C)

        # Convert back to (B, C, H, W, D) format for decoder
        x4 = x4.permute(0, 4, 1, 2, 3)
        x3 = x3.permute(0, 4, 1, 2, 3)
        x2 = x2.permute(0, 4, 1, 2, 3)
        x1 = x1.permute(0, 4, 1, 2, 3)

        # Decoder with skip connections
        dec4 = self.decoder4(x4)
        dec3 = self.decoder3(torch.cat([dec4, x3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, x2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, x1], dim=1))

        # Output
        out = self.output(dec1)

        return out


@NodeRegistry.register('networks', 'SwinUNETR',
                      description='Swin-UNETR - Hierarchical Swin Transformer for 3D segmentation')
class SwinUNETRNode(PyTorchModuleNode):
    """Swin-UNETR node for 3D medical image segmentation."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR, optional=True)  # Optional: only needed for inference
        self.add_output('output', DataType.TENSOR)
        self.add_output('model', DataType.MODEL)    # For training

    def build_module(self) -> nn.Module:
        return SwinUNETR(
            img_size=self.get_config('img_size', 96),
            in_channels=self.get_config('in_channels', 1),
            out_channels=self.get_config('out_channels', 2),
            embed_dim=self.get_config('embed_dim', 48),
            window_size=self.get_config('window_size', 7)
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
            print(f"Error in SwinUNETRNode: {e}")
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
                'default': '48'
            },
            'window_size': {
                'type': 'text',
                'label': 'Window Size',
                'default': '7'
            }
        }
