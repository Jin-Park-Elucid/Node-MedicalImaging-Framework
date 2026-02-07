"""
Transformer components for medical imaging.

Includes multi-head attention, transformer encoder/decoder blocks,
and vision transformer (ViT) adaptations for medical imaging.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal
from ...core import PyTorchModuleNode, NodeRegistry, DataType


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len, embed_dim)
            key: (batch, seq_len, embed_dim) or None (uses query)
            value: (batch, seq_len, embed_dim) or None (uses query)
            attn_mask: (seq_len, seq_len) or (batch, seq_len, seq_len)
        """
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len, _ = query.shape

        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        k = k.permute(0, 2, 3, 1)  # (batch, num_heads, head_dim, seq_len)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        attn_weights = torch.matmul(q, k) * self.scale  # (batch, num_heads, seq_len, seq_len)

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_heads, head_dim)
        out = out.view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attn_out = self.self_attn(x, attn_mask=self_attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention
        cross_out = self.cross_attn(x, encoder_out, encoder_out, attn_mask=cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + ff_out)

        return x


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, self_attn_mask, cross_attn_mask)
        return x


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        dimension: Literal['2d', '3d'] = '2d'
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** (2 if dimension == '2d' else 3)

        Conv = nn.Conv2d if dimension == '2d' else nn.Conv3d

        self.proj = Conv(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H', W') or (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for medical imaging.

    Adapted for medical image analysis tasks.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        dimension: Literal['2d', '3d'] = '2d'
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim, dimension
        )

        num_patches = self.patch_embed.num_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            depth, embed_dim, num_heads, ff_dim, dropout
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        x = self.transformer(x)

        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Take class token
        return self.head(cls_output)


# Node wrappers

@NodeRegistry.register('networks', 'TransformerEncoder',
                      description='Transformer encoder for sequence processing')
class TransformerEncoderNode(PyTorchModuleNode):
    """Transformer encoder node."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('output', DataType.TENSOR)

    def build_module(self) -> nn.Module:
        return TransformerEncoder(
            num_layers=self.get_config('num_layers', 6),
            embed_dim=self.get_config('embed_dim', 512),
            num_heads=self.get_config('num_heads', 8),
            ff_dim=self.get_config('ff_dim', 2048),
            dropout=self.get_config('dropout', 0.1)
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

            self.set_output_value('output', output)
            return True

        except Exception as e:
            print(f"Error in TransformerEncoderNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'num_layers': {'type': 'text', 'label': 'Number of Layers', 'default': '6'},
            'embed_dim': {'type': 'text', 'label': 'Embedding Dimension', 'default': '512'},
            'num_heads': {'type': 'text', 'label': 'Number of Heads', 'default': '8'},
            'ff_dim': {'type': 'text', 'label': 'Feed-Forward Dimension', 'default': '2048'},
            'dropout': {'type': 'text', 'label': 'Dropout Rate', 'default': '0.1'}
        }


@NodeRegistry.register('networks', 'VisionTransformer2D',
                      description='2D Vision Transformer for image classification')
class VisionTransformer2DNode(PyTorchModuleNode):
    """2D Vision Transformer node."""

    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('output', DataType.TENSOR)

    def build_module(self) -> nn.Module:
        return VisionTransformer(
            img_size=self.get_config('img_size', 224),
            patch_size=self.get_config('patch_size', 16),
            in_channels=self.get_config('in_channels', 1),
            num_classes=self.get_config('num_classes', 2),
            embed_dim=self.get_config('embed_dim', 768),
            depth=self.get_config('depth', 12),
            num_heads=self.get_config('num_heads', 12),
            ff_dim=self.get_config('ff_dim', 3072),
            dropout=self.get_config('dropout', 0.1),
            dimension='2d'
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

            self.set_output_value('output', output)
            return True

        except Exception as e:
            print(f"Error in VisionTransformer2DNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'img_size': {'type': 'text', 'label': 'Image Size', 'default': '224'},
            'patch_size': {'type': 'text', 'label': 'Patch Size', 'default': '16'},
            'in_channels': {'type': 'text', 'label': 'Input Channels', 'default': '1'},
            'num_classes': {'type': 'text', 'label': 'Number of Classes', 'default': '2'},
            'embed_dim': {'type': 'text', 'label': 'Embedding Dimension', 'default': '768'},
            'depth': {'type': 'text', 'label': 'Depth', 'default': '12'},
            'num_heads': {'type': 'text', 'label': 'Number of Heads', 'default': '12'}
        }
