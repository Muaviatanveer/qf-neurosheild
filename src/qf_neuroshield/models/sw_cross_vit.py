import math
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class DifferentiableHaarDWT2D(nn.Module):
    """
    GPU-accelerated 2-Level 2D Discrete Haar Wavelet Transform (DWT).
    Decomposes RGB images [B, 3, H, W] into orthogonal LL, LH, HL, HH sub-bands
    using fixed non-trainable strided depthwise convolutions.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

        # 2x2 Orthogonal Haar Wavelet Basis
        ll = torch.tensor([[ 0.5,  0.5], [ 0.5,  0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [ 0.5,  0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5,  0.5], [-0.5,  0.5]], dtype=torch.float32)
        hh = torch.tensor([[ 0.5, -0.5], [-0.5,  0.5]], dtype=torch.float32)

        # Stack filters for grouped depthwise conv: [4 * C, 1, 2, 2]
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        weight = kernels.repeat(in_channels, 1, 1, 1)                 # [12, 1, 2, 2]
        self.register_buffer("haar_weight", weight)

    def _dwt_single_level(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, H, W] -> out: [B, 4*C, H/2, W/2]
        out = F.conv2d(x, self.haar_weight, stride=2, groups=self.in_channels)
        b, _, h2, w2 = out.shape
        out = out.view(b, self.in_channels, 4, h2, w2)
        ll = out[:, :, 0, :, :]                                       # [B, 3, H/2, W/2]
        high_freq = out[:, :, 1:, :, :].reshape(b, self.in_channels * 3, h2, w2)  # [B, 9, H/2, W/2]
        return ll, high_freq

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          hf_level1: [B, 9, 112, 112] (Fine 1-px GAN upsampling & sensor noise sub-bands)
          hf_level2: [B, 9, 56, 56]   (Coarse 4-px JPEG block & diffusion sub-bands)
        """
        ll_1, hf_level1 = self._dwt_single_level(x)
        _, hf_level2 = self._dwt_single_level(ll_1)
        return hf_level1, hf_level2


class WaveletPatchTokenizer(nn.Module):
    """
    Projects Level-1 (112x112) and Level-2 (56x56) Wavelet High-Pass Sub-bands
    into a 14x14 = 196 token grid [B, 196, 768] that is 1-to-1 spatially aligned
    with the 196 patch tokens of ViT-Base-Patch16-224.
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.l1_stem = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # Fuse Level-1 (64 ch) + Level-2 (9 ch) = 73 channels at 56x56 resolution
        self.proj_net = nn.Sequential(
            nn.Conv2d(64 + 9, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, hf_l1: torch.Tensor, hf_l2: torch.Tensor) -> torch.Tensor:
        x1 = self.l1_stem(hf_l1)                  # [B, 64, 56, 56]
        fused = torch.cat([x1, hf_l2], dim=1)     # [B, 73, 56, 56]
        tokens_2d = self.proj_net(fused)          # [B, 768, 14, 14]
        tokens = tokens_2d.flatten(2).transpose(1, 2)  # [B, 196, 768]
        return self.norm(tokens)


class SWCrossViTDetector(nn.Module):
    """
    SW-CrossViT: Spatial-Wavelet Cross-Attention Vision Transformer.
    Combines:
      1. ViT-Base-Patch16-224 Spatial Semantic Stream (196 Patch Tokens + CLS)
      2. 2-Level Differentiable Haar Wavelet High-Pass Stream (196 Aligned Wavelet Tokens)
      3. 8-Head Spatial-to-Wavelet Cross-Attention + Discrepancy Residual |Z_s - Z_w|
    """
    def __init__(
        self,
        pretrained_name: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 2,
        unfreeze_top_layers: int = 2
    ):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(pretrained_name)
        hidden_size = self.backbone.config.hidden_size  # 768

        # Freeze backbone, then selectively unfreeze top transformer blocks
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_top_layers > 0:
            backbone_params = list(self.backbone.parameters())
            for p in backbone_params[-(unfreeze_top_layers * 16):]:
                p.requires_grad = True

        self.dwt = DifferentiableHaarDWT2D(in_channels=3)
        self.wavelet_tokenizer = WaveletPatchTokenizer(embed_dim=hidden_size)

        self.norm_spatial = nn.LayerNorm(hidden_size)
        self.norm_wavelet = nn.LayerNorm(hidden_size)

        # 8-Head Cross-Attention: Query = Spatial Patches, Key/Value = Wavelet High-Pass Patches
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_size)

        # Fusion Classifier: [CLS (768) || Cross-Attended Wavelet (768) || Discrepancy |Zs - Zw| (768)] = 2304
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward_with_telemetry(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)  # [B, 1, C, H, W]
        b, t, c, h, w = frames.shape
        flat_frames = frames.view(b * t, c, h, w)

        # 1. Spatial ViT Stream
        vit_out = self.backbone(pixel_values=flat_frames)
        hidden_states = vit_out.last_hidden_state           # [B*T, 197, 768]
        cls_token = hidden_states[:, 0, :]                  # [B*T, 768]
        spatial_patches = self.norm_spatial(hidden_states[:, 1:, :])  # [B*T, 196, 768]

        # 2. Multi-Level 2D Haar Wavelet Stream
        hf_l1, hf_l2 = self.dwt(flat_frames)
        wavelet_patches = self.norm_wavelet(self.wavelet_tokenizer(hf_l1, hf_l2))  # [B*T, 196, 768]

        # 3. Spatial-Wavelet Cross-Attention
        cross_out, attn_weights = self.cross_attn(
            query=spatial_patches,
            key=wavelet_patches,
            value=wavelet_patches,
            need_weights=True,
            average_attn_weights=True
        )
        cross_tokens = self.norm_cross(spatial_patches + cross_out)  # [B*T, 196, 768]
        cross_pooled = cross_tokens.mean(dim=1)                      # [B*T, 768]

        # 4. Spatial-Frequency Discrepancy Residual |Z_spatial - Z_wavelet|
        discrepancy_tokens = torch.abs(spatial_patches - wavelet_patches)
        discrepancy_pooled = discrepancy_tokens.mean(dim=1)          # [B*T, 768]

        # 5. Temporal Pooling across T frames & Classification
        fused = torch.cat([cls_token, cross_pooled, discrepancy_pooled], dim=-1)  # [B*T, 2304]
        fused_video = fused.view(b, t, -1).mean(dim=1)                            # [B, 2304]
        logits = self.classifier(fused_video)                                     # [B, 2]

        # Patch discrepancy heatmap [B, 14, 14] showing which facial regions have synthetic frequency artifacts
        patch_discrepancy_map = discrepancy_tokens.mean(dim=-1).view(b, t, 14, 14).mean(dim=1)

        telemetry = {
            "cls_features": cls_token.view(b, t, -1).mean(dim=1),
            "wavelet_energy_l1": hf_l1.abs().mean(dim=(1, 2, 3)).view(b, t).mean(dim=1),
            "wavelet_energy_l2": hf_l2.abs().mean(dim=(1, 2, 3)).view(b, t).mean(dim=1),
            "discrepancy_norm": discrepancy_pooled.norm(dim=-1).view(b, t).mean(dim=1),
            "patch_discrepancy_map": patch_discrepancy_map
        }
        return logits, telemetry

    def extract_video_features(self, frames: torch.Tensor) -> torch.Tensor:
        _, telemetry = self.forward_with_telemetry(frames)
        return telemetry["cls_features"]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_telemetry(frames)
        return logits
