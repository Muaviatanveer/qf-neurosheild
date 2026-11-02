# src/qf_neuroshield/models/vit_detector.py
import torch
import torch.nn as nn
from transformers import ViTModel

class ViTDeepfakeDetector(nn.Module):
    def __init__(self, pretrained_name: str = "google/vit-base-patch16-224-in21k", num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(pretrained_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for vit-base

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classical classification head (Can later be replaced with Quantum/Spiking heads)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def extract_video_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Tensor of shape [Batch, Num_Frames, Channels, Height, Width]
        Returns:
            video_features: Tensor of shape [Batch, Hidden_Size]
        """
        b, t, c, h, w = frames.shape
        flat_frames = frames.view(b * t, c, h, w)

        outputs = self.backbone(pixel_values=flat_frames)
        # Extract the [CLS] token representation for each frame
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # [B * T, Hidden_Size]

        # Reshape back to [B, T, Hidden_Size] and temporally pool across frames
        frame_features = cls_tokens.view(b, t, -1)
        video_features = frame_features.mean(dim=1)      # [B, Hidden_Size]
        return video_features

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        video_features = self.extract_video_features(frames)
        logits = self.classifier(video_features)
        return logits