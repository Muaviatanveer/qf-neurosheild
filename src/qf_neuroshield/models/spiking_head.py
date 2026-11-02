# src/qf_neuroshield/models/spiking_head.py
import torch
import torch.nn as nn
from transformers import ViTModel
import snntorch as snn
from snntorch import surrogate

class SpikingDeepfakeDetector(nn.Module):
    def __init__(self, pretrained_name: str = "google/vit-base-patch16-224-in21k", num_classes: int = 2, freeze_backbone: bool = True, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        self.backbone = ViTModel.from_pretrained(pretrained_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # SNN Hyperparameters
        beta = 0.9  # Voltage decay rate
        spike_grad = surrogate.fast_sigmoid(slope=25)  # Surrogate gradient for backprop

        # 1st Spiking Layer
        self.fc1 = nn.Linear(hidden_size, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # 2nd Spiking Layer (Output)
        self.fc2 = nn.Linear(128, num_classes)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def extract_video_features(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        flat_frames = frames.view(b * t, c, h, w)
        outputs = self.backbone(pixel_values=flat_frames)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        frame_features = cls_tokens.view(b, t, -1)
        video_features = frame_features.mean(dim=1)
        return video_features

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.extract_video_features(frames)

        # Initialize hidden states (membrane potentials) for LIF neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        mem2_rec = []

        # SNNs operate over discrete time steps. We feed the static features 
        # in continuously over num_steps to generate spike trains.
        for step in range(self.num_steps):
            cur1 = self.fc1(features)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Record membrane potential of output layer
            mem2_rec.append(mem2)

        # Stack to [num_steps, batch_size, num_classes]
        mem2_rec = torch.stack(mem2_rec)
        
        # To get logits for CrossEntropyLoss, we sum the output membrane 
        # potentials over all time steps.
        out_logits = mem2_rec.sum(dim=0) 
        return out_logits