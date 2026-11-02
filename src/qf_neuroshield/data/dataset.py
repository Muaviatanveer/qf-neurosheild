# src/qf_neuroshield/data/dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.qf_neuroshield.data.frame_sampler import VideoFrameSampler

class DeepfakeVideoDataset(Dataset):
    LABEL_MAP = {"REAL": 0, "FAKE": 1}

    def __init__(self, metadata_csv: str, split: str, num_frames: int = 8, image_size: int = 224, crop_face: bool = True):
        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        
        self.sampler = VideoFrameSampler(
            num_frames=num_frames,
            image_size=image_size,
            crop_face=crop_face
        )
        
        # Standard ViT normalization ([-1, 1])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = row["filepath"]
        label_str = row["label"]
        video_id = row["video_id"]

        frames_pil = self.sampler.sample(video_path)
        # Shape: [T, C, H, W] where T = num_frames
        frames_tensor = torch.stack([self.transform(img) for img in frames_pil])
        label_tensor = torch.tensor(self.LABEL_MAP[label_str], dtype=torch.long)

        return {
            "video_id": video_id,
            "frames": frames_tensor,
            "label": label_tensor
        }