# scripts/train_spiking.py
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.qf_neuroshield.data.dataset import DeepfakeVideoDataset
from src.qf_neuroshield.models.spiking_head import SpikingDeepfakeDetector
from scripts.train_baseline import evaluate 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    with open("configs/baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("=== Stage 6: Neuromorphic / Spiking SNN Component (Expt E5) ===")
    print("=================================================================")
    
    train_ds = DeepfakeVideoDataset(
        cfg["dataset"]["metadata_csv"], split="train",
        num_frames=cfg["dataset"]["num_frames"],
        image_size=cfg["dataset"]["image_size"],
        crop_face=cfg["dataset"]["crop_face"]
    )
    test_ds = DeepfakeVideoDataset(
        cfg["dataset"]["metadata_csv"], split="test",
        num_frames=cfg["dataset"]["num_frames"],
        image_size=cfg["dataset"]["image_size"],
        crop_face=cfg["dataset"]["crop_face"]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    num_time_steps = 10
    print(f"Initializing Spiking SNN ViT (Time Steps: {num_time_steps})...")
    model = SpikingDeepfakeDetector(
        pretrained_name=cfg["model"]["pretrained_name"],
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        num_steps=num_time_steps
    ).to(device)

    trainable_params = count_parameters(model)
    print(f"Trainable Parameters (SNN fc1 + fc2): {trainable_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"]
    )

    epochs = 3
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_acc, val_f1, val_auc = evaluate(model, test_loader, criterion, device)
        print(f"  [Epoch {epoch}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Test AUC: {val_auc:.2f} | Test F1: {val_f1:.2f}")

    total_time = time.time() - start_time
    print("\n--- Experiment E5: Neuromorphic Cost Analysis ---")
    print(f"Total Training Time (3 Epochs): {total_time:.2f} seconds")
    print(f"Avg Time per Epoch: {total_time/epochs:.2f} seconds")
    
    save_path = Path("checkpoints/spiking_vit.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Neuromorphic SNN Model saved to: {save_path}")
    print("=================================================================")

if __name__ == "__main__":
    main()