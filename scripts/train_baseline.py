# scripts/train_baseline.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.qf_neuroshield.data.dataset import DeepfakeVideoDataset
from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            logits = model(frames)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return total_loss / max(1, len(loader)), acc, f1, auc

def main():
    with open("configs/baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = DeepfakeVideoDataset(
        cfg["dataset"]["metadata_csv"], split="train",
        num_frames=cfg["dataset"]["num_frames"],
        image_size=cfg["dataset"]["image_size"],
        crop_face=cfg["dataset"]["crop_face"]
    )
    val_ds = DeepfakeVideoDataset(
        cfg["dataset"]["metadata_csv"], split="val",
        num_frames=cfg["dataset"]["num_frames"],
        image_size=cfg["dataset"]["image_size"],
        crop_face=cfg["dataset"]["crop_face"]
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    print("Loading Vision Transformer...")
    model = ViTDeepfakeDetector(
        pretrained_name=cfg["model"]["pretrained_name"],
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"]
    )

    for epoch in range(1, cfg["training"]["epochs"] + 1):
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

        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch}/{cfg['training']['epochs']}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f} | Val F1: {val_f1:.2f} | Val AUC: {val_auc:.2f}")

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "baseline_vit.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel saved to {ckpt_path}")

    # Milestone 1 Verification: Run inference on a single held-out test video
    test_ds = DeepfakeVideoDataset(
        cfg["dataset"]["metadata_csv"], split="test",
        num_frames=cfg["dataset"]["num_frames"],
        image_size=cfg["dataset"]["image_size"],
        crop_face=cfg["dataset"]["crop_face"]
    )
    sample = test_ds[0]
    model.eval()
    with torch.no_grad():
        logits = model(sample["frames"].unsqueeze(0).to(device))
        fake_prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred_label = "FAKE" if fake_prob >= 0.5 else "REAL"

    prediction_output = {
        "video_id": sample["video_id"],
        "model_version": "baseline-vit-v1",
        "predicted_label": pred_label,
        "fake_probability": round(fake_prob, 4),
        "ground_truth": "FAKE" if sample["label"].item() == 1 else "REAL",
        "calibration_status": "not_yet_calibrated"
    }
    print("\n--- Milestone 1: Single Video Inference Output ---")
    print(prediction_output)

if __name__ == "__main__":
    main()