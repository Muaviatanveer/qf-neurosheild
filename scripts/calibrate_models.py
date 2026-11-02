# scripts/calibrate_models.py
import sys
import json
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.qf_neuroshield.data.dataset import DeepfakeVideoDataset
from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.models.quantum_head import HybridQuantumDeepfakeDetector
from src.qf_neuroshield.models.spiking_head import SpikingDeepfakeDetector
from src.qf_neuroshield.training.calibration import fit_threshold_and_scale, apply_calibration

def collect_predictions(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels_list, probs_list = [], []

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            logits = model(frames)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            labels_list.extend(labels.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, np.array(labels_list), np.array(probs_list)

def main():
    with open("configs/baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================================")
    print("=== Phase 1: Post-Hoc Probability Calibration & Threshold Optimization ===")
    print("==========================================================================")

    meta_csv = cfg["dataset"]["metadata_csv"]
    ds_kwargs = {
        "num_frames": cfg["dataset"]["num_frames"],
        "image_size": cfg["dataset"]["image_size"],
        "crop_face": cfg["dataset"]["crop_face"]
    }

    train_ds = DeepfakeVideoDataset(meta_csv, split="train", **ds_kwargs)
    val_ds = DeepfakeVideoDataset(meta_csv, split="val", **ds_kwargs)
    test_ds = DeepfakeVideoDataset(meta_csv, split="test", **ds_kwargs)

    # Use train + val splits to fit calibration threshold without leaking test set
    calib_ds = ConcatDataset([train_ds, val_ds])
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    pretrained = cfg["model"]["pretrained_name"]

    variants = [
        ("baseline", "E1: Classical ViT (Centralized)", ViTDeepfakeDetector(pretrained_name=pretrained), Path("checkpoints/baseline_vit.pt")),
        ("federated", "E2/E3: Federated ViT + Blockchain", ViTDeepfakeDetector(pretrained_name=pretrained), Path("checkpoints/federated_global_vit.pt")),
        ("quantum", "E4: Hybrid Quantum ViT (4-Qubit VQC)", HybridQuantumDeepfakeDetector(pretrained_name=pretrained, num_qubits=4), Path("checkpoints/quantum_vit.pt")),
        ("spiking", "E5: Neuromorphic Spiking ViT (SNN)", SpikingDeepfakeDetector(pretrained_name=pretrained, num_steps=10), Path("checkpoints/spiking_vit.pt")),
    ]

    calibration_registry = {}
    ablation_rows = []

    for key, exp_name, model, ckpt_path in variants:
        if not ckpt_path.exists():
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 1. Fit optimal threshold on calibration set
        _, cal_labels, cal_probs = collect_predictions(model, calib_loader, criterion, device)
        calib_params = fit_threshold_and_scale(cal_labels, cal_probs)
        calibration_registry[key] = calib_params

        # 2. Evaluate on held-out test set using calibrated probabilities
        t0 = time.perf_counter()
        test_loss, test_labels, test_raw_probs = collect_predictions(model, test_loader, criterion, device)
        latency_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(test_ds))

        test_cal_probs = np.array([apply_calibration(p, calib_params) for p in test_raw_probs])
        test_preds = (test_cal_probs >= 0.5).astype(int)

        acc = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, zero_division=0)
        try:
            auc = roc_auc_score(test_labels, test_cal_probs)
        except ValueError:
            auc = 0.5

        print(f"[{key.upper()}] Optimal Threshold τ* = {calib_params['optimal_threshold']:.4f} | "
              f"Scale = {calib_params['scale']:.2f} | Calibrated Test Acc = {acc:.2f} | F1 = {f1:.2f} | AUC = {auc:.2f}")

        ablation_rows.append({
            "Experiment": exp_name,
            "Checkpoint": str(ckpt_path),
            "Trainable_Params": trainable_params,
            "Optimal_Threshold": calib_params["optimal_threshold"],
            "Test_Loss": round(test_loss, 4),
            "Test_Acc": round(acc, 4),
            "Test_F1": round(f1, 4),
            "Test_ROC_AUC": round(auc, 4),
            "Latency_ms_per_Video": round(latency_ms, 2)
        })

    # Save calibration parameters for FastAPI
    calib_file = Path("checkpoints/calibration_params.json")
    with open(calib_file, "w") as f:
        json.dump(calibration_registry, f, indent=2)
    print(f"\nSaved calibration parameters to: {calib_file}")

    # Save updated ablation table
    df = pd.DataFrame(ablation_rows)
    out_csv = Path("experiments/ablation_results.csv")
    df.to_csv(out_csv, index=False)
    print("\n--- Updated Calibrated Ablation Table ---")
    print(df.to_string(index=False))
    print("==========================================================================")

if __name__ == "__main__":
    main()