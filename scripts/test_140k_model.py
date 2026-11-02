import io
import sys
import json
import zipfile
import random
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from src.qf_neuroshield.models.sw_cross_vit import SWCrossViTDetector
from src.qf_neuroshield.training.calibration import apply_calibration
from scripts.train_140k_faces import find_140k_zip

def main():
    parser = argparse.ArgumentParser(description="Exact Verification Test for SW-CrossViT")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per class")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("checkpoints/sw_cross_vit_140k.pt")
    calib_path = Path("checkpoints/calibration_params.json")
    calib = json.loads(calib_path.read_text()).get("sw_cross_vit", {"optimal_threshold": 0.1462, "scale": 4.0}) if calib_path.exists() else {"optimal_threshold": 0.1462, "scale": 4.0}

    print("========================================================================================")
    print("🔬 VERITAS — SW-CrossViT Exact Programmatic Verification")
    print(f"   Device: {device} | Checkpoint: {ckpt_path} | Threshold τ* = {calib['optimal_threshold']}")
    print("========================================================================================")

    model = SWCrossViTDetector(unfreeze_top_layers=2).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    zip_path = find_140k_zip()
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        test_real = [n for n in all_names if "/test/real/" in n and n.endswith((".jpg", ".png"))]
        test_fake = [n for n in all_names if "/test/fake/" in n and n.endswith((".jpg", ".png"))]

    random.seed(42)
    samples = [(n, 0, "REAL") for n in random.sample(test_real, args.num_samples)] + \
              [(n, 1, "FAKE") for n in random.sample(test_fake, args.num_samples)]
    random.shuffle(samples)

    y_true, y_pred, y_prob = [], [], []
    print(f"{'#':<3} | {'True Label':<10} | {'Predicted':<10} | {'Raw Fake %':<11} | {'Calibrated %':<12} | {'Status':<8} | {'Filename'}")
    print("-" * 92)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for idx, (fname, label_int, label_str) in enumerate(samples, 1):
            img = Image.open(io.BytesIO(zf.read(fname))).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            tensor = transform(img).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)[0]
                raw_fake = float(torch.softmax(logits, dim=0)[1].item())

            cal_fake = apply_calibration(raw_fake, calib)
            pred_str = "FAKE" if cal_fake >= 0.5 else "REAL"
            pred_int = 1 if cal_fake >= 0.5 else 0
            status = "✓ PASS" if pred_str == label_str else "✗ FAIL"

            y_true.append(label_int); y_pred.append(pred_int); y_prob.append(raw_fake)
            print(f"{idx:<3} | {label_str:<10} | {pred_str:<10} | {raw_fake*100:>9.2f}%  | {cal_fake*100:>10.2f}%  | {status:<8} | {Path(fname).name}")

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print("-" * 92)
    print(f"Exact Sample Accuracy: {acc:.1f}% | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print("=" * 92)

if __name__ == "__main__":
    main()
