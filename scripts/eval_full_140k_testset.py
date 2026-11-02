import io
import sys
import json
import time
import zipfile
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.qf_neuroshield.models.sw_cross_vit import SWCrossViTDetector
from src.qf_neuroshield.training.calibration import apply_calibration
from scripts.train_140k_faces import find_140k_zip

class FullTestZipDataset(Dataset):
    def __init__(self, zip_path: Path, entries):
        self.zip_path = zip_path
        self.entries = entries
        self._zf = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _get_zf(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
        return self._zf

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fname, label = self.entries[idx]
        raw_bytes = self._get_zf().read(fname)
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        tensor = self.transform(img) # [3, 224, 224]
        return tensor, torch.tensor(label, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SW-CrossViT on the Entire 20,000 Test Set")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for MPS GPU inference")
    parser.add_argument("--eval-all-140k", action="store_true", help="Evaluate all 140,000 images across train+val+test")
    args = parser.parse_args()

    zip_path = find_140k_zip()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("checkpoints/sw_cross_vit_140k.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")

    calib_file = Path("checkpoints/calibration_params.json")
    calib = json.loads(calib_file.read_text()).get("sw_cross_vit", {"optimal_threshold": 0.1462, "scale": 4.0}) if calib_file.exists() else {"optimal_threshold": 0.1462, "scale": 4.0}

    print("========================================================================================")
    print("🔬 VERITAS — FULL-SCALE DATASET EVALUATION")
    print(f"   Model       : SW-CrossViT (Spatial-Wavelet Cross-Attention)")
    print(f"   Checkpoint  : {ckpt_path.name}")
    print(f"   Device      : {device}")
    print(f"   Threshold τ*: {calib['optimal_threshold']}")
    print("========================================================================================")

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

    if args.eval_all_140k:
        print("⚡ Mode: EVALUATING ENTIRE ARCHIVE (140,000 IMAGES)...")
        real_names = [n for n in all_names if "/real/" in n and n.endswith((".jpg", ".png"))]
        fake_names = [n for n in all_names if "/fake/" in n and n.endswith((".jpg", ".png"))]
    else:
        print("🎯 Mode: EVALUATING FULL OFFICIAL HELD-OUT TEST SPLIT (20,000 IMAGES)...")
        real_names = [n for n in all_names if "/test/real/" in n and n.endswith((".jpg", ".png"))]
        fake_names = [n for n in all_names if "/test/fake/" in n and n.endswith((".jpg", ".png"))]

    test_entries = [(n, 0) for n in real_names] + [(n, 1) for n in fake_names]
    print(f"Loaded Total Evaluation Split: {len(test_entries):,} images ({len(real_names):,} Real + {len(fake_names):,} Fake)")

    test_loader = DataLoader(
        FullTestZipDataset(zip_path, test_entries),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = SWCrossViTDetector(unfreeze_top_layers=2).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true = []
    y_raw_probs = []
    y_cal_probs = []

    print("\nExecuting Batched Inference Pipeline on Apple Silicon MPS...")
    t_start = time.perf_counter()

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Benchmarking", ncols=100):
            # Shape [B, 1, 3, 224, 224]
            x_batch = x_batch.unsqueeze(1).to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            y_true.extend(y_batch.numpy())
            y_raw_probs.extend(probs)
            for p in probs:
                y_cal_probs.append(apply_calibration(float(p), calib))

    total_time = time.perf_counter() - t_start
    y_true = np.array(y_true)
    y_raw_probs = np.array(y_raw_probs)
    y_cal_probs = np.array(y_cal_probs)

    y_preds = (y_cal_probs >= 0.50).astype(int)

    acc = accuracy_score(y_true, y_preds) * 100.0
    prec = precision_score(y_true, y_preds) * 100.0
    rec = recall_score(y_true, y_preds) * 100.0
    f1 = f1_score(y_true, y_preds)
    auc = roc_auc_score(y_true, y_raw_probs)
    cm = confusion_matrix(y_true, y_preds)

    fps = len(y_true) / total_time
    ms_per_img = (total_time * 1000.0) / len(y_true)

    print("\n" + "=" * 88)
    print(f"📊 FINAL FULL DATASET BENCHMARK RESULTS ({len(y_true):,} IMAGES)")
    print("=" * 88)
    print(f"Total Evaluated       : {len(y_true):,} images ({len(real_names):,} Real + {len(fake_names):,} Fake)")
    print(f"Execution Time        : {total_time:.1f} seconds ({fps:.1f} images/sec | {ms_per_img:.2f} ms/image)")
    print(f"Optimal Threshold (τ*): {calib['optimal_threshold']}")
    print("-" * 88)
    print(f"  • Overall Accuracy  : {acc:.2f}% ({cm[0,0]+cm[1,1]:,} / {len(y_true):,} correct)")
    print(f"  • Precision (PPV)   : {prec:.2f}%")
    print(f"  • Recall (TPR)      : {rec:.2f}% (Fakes correctly detected)")
    print(f"  • F1 Score          : {f1:.4f}")
    print(f"  • ROC-AUC Score     : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"                       Predicted REAL      Predicted FAKE")
    print(f"  Actual REAL ({len(real_names):,}):   {cm[0, 0]:>18,}    {cm[0, 1]:>18,}")
    print(f"  Actual FAKE ({len(fake_names):,}):   {cm[1, 0]:>18,}    {cm[1, 1]:>18,}")
    print("=" * 88)

    # Save full-scale results
    out_file = Path("experiments/full_20k_test_benchmark.json")
    out_data = {
        "dataset": "140k Real and Fake Faces (Full Test Split)",
        "total_images_evaluated": int(len(y_true)),
        "real_count": int(len(real_names)),
        "fake_count": int(len(fake_names)),
        "execution_seconds": round(total_time, 2),
        "throughput_fps": round(fps, 1),
        "ms_per_image": round(ms_per_img, 2),
        "accuracy_pct": round(acc, 2),
        "precision_pct": round(prec, 2),
        "recall_pct": round(rec, 2),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "optimal_threshold": calib["optimal_threshold"],
        "confusion_matrix": cm.tolist()
    }
    out_file.write_text(json.dumps(out_data, indent=2))
    print(f"Saved full evaluation record to: {out_file}")

if __name__ == "__main__":
    main()
