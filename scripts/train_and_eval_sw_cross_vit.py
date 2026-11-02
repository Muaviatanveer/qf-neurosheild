import io
import sys
import json
import time
import zipfile
import random
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.qf_neuroshield.models.sw_cross_vit import SWCrossViTDetector
from src.qf_neuroshield.training.calibration import fit_threshold_and_scale, apply_calibration
from scripts.train_140k_faces import find_140k_zip

class CompressionInvariantZipDataset(Dataset):
    """
    Streams images from archive (9).zip with optional JPEG/Screenshot compression
    augmentation during training so the model is invariant to WhatsApp/social media artifacts.
    """
    def __init__(self, zip_path: Path, entries, is_train: bool = False):
        self.zip_path = zip_path
        self.entries = entries
        self.is_train = is_train
        self._zf = None

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
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
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # Simulate WhatsApp / Screenshot JPEG compression on 30% of training images
        if self.is_train and random.random() < 0.30:
            buf = io.BytesIO()
            q = random.randint(60, 92)
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        tensor = self.transform(img).unsqueeze(0)  # [1, 3, 224, 224]
        return tensor, torch.tensor(label, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Train & Benchmark SW-CrossViT on 140k Real & Fake Faces")
    parser.add_argument("--train-samples", type=int, default=6000,
                        help="Number of balanced training images to stream (default: 6000; set 100000 for full 100k)")
    parser.add_argument("--test-samples", type=int, default=4000,
                        help="Number of held-out test images to evaluate (default: 4000; set 20000 for full 20k test split)")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for Apple MPS")
    args = parser.parse_args()

    zip_path = find_140k_zip()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print("========================================================================================")
    print("🚀 VERITAS — SW-CrossViT (Spatial-Wavelet Cross-Attention) 140k Benchmark")
    print(f"   Archive: {zip_path.name} | Compute Device: {device}")
    print("========================================================================================")

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

    train_real = [n for n in all_names if "/train/real/" in n and n.endswith((".jpg", ".png"))]
    train_fake = [n for n in all_names if "/train/fake/" in n and n.endswith((".jpg", ".png"))]
    val_real   = [n for n in all_names if "/valid/real/" in n and n.endswith((".jpg", ".png"))]
    val_fake   = [n for n in all_names if "/valid/fake/" in n and n.endswith((".jpg", ".png"))]
    test_real  = [n for n in all_names if "/test/real/" in n and n.endswith((".jpg", ".png"))]
    test_fake  = [n for n in all_names if "/test/fake/" in n and n.endswith((".jpg", ".png"))]

    total_indexed = len(train_real) + len(train_fake) + len(val_real) + len(val_fake) + len(test_real) + len(test_fake)
    print(f"📦 Indexed Total Archive: {total_indexed:,} Faces")
    print(f"   • Train Split : {len(train_real):,} Real FFHQ + {len(train_fake):,} Fake StyleGAN = {len(train_real)+len(train_fake):,}")
    print(f"   • Valid Split : {len(val_real):,} Real FFHQ + {len(val_fake):,} Fake StyleGAN = {len(val_real)+len(val_fake):,}")
    print(f"   • Test Split  : {len(test_real):,} Real FFHQ + {len(test_fake):,} Fake StyleGAN = {len(test_real)+len(test_fake):,}")

    random.seed(42)
    n_tr_per_cls = min(args.train_samples // 2, len(train_real))
    n_te_per_cls = min(args.test_samples // 2, len(test_real))

    tr_entries = [(n, 0) for n in random.sample(train_real, n_tr_per_cls)] + \
                 [(n, 1) for n in random.sample(train_fake, n_tr_per_cls)]
    val_entries = [(n, 0) for n in random.sample(val_real, min(400, len(val_real)))] + \
                  [(n, 1) for n in random.sample(val_fake, min(400, len(val_fake)))]
    te_entries = [(n, 0) for n in random.sample(test_real, n_te_per_cls)] + \
                 [(n, 1) for n in random.sample(test_fake, n_te_per_cls)]

    random.shuffle(tr_entries)

    train_loader = DataLoader(CompressionInvariantZipDataset(zip_path, tr_entries, is_train=True),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(CompressionInvariantZipDataset(zip_path, val_entries, is_train=False),
                              batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(CompressionInvariantZipDataset(zip_path, te_entries, is_train=False),
                              batch_size=args.batch_size, shuffle=False)

    print(f"\n🧠 Initializing SW-CrossViT (2-Level Haar DWT + 8-Head Cross-Attention + Discrepancy Gate)...")
    model = SWCrossViTDetector(unfreeze_top_layers=2).to(device)

    # Warm-start backbone from existing 140k checkpoint if available
    prev_ckpt = Path("checkpoints/image_140k_vit.pt")
    if prev_ckpt.exists():
        state = torch.load(prev_ckpt, map_location=device)
        backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
        model.backbone.load_state_dict(backbone_state, strict=False)
        print("   ✓ Warm-started ViT-Base backbone weights from checkpoints/image_140k_vit.pt")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Trainable Parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1.8e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-5)

    best_val_auc = 0.0
    save_path = Path("checkpoints/sw_cross_vit_140k.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{ep}/{args.epochs}] Training", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{running_loss/(total/args.batch_size):.4f}", acc=f"{correct/total*100:.1f}%")

        # Validation Check
        model.eval()
        v_true, v_prob = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                v_true.extend(y.numpy())
                v_prob.extend(probs)

        val_auc = roc_auc_score(v_true, v_prob)
        val_acc = accuracy_score(v_true, (np.array(v_prob) >= 0.5).astype(int)) * 100.0
        print(f"   → Validation Check | Val Acc: {val_acc:.2f}% | Val ROC-AUC: {val_auc:.4f}")

        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)

    # Fit Youden's J Optimal Calibration on Validation Set
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    v_true, v_prob = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            v_true.extend(y.numpy())
            v_prob.extend(probs)
    calib_params = fit_threshold_and_scale(np.array(v_true), np.array(v_prob))

    # Full-Scale Held-Out Test Evaluation
    print(f"\n🔬 Evaluating Best SW-CrossViT Checkpoint on {len(te_entries):,} Held-Out Test Images...")
    t_eval_start = time.perf_counter()
    te_true, te_raw_prob, te_cal_prob = [], [], []

    for x, y in tqdm(test_loader, desc="Testing Held-Out Split", ncols=100):
        with torch.no_grad():
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            te_true.extend(y.numpy())
            te_raw_prob.extend(probs)
            te_cal_prob.extend([apply_calibration(p, calib_params) for p in probs])

    eval_time = time.perf_counter() - t_eval_start
    te_true = np.array(te_true)
    te_raw_prob = np.array(te_raw_prob)
    te_cal_prob = np.array(te_cal_prob)
    te_preds = (te_cal_prob >= 0.5).astype(int)

    acc = accuracy_score(te_true, te_preds) * 100.0
    prec = precision_score(te_true, te_preds) * 100.0
    rec = recall_score(te_true, te_preds) * 100.0
    f1 = f1_score(te_true, te_preds)
    auc = roc_auc_score(te_true, te_raw_prob)
    cm = confusion_matrix(te_true, te_preds)

    print("\n" + "=" * 88)
    print("�� SW-CrossViT (Spatial-Wavelet Cross-Attention) 140k BENCHMARK REPORT")
    print("=" * 88)
    print(f"Total Archive Size      : {total_indexed:,} images")
    print(f"Training Set Streamed   : {len(tr_entries):,} images ({n_tr_per_cls:,} Real + {n_tr_per_cls:,} Fake)")
    print(f"Held-Out Test Evaluated : {len(te_entries):,} images ({n_te_per_cls:,} Real + {n_te_per_cls:,} Fake)")
    print(f"Optimal Threshold (τ*)  : {calib_params['optimal_threshold']:.4f}")
    print("-" * 88)
    print(f"  • Test Accuracy       : {acc:.2f}%")
    print(f"  • Test Precision      : {prec:.2f}%")
    print(f"  • Test Recall (TPR)   : {rec:.2f}%")
    print(f"  • Test F1 Score       : {f1:.4f}")
    print(f"  • Test ROC-AUC        : {auc:.4f}")
    print(f"  • Throughput Speed    : {len(te_entries)/eval_time:.1f} images/sec ({eval_time*1000/len(te_entries):.2f} ms/image)")
    print("\nConfusion Matrix:")
    print(f"                 Predicted REAL    Predicted FAKE")
    print(f"  Actual REAL:   {cm[0, 0]:>14,}    {cm[0, 1]:>14,}")
    print(f"  Actual FAKE:   {cm[1, 0]:>14,}    {cm[1, 1]:>14,}")
    print("=" * 88)

    # Save calibration & benchmark report
    calib_file = Path("checkpoints/calibration_params.json")
    calib_data = json.loads(calib_file.read_text()) if calib_file.exists() else {}
    calib_data["sw_cross_vit"] = calib_params
    calib_file.write_text(json.dumps(calib_data, indent=2))

    report = {
        "architecture": "SW-CrossViT (2-Level Haar DWT + 8-Head Spatial-Wavelet Cross-Attention)",
        "total_archive_indexed": total_indexed,
        "train_samples": len(tr_entries),
        "test_samples": len(te_entries),
        "accuracy_pct": round(acc, 2),
        "precision_pct": round(prec, 2),
        "recall_pct": round(rec, 2),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "optimal_threshold_tau": calib_params["optimal_threshold"],
        "confusion_matrix": cm.tolist(),
        "ms_per_image": round(eval_time * 1000 / len(te_entries), 2)
    }
    Path("experiments/sw_cross_vit_140k_benchmark.json").write_text(json.dumps(report, indent=2))
    print(f"\n✅ Saved SW-CrossViT checkpoint to: {save_path}")
    print("✅ Saved benchmark report to: experiments/sw_cross_vit_140k_benchmark.json")

if __name__ == "__main__":
    main()
