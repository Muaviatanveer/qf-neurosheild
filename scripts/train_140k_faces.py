import io
import sys
import zipfile
import random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector

class ZipStreamFaceDataset(Dataset):
    """Streams real and fake images directly from inside archive (9).zip without disk extraction!"""
    def __init__(self, zip_path: Path, entries, transform):
        self.zip_path = zip_path
        self.entries = entries  # List of (zip_filename, label)
        self.transform = transform
        self._zf = None

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
        # Shape [1, 3, 224, 224] for single-frame ViT compatibility
        tensor = self.transform(img).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)

def find_140k_zip() -> Path:
    candidates = [
        Path("~/Downloads/archive (9).zip").expanduser(),
        Path("~/Downloads/140k-real-and-fake-faces.zip").expanduser(),
        Path("140k-real-and-fake-faces.zip"),
        Path("archive (9).zip"),
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in sorted(Path("~/Downloads").expanduser().glob("archive*.zip"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.stat().st_size > 500_000_000:
            return p
    raise FileNotFoundError("Could not find the 140k zip file in ~/Downloads/")

def main():
    zip_path = find_140k_zip()
    print("==========================================================================")
    print(f"🚀 Streaming 140k Real & Fake Faces Directly From: {zip_path.name}")
    print("==========================================================================")

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

    train_real = [n for n in all_names if "/train/real/" in n and n.endswith((".jpg", ".png"))]
    train_fake = [n for n in all_names if "/train/fake/" in n and n.endswith((".jpg", ".png"))]
    test_real  = [n for n in all_names if "/test/real/" in n and n.endswith((".jpg", ".png"))]
    test_fake  = [n for n in all_names if "/test/fake/" in n and n.endswith((".jpg", ".png"))]

    print(f"Archive Index: {len(train_real):,} Train Real | {len(train_fake):,} Train Fake | {len(test_real)+len(test_fake):,} Test")

    random.seed(42)
    # Balanced high-diversity sample (800 Real FFHQ + 800 Fake StyleGAN)
    tr_entries = [(n, 0) for n in random.sample(train_real, min(800, len(train_real)))] + \
                 [(n, 1) for n in random.sample(train_fake, min(800, len(train_fake)))]
    te_entries = [(n, 0) for n in random.sample(test_real, min(200, len(test_real)))] + \
                 [(n, 1) for n in random.sample(test_fake, min(200, len(test_fake)))]
    random.shuffle(tr_entries)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_loader = DataLoader(ZipStreamFaceDataset(zip_path, tr_entries, transform), batch_size=16, shuffle=True)
    test_loader  = DataLoader(ZipStreamFaceDataset(zip_path, te_entries, transform), batch_size=16, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")
    model = ViTDeepfakeDetector().to(device)

    # Universal Unfreezing: Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 16 parameter tensors of the backbone (top attention blocks)
    backbone_params = list(model.backbone.parameters())
    for p in backbone_params[-16:]:
        p.requires_grad = True

    # Unfreeze the classification head
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters (Top ViT Blocks + Head): {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.01)

    epochs = 3
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_prob, y_pred = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                y_true.extend(y.numpy())
                y_prob.extend(probs)
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        print(f"  Epoch [{ep}/{epochs}] Train Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc*100:.1f}% | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")

    save_path = Path("checkpoints/image_140k_vit.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Saved 140k-Trained Deepfake Image ViT to: {save_path}")
    print("VERITAS will now automatically use this model checkpoint for all image detections!")

if __name__ == "__main__":
    main()
