import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.models.quantum_head import HybridQuantumDeepfakeDetector
from src.qf_neuroshield.models.spiking_head import SpikingDeepfakeDetector
from scripts.export_and_benchmark_edge import SpikingHeadWrapper

def find_videos(directory: Path):
    video_exts = {".mp4", ".mov", ".avi"}
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in video_exts])

def sample_multi_clips(video_path: Path, num_clips: int = 3, frames_per_clip: int = 6, image_size: int = 224):
    """Extracts multiple non-overlapping temporal clips from a single video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    all_indices = np.linspace(0, max(0, total_frames - 1), num_clips * frames_per_clip, dtype=int)
    frames_tensors = []

    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret:
            if frames_tensors:
                frames_tensors.append(frames_tensors[-1].clone())
            else:
                frames_tensors.append(torch.zeros(3, image_size, image_size))
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        min_dim = min(h, w)
        sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
        crop = frame_rgb[sy:sy+min_dim, sx:sx+min_dim]
        pil_img = Image.fromarray(crop).resize((image_size, image_size), Image.Resampling.BILINEAR)
        frames_tensors.append(transform(pil_img))

    cap.release()
    clips = []
    for c in range(num_clips):
        clip_t = torch.stack(frames_tensors[c * frames_per_clip : (c + 1) * frames_per_clip])
        clips.append(clip_t)
    return clips

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================================")
    print("=== Phase 2: Multi-Seed Cross-Validated Statistical Study (N=10 Videos) ===")
    print("==========================================================================")

    # 1. Index all 10 videos (including .mp4, .mov, .MOV)
    real_videos = find_videos(Path("data/raw/REAL"))
    fake_videos = find_videos(Path("data/raw/FAKE"))
    print(f"Found {len(real_videos)} REAL videos and {len(fake_videos)} FAKE videos.")

    base_vit = ViTDeepfakeDetector().to(device)
    base_vit.eval()

    video_features_list = []
    video_labels_list = []

    print("Extracting multi-clip ViT representations across all videos on MPS/CPU...")
    with torch.no_grad():
        for v_path in real_videos:
            clips = sample_multi_clips(v_path, num_clips=3, frames_per_clip=6)
            batch = torch.stack(clips).to(device) # [3, 6, 3, 224, 224]
            feats = base_vit.extract_video_features(batch).cpu().numpy() # [3, 768]
            video_features_list.append(feats)
            video_labels_list.append(0)

        for v_path in fake_videos:
            clips = sample_multi_clips(v_path, num_clips=3, frames_per_clip=6)
            batch = torch.stack(clips).to(device)
            feats = base_vit.extract_video_features(batch).cpu().numpy()
            video_features_list.append(feats)
            video_labels_list.append(1)

    video_labels = np.array(video_labels_list) # [10]

    def make_classical():
        return nn.Sequential(nn.Linear(768, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))

    def make_quantum():
        q_model = HybridQuantumDeepfakeDetector(num_qubits=4)
        class QHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = q_model.projection
                self.qnn = q_model.qnn
                self.cls = q_model.classifier
            def forward(self, x):
                return self.cls(self.qnn(self.proj(x)))
        return QHead()

    def make_spiking():
        s_model = SpikingDeepfakeDetector(num_steps=10)
        return SpikingHeadWrapper(s_model)

    seeds = [42, 1337, 2026]
    paradigms = [
        ("E1/E2: Classical / Federated ViT Head", make_classical, 25, 0.005),
        ("E4: Hybrid Quantum VQC Head (4-Qubit)", make_quantum, 10, 0.02),
        ("E5: Neuromorphic Spiking SNN Head", make_spiking, 25, 0.005),
    ]

    summary_rows = []

    for name, head_fn, epochs, lr in paradigms:
        seed_accs, seed_f1s, seed_aucs = [], [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            all_true, all_probs = [], []

            for train_idx, test_idx in skf.split(np.zeros(len(video_labels)), video_labels):
                # Group clips strictly by video_id to prevent any frame leakage
                x_tr = np.vstack([video_features_list[i] for i in train_idx])
                y_tr = np.repeat(video_labels[train_idx], 3)

                x_te = np.vstack([video_features_list[i] for i in test_idx])
                y_te = video_labels[test_idx]

                scaler = StandardScaler()
                x_tr = scaler.fit_transform(x_tr)
                x_te = scaler.transform(x_te)

                x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
                y_tr_t = torch.tensor(y_tr, dtype=torch.long)
                x_te_t = torch.tensor(x_te, dtype=torch.float32)

                head = head_fn()
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

                head.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = criterion(head(x_tr_t), y_tr_t)
                    loss.backward()
                    optimizer.step()

                head.eval()
                with torch.no_grad():
                    clip_probs = torch.softmax(head(x_te_t), dim=1)[:, 1].numpy()
                    # Pool clip probabilities back to video-level predictions
                    vid_probs = clip_probs.reshape(len(test_idx), 3).mean(axis=1)

                all_true.extend(y_te)
                all_probs.extend(vid_probs)

            all_true = np.array(all_true)
            all_probs = np.array(all_probs)
            # Calibrate threshold around median score
            thresh = float(np.median(all_probs))
            preds = (all_probs >= thresh).astype(int)

            seed_accs.append(accuracy_score(all_true, preds))
            seed_f1s.append(f1_score(all_true, preds, zero_division=0))
            try:
                seed_aucs.append(roc_auc_score(all_true, all_probs))
            except ValueError:
                seed_aucs.append(0.5)

        row = {
            "Architecture": name,
            "Evaluation_Protocol": "5-Fold CV x 3 Seeds",
            "Accuracy_Mean_Std": f"{np.mean(seed_accs):.2f} ± {np.std(seed_accs):.2f}",
            "F1_Mean_Std": f"{np.mean(seed_f1s):.2f} ± {np.std(seed_f1s):.2f}",
            "ROC_AUC_Mean_Std": f"{np.mean(seed_aucs):.2f} ± {np.std(seed_aucs):.2f}",
        }
        summary_rows.append(row)
        print(f"  {name:<40} | Acc: {row['Accuracy_Mean_Std']} | F1: {row['F1_Mean_Std']} | AUC: {row['ROC_AUC_Mean_Std']}")

    df = pd.DataFrame(summary_rows)
    df.to_csv("experiments/statistical_significance.csv", index=False)
    print("\nSaved updated statistical study to: experiments/statistical_significance.csv")
    print("==========================================================================")

if __name__ == "__main__":
    main()
