# scripts/run_ablation_benchmark.py
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.qf_neuroshield.data.dataset import DeepfakeVideoDataset
from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.models.quantum_head import HybridQuantumDeepfakeDetector
from src.qf_neuroshield.models.spiking_head import SpikingDeepfakeDetector
from scripts.train_baseline import evaluate

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(name, model, ckpt_path, test_loader, criterion, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    params = count_trainable_params(model)
    num_videos = len(test_loader.dataset)

    t0 = time.perf_counter()
    loss, acc, f1, auc = evaluate(model, test_loader, criterion, device)
    elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, num_videos)

    return {
        "Experiment": name,
        "Checkpoint": str(ckpt_path),
        "Trainable_Params": params,
        "Test_Loss": round(loss, 4),
        "Test_Acc": round(acc, 4),
        "Test_F1": round(f1, 4),
        "Test_ROC_AUC": round(auc, 4),
        "Latency_ms_per_Video": round(elapsed_ms, 2)
    }

def main():
    with open("configs/baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================================")
    print("=== QF-NeuroShield Controlled Ablation Study Benchmark (E1 - E5) ===")
    print("==========================================================================")

    test_ds = DeepfakeVideoDataset(
        cfg["dataset"]["metadata_csv"], split="test",
        num_frames=cfg["dataset"]["num_frames"],
        image_size=cfg["dataset"]["image_size"],
        crop_face=cfg["dataset"]["crop_face"]
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    pretrained = cfg["model"]["pretrained_name"]

    models_to_test = [
        ("E1: Classical ViT (Centralized)", ViTDeepfakeDetector(pretrained_name=pretrained), Path("checkpoints/baseline_vit.pt")),
        ("E2/E3: Federated ViT + Blockchain", ViTDeepfakeDetector(pretrained_name=pretrained), Path("checkpoints/federated_global_vit.pt")),
        ("E4: Hybrid Quantum ViT (4-Qubit VQC)", HybridQuantumDeepfakeDetector(pretrained_name=pretrained, num_qubits=4), Path("checkpoints/quantum_vit.pt")),
        ("E5: Neuromorphic Spiking ViT (SNN)", SpikingDeepfakeDetector(pretrained_name=pretrained, num_steps=10), Path("checkpoints/spiking_vit.pt")),
    ]

    results = []
    for name, model_instance, ckpt in models_to_test:
        if ckpt.exists():
            print(f"Benchmarking {name}...")
            res = benchmark_model(name, model_instance, ckpt, test_loader, criterion, device)
            results.append(res)
        else:
            print(f"Skipping {name} (Checkpoint {ckpt} not found)")

    df = pd.DataFrame(results)
    out_csv = Path("experiments/ablation_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n--- Final Comparative Ablation Table ---")
    print(df.to_string(index=False))
    print(f"\nSaved benchmark results to: {out_csv}")
    print("==========================================================================")

if __name__ == "__main__":
    main()