# scripts/run_federation.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import copy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.qf_neuroshield.data.dataset import DeepfakeVideoDataset
from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.federated.client import FlowerDeepfakeClient
from src.qf_neuroshield.federated.aggregation import fedavg_aggregate, compute_weights_hash

def evaluate_global_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for batch in test_loader:
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

    return total_loss / max(1, len(test_loader)), acc, f1, auc

def partition_dataset(full_dataset, num_clients: int):
    """Partitions the training dataset indices among clients."""
    total_samples = len(full_dataset)
    indices = list(range(total_samples))
    split_size = total_samples // num_clients
    
    subsets = []
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_clients - 1 else total_samples
        client_indices = indices[start_idx:end_idx]
        subsets.append(Subset(full_dataset, client_indices))
    return subsets

def main():
    with open("configs/federated.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting QF-NeuroShield Federated Orchestration ===")
    print(f"Compute Device: {device}")

    # 1. Load full datasets
    full_train_ds = DeepfakeVideoDataset(
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
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    criterion = nn.CrossEntropyLoss()

    # 2. Partition dataset for the 3 organizations
    num_clients = cfg["federated"]["num_clients"]
    client_subsets = partition_dataset(full_train_ds, num_clients)
    
    clients = []
    for i in range(num_clients):
        name = cfg["federated"]["client_names"][i]
        c = FlowerDeepfakeClient(
            client_id=i,
            client_name=name,
            dataset=client_subsets[i],
            model_name=cfg["model"]["pretrained_name"],
            device=device,
            batch_size=cfg["training"]["batch_size"],
            lr=cfg["training"]["learning_rate"],
            local_epochs=cfg["federated"]["local_epochs"]
        )
        clients.append(c)
        print(f"Registered Participant [{i}]: {name} with {len(client_subsets[i])} local training videos.")

    # 3. Initialize Global Model
    global_model = ViTDeepfakeDetector(
        pretrained_name=cfg["model"]["pretrained_name"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)

    # Initial parameter vector W_0
    global_parameters = [val.cpu().numpy() for _, val in global_model.classifier.state_dict().items()]
    initial_hash = compute_weights_hash(global_parameters)
    print(f"\nInitial Global Model Hash (W_0): {initial_hash}")

    # 4. Federated Training Rounds
    num_rounds = cfg["federated"]["num_rounds"]
    for r in range(1, num_rounds + 1):
        print(f"\n-------------------------------------------------------------")
        print(f">>> Global Training Round {r}/{num_rounds}")
        print(f"-------------------------------------------------------------")

        round_results = []
        for client in clients:
            updated_params, num_samples, metrics = client.fit(global_parameters, config={})
            print(f"  [{metrics['client_name']}] Trained {num_samples} samples | "
                  f"Loss: {metrics['train_loss']:.4f} | Update Hash: {metrics['update_hash'][:14]}...")
            round_results.append((updated_params, num_samples))

        # Federated Averaging Aggregation
        global_parameters = fedavg_aggregate(round_results)
        round_hash = compute_weights_hash(global_parameters)

        # Update global model with aggregated weights
        params_dict = zip(global_model.classifier.state_dict().keys(), global_parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        global_model.classifier.load_state_dict(state_dict)

        # Global Evaluation on held-out test set
        test_loss, test_acc, test_f1, test_auc = evaluate_global_model(
            global_model, test_loader, criterion, device
        )

        print(f"\n[Round {r} Aggregation Complete]")
        print(f"  New Global Model Hash (W_{r}): {round_hash}")
        print(f"  Held-out Global Test Evaluation -> Loss: {test_loss:.4f} | Acc: {test_acc:.2f} | F1: {test_f1:.2f} | AUC: {test_auc:.2f}")

    # 5. Save final federated model
    save_path = Path(cfg["training"]["save_dir"]) / "federated_global_vit.pt"
    torch.save(global_model.state_dict(), save_path)
    print(f"\n=============================================================")
    print(f"Federated Learning Finished!")
    print(f"Saved Global Model to: {save_path}")
    print(f"Milestone 2 Complete: 3 clients trained collaboratively without sharing raw data.")
    print(f"=============================================================")

if __name__ == "__main__":
    main()