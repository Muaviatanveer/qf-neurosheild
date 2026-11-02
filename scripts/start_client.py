# scripts/start_client.py
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import flwr as fl
import torch
import yaml

from src.qf_neuroshield.data.dataset import DeepfakeVideoDataset
from src.qf_neuroshield.federated.client import FlowerDeepfakeClient
from scripts.run_federation import partition_dataset

def main():
    parser = argparse.ArgumentParser(description="QF-NeuroShield Client Node")
    parser.add_argument("--client-id", type=int, required=True, help="0, 1, or 2")
    parser.add_argument("--org-name", type=str, required=True, help="Name of the organization")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    with open("configs/federated.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Client {args.client_id}: {args.org_name} on {device} ===")

    # Load and partition data (In reality, this client only has access to its own local folder)
    full_train_ds = DeepfakeVideoDataset(
        cfg["dataset"]["metadata_csv"], split="train",
        num_frames=cfg["dataset"]["num_frames"],
        image_size=cfg["dataset"]["image_size"],
        crop_face=cfg["dataset"]["crop_face"]
    )
    
    # Secure partitioning: ensure this client only gets its data subset
    subsets = partition_dataset(full_train_ds, num_clients=3)
    local_dataset = subsets[args.client_id]

    print(f"Loaded {len(local_dataset)} local videos. Connecting to {args.server_address}...")

    # Initialize standard client
    client = FlowerDeepfakeClient(
        client_id=args.client_id,
        client_name=args.org_name,
        dataset=local_dataset,
        model_name=cfg["model"]["pretrained_name"],
        device=device,
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["learning_rate"],
        local_epochs=cfg["federated"]["local_epochs"]
    )

    # Wrapper to inject Blockchain ID into metrics
    class DistributedClientWrapper(fl.client.NumPyClient):
        def get_parameters(self, config):
            return client.get_parameters(config)
        def set_parameters(self, parameters):
            client.set_parameters(parameters)
        def fit(self, parameters, config):
            updated_params, num_examples, metrics = client.fit(parameters, config)
            # Inject identity for the Blockchain Smart Contract
            metrics["client_id"] = f"0xClient_{args.client_id}"
            metrics["org_name"] = args.org_name
            return updated_params, num_examples, metrics
        def evaluate(self, parameters, config):
            return 0.0, len(local_dataset), {"accuracy": 0.0}

    # Start network connection
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=DistributedClientWrapper()
    )

if __name__ == "__main__":
    main()