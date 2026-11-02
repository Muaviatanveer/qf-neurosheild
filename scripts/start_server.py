# scripts/start_server.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import flwr as fl
import torch
import numpy as np

from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.federated.aggregation import compute_weights_hash
from src.qf_neuroshield.blockchain.audit import PermissionedAuditLedger
from src.qf_neuroshield.federated.blockchain_strategy import BlockchainFedAvg

def main():
    print("=======================================================")
    print("🚀 Starting QF-NeuroShield Federated Server (Port 8080)")
    print("=======================================================")

    # 1. Initialize W_0
    model = ViTDeepfakeDetector(freeze_backbone=True)
    initial_params = [val.cpu().numpy() for _, val in model.classifier.state_dict().items()]
    initial_hash = compute_weights_hash(initial_params)
    
    # 2. Initialize Blockchain Ledger
    ledger = PermissionedAuditLedger(initial_global_hash=initial_hash, max_update_l2_norm=5.0)
    
    # Pre-register our expected clients (In Web3, they would register via smart contract themselves)
    ledger.register_participant("0xClient_0", "University_A")
    ledger.register_participant("0xClient_1", "University_B")
    ledger.register_participant("0xClient_2", "Verification_Lab")

    # 3. Configure Strategy
    strategy = BlockchainFedAvg(
        ledger=ledger,
        base_params=initial_params,
        fraction_fit=1.0,  # Require 100% of available clients to train
        min_fit_clients=3, # Wait until 3 clients connect
        min_available_clients=3,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params)
    )

    # 4. Start gRPC Network Server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
    
    # 5. Save Ledger after training
    ledger.save_ledger("experiments/federated/distributed_blockchain_ledger.json")
    print("\n✅ Training complete. Distributed Ledger saved.")

if __name__ == "__main__":
    main()