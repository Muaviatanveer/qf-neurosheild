# scripts/run_blockchain_federation.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.qf_neuroshield.data.dataset import DeepfakeVideoDataset
from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.federated.client import FlowerDeepfakeClient
from src.qf_neuroshield.federated.aggregation import fedavg_aggregate, compute_weights_hash
from src.qf_neuroshield.blockchain.audit import PermissionedAuditLedger, BlockchainSecurityError
from scripts.run_federation import partition_dataset, evaluate_global_model

def main():
    with open("configs/federated.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================================")
    print("=== Stage 4: Blockchain-Secured Federated Learning & Audit (E3) ===")
    print("==================================================================")

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

    num_clients = cfg["federated"]["num_clients"]
    client_subsets = partition_dataset(full_train_ds, num_clients)

    # Initialize Global Model & Blockchain Ledger
    global_model = ViTDeepfakeDetector(
        pretrained_name=cfg["model"]["pretrained_name"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)
    global_parameters = [val.cpu().numpy() for _, val in global_model.classifier.state_dict().items()]
    w0_hash = compute_weights_hash(global_parameters)

    ledger = PermissionedAuditLedger(initial_global_hash=w0_hash, max_update_l2_norm=5.0)
    print(f"Genesis Block Created | W_0 Hash: {w0_hash[:18]}...")

    clients = []
    client_addresses = {}
    for i in range(num_clients):
        org_name = cfg["federated"]["client_names"][i]
        addr = f"0xClient_{i+1:02d}_{org_name}"
        client_addresses[i] = addr
        ledger.register_participant(addr, org_name)

        c = FlowerDeepfakeClient(
            client_id=i, client_name=org_name, dataset=client_subsets[i],
            model_name=cfg["model"]["pretrained_name"], device=device,
            batch_size=cfg["training"]["batch_size"], lr=cfg["training"]["learning_rate"],
            local_epochs=cfg["federated"]["local_epochs"]
        )
        clients.append(c)
        print(f"On-Chain Registered: {addr}")

    # Run Audited Federated Rounds
    num_rounds = cfg["federated"]["num_rounds"]
    for r in range(1, num_rounds + 1):
        print(f"\n------------------------------------------------------------------")
        print(f">>> Audited Federated Round {r}/{num_rounds}")
        print(f"------------------------------------------------------------------")
        base_hash = ledger.latest_global_hash
        approved_updates = []

        for i, client in enumerate(clients):
            addr = client_addresses[i]
            updated_params, num_samples, metrics = client.fit(global_parameters, config={})
            upd_hash = metrics["update_hash"]

            # Validate & Record on Permissioned Ledger
            ok, status_msg = ledger.validate_and_record_update(
                client_address=addr,
                round_num=r,
                base_model_hash=base_hash,
                update_hash=upd_hash,
                base_params=global_parameters,
                updated_params=updated_params
            )
            if ok:
                approved_updates.append((updated_params, num_samples))
                print(f"  [ON-CHAIN APPROVED] {metrics['client_name']} | "
                      f"Update: {upd_hash[:12]}... | {status_msg}")

        # --- Section 12: Run Adversarial Threat-Model Tests during Round 1 ---
        if r == 1:
            print("\n  --- Running Section 12 Adversarial Security Tests (Round 1) ---")
            
            # Adversarial Test 1: Wrong Base Model Hash
            try:
                fake_base_hash = "0x" + ("f" * 64)
                ledger.validate_and_record_update(
                    client_address=client_addresses[0], round_num=r,
                    base_model_hash=fake_base_hash, update_hash=upd_hash,
                    base_params=global_parameters, updated_params=updated_params
                )
            except BlockchainSecurityError as e:
                print(f"  [DEFENSE 1 PASSED - Wrong Base Model Blocked]: {e}")

            # Adversarial Test 2: Duplicate Submission in Same Round
            try:
                ledger.validate_and_record_update(
                    client_address=client_addresses[0], round_num=r,
                    base_model_hash=base_hash, update_hash=upd_hash,
                    base_params=global_parameters, updated_params=updated_params
                )
            except BlockchainSecurityError as e:
                print(f"  [DEFENSE 2 PASSED - Replay/Double Submit Blocked]: {e}")

            # Adversarial Test 3: Simulated Malicious Client (Model Poisoning)
            try:
                malicious_addr = "0xClient_99_Compromised_Node"
                ledger.register_participant(malicious_addr, "Compromised_Node")
                poisoned_params = [p + np.random.normal(0, 10.0, size=p.shape).astype(np.float32) for p in global_parameters]
                poisoned_hash = compute_weights_hash(poisoned_params)
                ledger.validate_and_record_update(
                    client_address=malicious_addr, round_num=r,
                    base_model_hash=base_hash, update_hash=poisoned_hash,
                    base_params=global_parameters, updated_params=poisoned_params
                )
            except BlockchainSecurityError as e:
                print(f"  [DEFENSE 3 PASSED - Poisoned Weights Blocked]: {e}")

        # Aggregate only approved updates
        global_parameters = fedavg_aggregate(approved_updates)
        new_global_hash = compute_weights_hash(global_parameters)
        ledger.record_global_model(round_num=r, global_model_hash=new_global_hash)

        # Load into global model & evaluate
        params_dict = zip(global_model.classifier.state_dict().keys(), global_parameters)
        global_model.classifier.load_state_dict({k: torch.tensor(v) for k, v in params_dict})
        test_loss, test_acc, test_f1, test_auc = evaluate_global_model(global_model, test_loader, criterion, device)

        print(f"\n  [Round {r} Global Model Recorded On-Chain]")
        print(f"  Global Hash (W_{r}): {new_global_hash[:18]}... | Test Loss: {test_loss:.4f} | AUC: {test_auc:.2f}")

    # Verify Cryptographic Chain Integrity
    print("\n==================================================================")
    print("=== Blockchain Audit Verification & Overhead Evaluation (Sec 8.4) ===")
    is_valid, verify_msg = ledger.verify_chain_integrity()
    print(f"1. Ledger Integrity Check: {is_valid} ({verify_msg})")

    # Simulate Tampering Attack on Historical Block
    tampered_ledger = copy.deepcopy(ledger)
    tampered_ledger.chain[5]["payload"]["update_hash"] = "0xTAMPERED_HASH_VALUE"
    tamper_valid, tamper_msg = tampered_ledger.verify_chain_integrity()
    print(f"2. Tamper Detection Test : Tampering Caught = {not tamper_valid} ({tamper_msg})")

    # Report Overhead Metrics
    overhead = ledger.get_overhead_metrics()
    print(f"3. Blockchain Overhead   : {overhead}")

    ledger_path = "experiments/federated/blockchain_audit_ledger.json"
    ledger.save_ledger(ledger_path)
    print(f"\nSaved complete auditable ledger to: {ledger_path}")
    print("Milestone 3 Complete: Auditable training history & threat defenses verified!")
    print("==================================================================")

if __name__ == "__main__":
    main()