# src/qf_neuroshield/federated/blockchain_strategy.py
import flwr as fl
from flwr.common import Parameters, FitRes, FitIns
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from src.qf_neuroshield.federated.aggregation import compute_weights_hash
from src.qf_neuroshield.blockchain.audit import PermissionedAuditLedger, BlockchainSecurityError

class BlockchainFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, ledger: PermissionedAuditLedger, base_params: List[np.ndarray], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = ledger
        self.base_params = base_params # Stores W_t for L2 comparison

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        
        print(f"\n[Blockchain Strategy] Intercepting weights for Round {server_round}...")
        
        approved_results = []
        base_hash = self.ledger.latest_global_hash

        for client, fit_res in results:
            # Convert Flower Parameters back to NumPy arrays
            updated_params = [np.array(p) for p in fl.common.parameters_to_ndarrays(fit_res.parameters)]
            update_hash = compute_weights_hash(updated_params)
            
            # The client sends its ID and Org Name in the metrics dictionary
            client_id = fit_res.metrics.get("client_id", "Unknown")
            org_name = fit_res.metrics.get("org_name", "Unknown")
            
            try:
                # 1. Blockchain validation & Poisoning Check
                ok, msg = self.ledger.validate_and_record_update(
                    client_address=client_id,
                    round_num=server_round,
                    base_model_hash=base_hash,
                    update_hash=update_hash,
                    base_params=self.base_params,
                    updated_params=updated_params
                )
                if ok:
                    print(f"  ✅ [APPROVED] {org_name} | Hash: {update_hash[:10]}... | {msg}")
                    approved_results.append((client, fit_res))
            except BlockchainSecurityError as e:
                print(f"  ❌ [REJECTED] {org_name} dropped by Smart Contract: {e}")

        # 2. Pass ONLY approved updates to standard FedAvg
        aggregated_parameters, metrics = super().aggregate_fit(server_round, approved_results, failures)
        
        if aggregated_parameters is not None:
            # 3. Hash and record the new global model
            new_global_params = [np.array(p) for p in fl.common.parameters_to_ndarrays(aggregated_parameters)]
            new_global_hash = compute_weights_hash(new_global_params)
            self.ledger.record_global_model(server_round, new_global_hash)
            self.base_params = new_global_params # Update base for next round
            
            print(f"  🌐 [ROUND {server_round} COMPLETE] New Global Model Hash: {new_global_hash[:15]}...")
            
        return aggregated_parameters, metrics