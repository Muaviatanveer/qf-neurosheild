# src/qf_neuroshield/blockchain/audit.py
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

class BlockchainSecurityError(Exception):
    """Raised when an update violates smart contract or threat-model rules."""
    pass

class PermissionedAuditLedger:
    def __init__(self, initial_global_hash: str, max_update_l2_norm: float = 5.0):
        self.coordinator_id = "0xCoordinator_Node_01"
        self.current_round = 0
        self.latest_global_hash = initial_global_hash
        self.max_update_l2_norm = max_update_l2_norm

        self.participants: Dict[str, str] = {}  # address -> org_name
        self.round_submissions: Dict[int, set] = {}
        self.chain: List[Dict] = []
        self.tx_latencies_ms: List[float] = []

        # Create Genesis Block (Block 0)
        self._append_block(
            tx_type="GENESIS_GLOBAL_MODEL",
            payload={
                "round": 0,
                "global_model_hash": initial_global_hash,
                "aggregation_policy": "INITIAL_W0",
                "validation_status": "APPROVED"
            }
        )

    def _compute_block_hash(self, block_data: Dict) -> str:
        block_Copy = {k: v for k, v in block_data.items() if k != "block_hash"}
        encoded = json.dumps(block_Copy, sort_keys=True).encode("utf-8")
        return "0x" + hashlib.sha256(encoded).hexdigest()

    def _append_block(self, tx_type: str, payload: Dict) -> Dict:
        t0 = time.perf_counter()
        prev_hash = self.chain[-1]["block_hash"] if self.chain else "0x" + ("0" * 64)
        block = {
            "block_index": len(self.chain),
            "timestamp": time.time(),
            "tx_type": tx_type,
            "prev_block_hash": prev_hash,
            "payload": payload
        }
        block["block_hash"] = self._compute_block_hash(block)
        self.chain.append(block)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.tx_latencies_ms.append(latency_ms)
        return block

    def register_participant(self, client_address: str, org_name: str) -> Dict:
        self.participants[client_address] = org_name
        return self._append_block(
            tx_type="REGISTER_PARTICIPANT",
            payload={"client_address": client_address, "org_name": org_name}
        )

    def check_poisoning_anomaly(
        self, base_params: List[np.ndarray], updated_params: List[np.ndarray]
    ) -> Tuple[bool, float]:
        """
        Calculates the L2 norm of the parameter delta: ||W_client - W_base||_2.
        Flags updates that exceed max_update_l2_norm as potential model poisoning.
        """
        sq_norm = 0.0
        for b_p, u_p in zip(base_params, updated_params):
            diff = u_p.astype(np.float64) - b_p.astype(np.float64)
            sq_norm += np.sum(diff ** 2)
        l2_norm = float(np.sqrt(sq_norm))
        is_benign = l2_norm <= self.max_update_l2_norm
        return is_benign, l2_norm

    def validate_and_record_update(
        self,
        client_address: str,
        round_num: int,
        base_model_hash: str,
        update_hash: str,
        base_params: List[np.ndarray],
        updated_params: List[np.ndarray]
    ) -> Tuple[bool, str]:
        """
        Enforces smart-contract rules & poisoning defense before accepting a client update.
        """
        # 1. Check Registration
        if client_address not in self.participants:
            self._append_block("REJECTED_UPDATE", {
                "round": round_num, "client": client_address, "reason": "UNAUTHORIZED_CLIENT"
            })
            raise BlockchainSecurityError(f"Client {client_address} is not registered.")

        # 2. Check Base Model Provenance (Adversarial Test Case 1)
        if base_model_hash != self.latest_global_hash:
            self._append_block("REJECTED_UPDATE", {
                "round": round_num, "client": client_address,
                "base_model_hash": base_model_hash,
                "expected_base_hash": self.latest_global_hash,
                "reason": "WRONG_BASE_MODEL_HASH"
            })
            raise BlockchainSecurityError(
                f"Base model hash mismatch! Expected {self.latest_global_hash[:12]}..., got {base_model_hash[:12]}..."
            )

        # 3. Check Duplicate Submission in Same Round (Adversarial Test Case 2)
        if round_num not in self.round_submissions:
            self.round_submissions[round_num] = set()
        if client_address in self.round_submissions[round_num]:
            self._append_block("REJECTED_UPDATE", {
                "round": round_num, "client": client_address, "reason": "DUPLICATE_SUBMISSION"
            })
            raise BlockchainSecurityError(
                f"Duplicate submission! Client {client_address} already submitted in round {round_num}."
            )

        # 4. Check Poisoning / Anomaly Detection (Adversarial Test Case 3)
        is_benign, l2_delta = self.check_poisoning_anomaly(base_params, updated_params)
        if not is_benign:
            self._append_block("REJECTED_UPDATE", {
                "round": round_num, "client": client_address,
                "update_hash": update_hash,
                "delta_l2_norm": round(l2_delta, 4),
                "reason": "MODEL_POISONING_ANOMALY_DETECTED"
            })
            raise BlockchainSecurityError(
                f"Poisoned update detected! L2 delta ({l2_delta:.2f}) exceeds threshold ({self.max_update_l2_norm})."
            )

        # Record valid update on-chain
        self.round_submissions[round_num].add(client_address)
        self._append_block("RECORD_UPDATE", {
            "round": round_num,
            "client_address": client_address,
            "org_name": self.participants[client_address],
            "base_model_hash": base_model_hash,
            "update_hash": update_hash,
            "delta_l2_norm": round(l2_delta, 4),
            "validation_status": "APPROVED"
        })
        return True, f"Approved (L2 delta: {l2_delta:.4f})"

    def record_global_model(self, round_num: int, global_model_hash: str, policy: str = "FedAvg-Robust") -> Dict:
        self.current_round = round_num
        self.latest_global_hash = global_model_hash
        return self._append_block("RECORD_GLOBAL_MODEL", {
            "round": round_num,
            "global_model_hash": global_model_hash,
            "aggregation_policy": policy,
            "validation_status": "FINALIZED"
        })

    def verify_chain_integrity(self) -> Tuple[bool, str]:
        """Verifies every block hash and previous-block link across the entire ledger."""
        for i, block in enumerate(self.chain):
            expected_hash = self._compute_block_hash(block)
            if block["block_hash"] != expected_hash:
                return False, f"Tampering detected in block #{i}: hash mismatch!"
            if i > 0 and block["prev_block_hash"] != self.chain[i - 1]["block_hash"]:
                return False, f"Broken chain link between block #{i-1} and #{i}!"
        return True, f"All {len(self.chain)} blocks cryptographically verified."

    def get_overhead_metrics(self) -> Dict:
        serialized = json.dumps(self.chain, indent=2).encode("utf-8")
        return {
            "total_blocks": len(self.chain),
            "avg_tx_latency_ms": round(float(np.mean(self.tx_latencies_ms)), 4),
            "total_ledger_size_bytes": len(serialized),
            "bytes_per_block": round(len(serialized) / max(1, len(self.chain)), 1)
        }

    def save_ledger(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.chain, f, indent=2)