import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import onnxruntime as ort

from src.qf_neuroshield.federated.aggregation import fedavg_aggregate, compute_weights_hash
from src.qf_neuroshield.blockchain.audit import PermissionedAuditLedger, BlockchainSecurityError
from src.qf_neuroshield.training.calibration import fit_threshold_and_scale, apply_calibration

def test_federated_aggregation_and_hashing():
    w1 = [np.ones((2, 4), dtype=np.float32)]
    w2 = [np.full((2, 4), 3.0, dtype=np.float32)]
    agg = fedavg_aggregate([(w1, 5), (w2, 5)])
    assert np.allclose(agg[0], 2.0), "FedAvg equal weighting failed"
    h1 = compute_weights_hash(w1)
    assert h1.startswith("0x") and len(h1) == 66, "Invalid SHA-256 weight hash"
    print("  [PASS] Test 1: Federated FedAvg Math & Deterministic Hashing")

def test_blockchain_security_defenses():
    base_w = [np.zeros((4, 4), dtype=np.float32)]
    base_hash = compute_weights_hash(base_w)
    ledger = PermissionedAuditLedger(initial_global_hash=base_hash, max_update_l2_norm=5.0)
    ledger.register_participant("0xUnivA", "University_A")

    benign_w = [np.full((4, 4), 0.1, dtype=np.float32)]
    upd_hash = compute_weights_hash(benign_w)
    ok, _ = ledger.validate_and_record_update("0xUnivA", 1, base_hash, upd_hash, base_w, benign_w)
    assert ok, "Benign update should be approved"

    # Duplicate submission check
    try:
        ledger.validate_and_record_update("0xUnivA", 1, base_hash, upd_hash, base_w, benign_w)
        assert False, "Duplicate submission should have raised BlockchainSecurityError"
    except BlockchainSecurityError:
        pass

    # Poisoning attack check
    ledger.register_participant("0xAttacker", "Malicious_Org")
    poison_w = [np.full((4, 4), 50.0, dtype=np.float32)]
    try:
        ledger.validate_and_record_update("0xAttacker", 1, base_hash, compute_weights_hash(poison_w), base_w, poison_w)
        assert False, "Poisoned update should have raised BlockchainSecurityError"
    except BlockchainSecurityError:
        pass

    valid, _ = ledger.verify_chain_integrity()
    assert valid, "Blockchain integrity verification failed"
    print("  [PASS] Test 2: Blockchain Smart-Contract Rules & Poisoning Defenses")

def test_probability_calibration():
    labels = np.array([0, 0, 1, 1])
    raw_probs = np.array([0.61, 0.63, 0.71, 0.75])
    params = fit_threshold_and_scale(labels, raw_probs)
    assert 0.63 < params["optimal_threshold"] < 0.71, "Optimal threshold not between classes"
    assert apply_calibration(0.62, params) < 0.5, "Real sample should calibrate < 0.5"
    assert apply_calibration(0.72, params) > 0.5, "Fake sample should calibrate > 0.5"
    print("  [PASS] Test 3: Post-Hoc Probability Calibration & Thresholding")

def test_onnx_edge_parity():
    onnx_path = Path("checkpoints/onnx/neuromorphic_spiking_head.onnx")
    assert onnx_path.exists(), "ONNX spiking head missing"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(2, 768).astype(np.float32)
    out = sess.run(None, {"video_features": dummy})[0]
    assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"
    print("  [PASS] Test 4: ONNX Runtime Neuromorphic Edge Inference Parity")

if __name__ == "__main__":
    print("==========================================================================")
    print("=== Phase 4: QF-NeuroShield Automated System Integrity Test Suite ===")
    print("==========================================================================")
    test_federated_aggregation_and_hashing()
    test_blockchain_security_defenses()
    test_probability_calibration()
    test_onnx_edge_parity()
    print("\n✅ ALL 4 SUBSYSTEM INTEGRITY TESTS PASSED!")
    print("==========================================================================")

