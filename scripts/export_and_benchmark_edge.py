import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort

from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.models.spiking_head import SpikingDeepfakeDetector

class SpikingHeadWrapper(nn.Module):
    """Wraps the SNN head (fc1 + lif1 + fc2 + lif2) for standalone edge inference on 768-D features."""
    def __init__(self, snn_model: SpikingDeepfakeDetector):
        super().__init__()
        self.num_steps = snn_model.num_steps
        self.fc1 = snn_model.fc1
        self.lif1 = snn_model.lif1
        self.fc2 = snn_model.fc2
        self.lif2 = snn_model.lif2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mem1 = torch.zeros(features.size(0), self.fc1.out_features, device=features.device)
        mem2 = torch.zeros(features.size(0), self.fc2.out_features, device=features.device)
        mem2_sum = torch.zeros(features.size(0), self.fc2.out_features, device=features.device)

        for _ in range(self.num_steps):
            cur1 = self.fc1(features)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2_sum = mem2_sum + mem2

        return mem2_sum

def main():
    print("==========================================================================")
    print("=== Phase 3: Edge Model Export (ONNX) & Hardware Latency Benchmark ===")
    print("==========================================================================")

    onnx_dir = Path("checkpoints/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Federated ViT and extract its classification head
    fed_model = ViTDeepfakeDetector()
    fed_model.load_state_dict(torch.load("checkpoints/federated_global_vit.pt", map_location="cpu"))
    fed_model.eval()
    classical_head = fed_model.classifier

    # 2. Load Neuromorphic SNN and wrap its spiking head
    snn_model = SpikingDeepfakeDetector(num_steps=10)
    snn_model.load_state_dict(torch.load("checkpoints/spiking_vit.pt", map_location="cpu"))
    snn_model.eval()
    spiking_head = SpikingHeadWrapper(snn_model).eval()

    dummy_features = torch.randn(1, 768, dtype=torch.float32)

    # Export Classical Head to ONNX
    clas_onnx_path = onnx_dir / "federated_classical_head.onnx"
    torch.onnx.export(
        classical_head,
        dummy_features,
        clas_onnx_path,
        input_names=["video_features"],
        output_names=["logits"],
        dynamic_axes={"video_features": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False
    )

    # Export Spiking SNN Head to ONNX
    snn_onnx_path = onnx_dir / "neuromorphic_spiking_head.onnx"
    torch.onnx.export(
        spiking_head,
        dummy_features,
        snn_onnx_path,
        input_names=["video_features"],
        output_names=["logits"],
        dynamic_axes={"video_features": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False
    )

    # Benchmark PyTorch vs ONNX Runtime over 200 iterations
    np_input = dummy_features.numpy()
    num_runs = 200
    rows = []

    for name, pt_module, onnx_path in [
        ("Classical Head (Federated)", classical_head, clas_onnx_path),
        ("Neuromorphic Head (10-Step SNN)", spiking_head, snn_onnx_path)
    ]:
        # PyTorch latency
        with torch.no_grad():
            for _ in range(10): _ = pt_module(dummy_features)
            t0 = time.perf_counter()
            for _ in range(num_runs): _ = pt_module(dummy_features)
            pt_ms = ((time.perf_counter() - t0) * 1000.0) / num_runs

        # ONNX Runtime latency
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for _ in range(10): _ = sess.run(None, {"video_features": np_input})
        t0 = time.perf_counter()
        for _ in range(num_runs): _ = sess.run(None, {"video_features": np_input})
        ort_ms = ((time.perf_counter() - t0) * 1000.0) / num_runs

        size_kb = onnx_path.stat().st_size / 1024.0
        speedup = pt_ms / max(ort_ms, 1e-6)

        rows.append({
            "Edge_Module": name,
            "ONNX_Artifact": str(onnx_path),
            "Size_KB": round(size_kb, 2),
            "PyTorch_Latency_ms": round(pt_ms, 4),
            "ONNX_Runtime_Latency_ms": round(ort_ms, 4),
            "Speedup_Factor": f"{speedup:.2f}x"
        })

    df = pd.DataFrame(rows)
    df.to_csv("experiments/edge_onnx_benchmark.csv", index=False)
    print("\n--- Edge Deployment Benchmark Results ---")
    print(df.to_string(index=False))
    print("\nSaved benchmark to: experiments/edge_onnx_benchmark.csv")
    print("==========================================================================")

if __name__ == "__main__":
    main()
