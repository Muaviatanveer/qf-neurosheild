# src/qf_neuroshield/models/quantum_head.py
import torch
import torch.nn as nn
from transformers import ViTModel

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class HybridQuantumDeepfakeDetector(nn.Module):
    def __init__(self, pretrained_name: str = "google/vit-base-patch16-224-in21k", num_qubits: int = 4, freeze_backbone: bool = True):
        super().__init__()
        
        # 1. Classical Feature Extractor
        self.backbone = ViTModel.from_pretrained(pretrained_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Classical Projection (Compress 768 -> 4 dimensions for the Qubits)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_qubits),
            nn.Tanh() # Scale inputs for stable quantum phase encoding
        )

        # 3. Variational Quantum Circuit (VQC)
        self.num_qubits = num_qubits
        qc = QuantumCircuit(num_qubits)
        
        # Data Encoding Map (Classical -> Quantum)
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement="linear")
        # Trainable Quantum Gates
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1, entanglement="linear")
        
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        # Wrap circuit in a QNN
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True
        )
        
        # Connect QNN to PyTorch
        self.qnn = TorchConnector(qnn)

        # 4. Classical Classification Head
        # The default EstimatorQNN outputs 1 expectation value
        self.classifier = nn.Linear(1, 2)

    def extract_video_features(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        flat_frames = frames.view(b * t, c, h, w)
        outputs = self.backbone(pixel_values=flat_frames)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        frame_features = cls_tokens.view(b, t, -1)
        video_features = frame_features.mean(dim=1)
        return video_features

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Extract features natively on device (MPS/CUDA)
        features = self.extract_video_features(frames)
        
        # Project down to qubit dimensions
        reduced_features = self.projection(features)
        
        # Qiskit simulator runs on CPU; safely transfer tensors
        device = reduced_features.device
        quantum_in = reduced_features.cpu()
        
        # Execute Quantum Circuit
        quantum_out = self.qnn(quantum_in)
        
        # Transfer back to device and classify
        quantum_out = quantum_out.to(device)
        logits = self.classifier(quantum_out)
        
        return logits