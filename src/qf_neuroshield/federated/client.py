# src/qf_neuroshield/federated/client.py
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl

from src.qf_neuroshield.data.dataset import DeepfakeVideoDataset
from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.federated.aggregation import compute_weights_hash

class FlowerDeepfakeClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        client_name: str,
        dataset: DeepfakeVideoDataset,
        model_name: str,
        device: torch.device,
        batch_size: int = 2,
        lr: float = 0.0005,
        local_epochs: int = 1
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr

        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = ViTDeepfakeDetector(pretrained_name=model_name, freeze_backbone=True).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict[str, str] = None) -> List[np.ndarray]:
        """Extract trainable parameters as NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.classifier.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load global parameters into the local model head."""
        params_dict = zip(self.model.classifier.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.classifier.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the local model on the client's private partition."""
        self.set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )

        epoch_loss = 0.0
        total_batches = 0
        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                frames = batch["frames"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(frames)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_batches += 1

        updated_params = self.get_parameters()
        update_hash = compute_weights_hash(updated_params)
        num_examples = len(self.train_loader.dataset)
        avg_loss = epoch_loss / max(1, total_batches)

        metrics = {
            "client_name": self.client_name,
            "train_loss": float(avg_loss),
            "update_hash": update_hash
        }
        return updated_params, num_examples, metrics