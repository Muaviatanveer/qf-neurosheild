# src/qf_neuroshield/federated/aggregation.py
import hashlib
from typing import List, Tuple
import numpy as np

def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Computes FedAvg: W_{t+1} = sum( (n_k / N) * W_k )
    
    Args:
        results: List of tuples (client_parameters, num_examples)
    Returns:
        aggregated_parameters: List of numpy arrays representing the new global model
    """
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples == 0:
        raise ValueError("Total number of examples across clients is zero.")

    # Initialize accumulator with zeros matching the shapes of parameter arrays
    num_tensors = len(results[0][0])
    aggregated = [np.zeros_like(param, dtype=np.float32) for param in results[0][0]]

    for client_params, num_examples in results:
        weight = num_examples / total_examples
        for idx in range(num_tensors):
            aggregated[idx] += client_params[idx].astype(np.float32) * weight

    return aggregated

def compute_weights_hash(parameters: List[np.ndarray]) -> str:
    """
    Computes a deterministic SHA-256 hash of model parameters.
    This hash will be committed to the blockchain in Stage 4.
    """
    hasher = hashlib.sha256()
    for param in parameters:
        hasher.update(param.tobytes())
    return "0x" + hasher.hexdigest()